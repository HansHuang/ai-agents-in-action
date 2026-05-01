"""Debate multi-agent system: Generator and Critic improve through adversarial review.

Two agents collaborate adversarially: one generates, the other challenges.
The generator revises its answer based on the critique. The loop continues
until the critic is satisfied or max_rounds is reached.

Implements Pattern 2 from:
docs/02-the-agent-loop/04-multi-agent-patterns.md

Run:
    python debate_agent.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

_NO_ISSUES_SIGNAL = "NO_ISSUES"


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_GENERATOR_SYSTEM = """\
You are an expert consultant producing high-quality, actionable answers.
Be thorough, specific, and concrete. Include data and examples where possible.
Structure your response with clear headings when the topic warrants it.

When you receive a critique, revise your answer to address EVERY point raised.
Do not defend your original answer — improve it. Do not mention the critique
process in your revised answer; just produce the better version."""

_CRITIC_SYSTEM = """\
You are a rigorous reviewer. Your job is to find genuine flaws in an answer.

Check for:
1. Logical errors or contradictions
2. Missing edge cases or important considerations
3. Weak or unsupported assumptions
4. Implementation gaps (things that sound good but won't work)
5. Anything a well-informed challenger would push back on

Be specific: for every flaw, say exactly what is wrong AND how to fix it.

If the answer has no significant flaws, respond with exactly:
NO_ISSUES: <brief explanation of why the answer is solid>

Do not praise the answer unnecessarily. Only report issues that would
meaningfully improve it."""


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class RoundRecord:
    """Captures one full generate-and-critique round."""

    round: int
    generator_output: str
    critic_feedback: str
    critic_satisfied: bool


@dataclass
class DebateResult:
    """The final output of a debate session."""

    final_answer: str
    rounds_completed: int
    history: list[RoundRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_answer": self.final_answer,
            "rounds_completed": self.rounds_completed,
            "history": [
                {
                    "round": r.round,
                    "generator": r.generator_output,
                    "critic": r.critic_feedback,
                    "critic_satisfied": r.critic_satisfied,
                }
                for r in self.history
            ],
        }


# ---------------------------------------------------------------------------
# DebateSystem
# ---------------------------------------------------------------------------


class DebateSystem:
    """Adversarial collaboration between a Generator and a Critic agent.

    Args:
        max_rounds:         Maximum critique-and-revise cycles.
        model:              OpenAI model for both agents.
        similarity_cutoff:  Character-overlap ratio above which the answer is
                            considered unchanged (early stop).
    """

    def __init__(
        self,
        max_rounds: int = 3,
        model: str = "gpt-4o",
        similarity_cutoff: float = 0.95,
    ) -> None:
        self.max_rounds = max_rounds
        self.model = model
        self.similarity_cutoff = similarity_cutoff
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, task: str) -> dict[str, Any]:
        """Run the Generator → Critic → Revise loop.

        Args:
            task: The question or task to debate.

        Returns:
            A dict with ``final_answer``, ``rounds_completed``, and ``history``.
        """
        # Maintain separate histories for each agent
        gen_messages = [
            {"role": "system", "content": _GENERATOR_SYSTEM},
            {"role": "user", "content": task},
        ]
        critic_messages = [
            {"role": "system", "content": _CRITIC_SYSTEM},
        ]

        history: list[RoundRecord] = []
        previous_answer = ""

        # Round 0: initial generation (no critique yet)
        logger.info("[debate] Generating initial answer")
        current_answer = self._generate(gen_messages)
        gen_messages.append({"role": "assistant", "content": current_answer})

        for round_num in range(1, self.max_rounds + 1):
            logger.info("[debate] Round %d/%d", round_num, self.max_rounds)

            # Critic evaluates current answer
            critique = self._critique(critic_messages, task, current_answer)
            critic_messages.append({
                "role": "user",
                "content": f"Evaluate this answer:\n\n{current_answer}",
            })
            critic_messages.append({"role": "assistant", "content": critique})

            satisfied = critique.strip().upper().startswith(_NO_ISSUES_SIGNAL)

            history.append(RoundRecord(
                round=round_num,
                generator_output=current_answer,
                critic_feedback=critique,
                critic_satisfied=satisfied,
            ))

            if satisfied:
                logger.info("[debate] Critic satisfied at round %d", round_num)
                break

            # Generator revises
            previous_answer = current_answer
            gen_messages.append({
                "role": "user",
                "content": (
                    f"Here is a critique of your answer:\n\n{critique}\n\n"
                    "Revise your answer to address every point raised."
                ),
            })
            current_answer = self._generate(gen_messages)
            gen_messages.append({"role": "assistant", "content": current_answer})

            # Early stop: answer hasn't changed meaningfully
            if self._similarity(previous_answer, current_answer) >= self.similarity_cutoff:
                logger.info("[debate] Answer unchanged (similarity=%.2f); stopping", self.similarity_cutoff)
                break

        result = DebateResult(
            final_answer=current_answer,
            rounds_completed=len(history),
            history=history,
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Internal LLM calls
    # ------------------------------------------------------------------

    def _generate(self, messages: list[dict]) -> str:
        """Ask the generator for its (revised) answer."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7,
        )
        return response.choices[0].message.content or ""

    def _critique(self, messages: list[dict], task: str, answer: str) -> str:
        """Ask the critic to evaluate the current answer."""
        eval_messages = messages + [
            {
                "role": "user",
                "content": (
                    f"Original task:\n{task}\n\n"
                    f"Answer to evaluate:\n{answer}"
                ),
            }
        ]
        response = self.client.chat.completions.create(
            model=self.model,
            messages=eval_messages,
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Approximate character-level overlap ratio between two strings.

        This is a lightweight heuristic (not semantic similarity). It's used
        only as a cheap early-stop guard when the answer barely changes.
        """
        if not a or not b:
            return 0.0
        shorter, longer = sorted([a, b], key=len)
        matches = sum(c in longer for c in shorter)
        return matches / len(longer)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    system = DebateSystem(max_rounds=3)
    task = "Design a go-to-market strategy for an AI-powered email client"

    print(f"\n{'=' * 60}")
    print(f"Task: {task}")
    print("=" * 60)

    result = system.run(task)

    print(f"\n--- Debate History ({result['rounds_completed']} round(s)) ---")
    for entry in result["history"]:
        r = entry["round"]
        gen_preview = entry["generator"][:200].replace("\n", " ")
        critic_preview = entry["critic"][:150].replace("\n", " ")
        satisfied_label = " [SATISFIED]" if entry["critic_satisfied"] else ""
        print(f"\n  Round {r}:")
        print(f"  Generator: {gen_preview}…")
        print(f"  Critic{satisfied_label}: {critic_preview}…")

    print(f"\n{'=' * 60}")
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result["final_answer"])


if __name__ == "__main__":
    main()
