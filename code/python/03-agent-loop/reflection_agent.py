"""Reflection agent: Generate → Reflect → Revise loop.

Implements the self-critique pattern described in:
docs/02-the-agent-loop/03-planning-strategies.md

The agent generates an initial answer, then evaluates it with a separate
critic call. If the score is below the threshold, it revises the answer and
critiques again, up to max_reflections times.

Run:
    python reflection_agent.py
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CritiqueResult(BaseModel):
    """Structured output from the critic LLM call."""

    overall_score: int = Field(..., ge=1, le=10)
    is_satisfied: bool
    feedback: str = Field(..., min_length=10)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class IterationRecord(BaseModel):
    """A single generate-or-revise iteration with its critique."""

    iteration: int
    answer: str
    critique: Optional[CritiqueResult] = None


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

_GENERATOR_SYSTEM = """\
Answer the user's question thoroughly and accurately.
Provide specific details and cite sources or data where possible.
Structure your response clearly with headings if the topic warrants it."""

_CRITIC_SYSTEM = """\
You are a strict quality reviewer. Evaluate the answer against the original question.

Score on 1-10 for:
1. Completeness: Did it answer everything asked?
2. Accuracy: Are there factual errors or unsupported claims?
3. Clarity: Is it easy to understand?
4. Structure: Is it well-organised with appropriate formatting?
5. Actionability: Can the user act on this information?

Output ONLY valid JSON with this exact schema (no markdown fences):
{
  "overall_score": <int 1-10>,
  "is_satisfied": <bool>,
  "feedback": "<specific, actionable critique>",
  "strengths": ["<strength 1>", "..."],
  "weaknesses": ["<weakness 1>", "..."]
}

Set is_satisfied=true when overall_score >= 8 and there are no major weaknesses.
Be honest and specific. Vague feedback like 'good answer' is not acceptable."""

_REVISER_SYSTEM = """\
You are revising your previous answer based on a critic's feedback.
Address every weakness listed in the critique.
Keep all the strengths from the original.
Do not acknowledge the critique in your revised answer — just improve it."""


# ---------------------------------------------------------------------------
# ReflectionAgent
# ---------------------------------------------------------------------------


class ReflectionAgent:
    """Generate → Reflect → Revise agent with structured critique.

    Args:
        model:              OpenAI model for all LLM calls.
        max_reflections:    Maximum critique-and-revise rounds.
        quality_threshold:  Minimum score (1-10) to accept an answer without further revision.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        max_reflections: int = 2,
        quality_threshold: int = 7,
    ) -> None:
        self.model = model
        self.max_reflections = max_reflections
        self.quality_threshold = quality_threshold
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> dict[str, Any]:
        """Run the Generate → Reflect → Revise loop.

        Args:
            user_input: The user's question or request.

        Returns:
            A dict with keys:
                ``final_answer``      — the best answer produced
                ``iterations``        — list of IterationRecord dicts
                ``reflections_used``  — number of reflection rounds performed
        """
        iterations: list[IterationRecord] = []

        # Phase 1: Generate initial answer
        logger.info("[Reflection] Generating initial answer")
        answer = self._generate(user_input)
        iterations.append(IterationRecord(iteration=1, answer=answer))

        reflections_used = 0
        for round_num in range(1, self.max_reflections + 1):
            # Phase 2: Reflect
            logger.info("[Reflection] Reflection round %d/%d", round_num, self.max_reflections)
            critique = self._reflect(user_input, answer)
            iterations[-1] = IterationRecord(
                iteration=iterations[-1].iteration,
                answer=answer,
                critique=critique,
            )

            if critique.is_satisfied or critique.overall_score >= self.quality_threshold:
                logger.info(
                    "[Reflection] Satisfied at round %d (score=%d)",
                    round_num,
                    critique.overall_score,
                )
                break

            # Phase 3: Revise
            logger.info(
                "[Reflection] Score %d < threshold %d; revising",
                critique.overall_score,
                self.quality_threshold,
            )
            answer = self._revise(user_input, answer, critique.feedback)
            reflections_used = round_num
            iterations.append(
                IterationRecord(iteration=round_num + 1, answer=answer)
            )

        return {
            "final_answer": answer,
            "iterations": [it.model_dump() for it in iterations],
            "reflections_used": reflections_used,
        }

    # ------------------------------------------------------------------
    # Phases
    # ------------------------------------------------------------------

    def _generate(self, user_input: str) -> str:
        """Generate an initial answer."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _GENERATOR_SYSTEM},
                {"role": "user", "content": user_input},
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content or ""

    def _reflect(self, user_input: str, answer: str) -> CritiqueResult:
        """Evaluate the answer and return a structured critique.

        Falls back to a default satisfied critique if the LLM output cannot
        be parsed, to avoid crashing the agent on a malformed JSON response.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _CRITIC_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Original question:\n{user_input}\n\n"
                        f"Answer to review:\n{answer}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        try:
            return CritiqueResult.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("[Reflection] Critique parsing failed (%s); using fallback", exc)
            return CritiqueResult(
                overall_score=8,
                is_satisfied=True,
                feedback="Unable to parse structured critique; assuming answer is acceptable.",
                strengths=[],
                weaknesses=[],
            )

    def _revise(self, user_input: str, previous_answer: str, feedback: str) -> str:
        """Produce a revised answer that addresses the critique."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _REVISER_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Original question:\n{user_input}\n\n"
                        f"Previous answer:\n{previous_answer}\n\n"
                        f"Critique feedback:\n{feedback}"
                    ),
                },
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content or previous_answer


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    agent = ReflectionAgent(max_reflections=2, quality_threshold=8)
    question = (
        "Explain how transformers work in LLMs. Include their advantages over RNNs, "
        "the role of attention mechanisms, and real-world applications."
    )

    print(f"\n{'=' * 60}")
    print(f"Question: {question}")
    print("=" * 60)

    result = agent.run(question)

    print(f"\nReflections used: {result['reflections_used']}")
    for iteration in result["iterations"]:
        n = iteration["iteration"]
        answer_preview = iteration["answer"][:200].replace("\n", " ")
        print(f"\n--- Iteration {n} ---")
        print(f"Answer (first 200 chars): {answer_preview}…")
        if iteration.get("critique"):
            c = iteration["critique"]
            print(f"Score: {c['overall_score']}/10  |  Satisfied: {c['is_satisfied']}")
            print(f"Feedback: {c['feedback'][:150]}…")

    print(f"\n{'=' * 60}")
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result["final_answer"])


if __name__ == "__main__":
    main()
