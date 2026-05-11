"""Swarm multi-agent system: diverse independent generation then merging.

Multiple agents work on the same problem independently, each with a different
perspective. A merger agent consolidates the best ideas into a single answer.

Implements Pattern 4 from:
docs/02-the-agent-loop/04-multi-agent-patterns.md

Run:
    python swarm_agent.py
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


_MERGER_SYSTEM = """\
You are a synthesizer. You have received multiple responses to the same task,
each written from a different perspective.

Your job:
1. Identify the best ideas and insights from each response.
2. Remove or consolidate duplicates (keep the best-worded version).
3. Highlight unique, valuable contributions that only one agent mentioned.
4. Produce a single, consolidated answer that is more complete than any
   individual response.

Do NOT average the responses. Select and combine the strongest elements.
Structure the output clearly. Do not mention the individual agents or
their perspectives."""

_DEFAULT_PERSPECTIVES = [
    "Focus on practicality and real-world usability.",
    "Focus on innovation, novelty, and cutting-edge ideas.",
    "Focus on simplicity, elegance, and user-friendliness.",
    "Focus on scalability, technical excellence, and performance.",
    "Focus on cost-effectiveness and business viability.",
    "Focus on aesthetics, design quality, and user experience.",
]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """One agent's independent response."""

    agent_index: int
    perspective: str
    response: str


@dataclass
class SwarmResult:
    """Full output of a swarm run."""

    merged_answer: str
    individual_responses: list[AgentResponse]
    unique_ideas_count: int
    diversity_score: float  # 0.0-1.0: fraction of responses that are distinct

    def to_dict(self) -> dict[str, Any]:
        return {
            "merged_answer": self.merged_answer,
            "individual_responses": [
                {
                    "agent": r.agent_index,
                    "perspective": r.perspective,
                    "response": r.response,
                }
                for r in self.individual_responses
            ],
            "unique_ideas_count": self.unique_ideas_count,
            "diversity_score": round(self.diversity_score, 2),
        }


# ---------------------------------------------------------------------------
# SwarmAgent
# ---------------------------------------------------------------------------


class SwarmAgent:
    """Run the same task through multiple independent agents, then merge.

    Args:
        swarm_size:   Number of parallel agents.
        model:        OpenAI model for all agents.
        perspectives: List of perspective strings to cycle through.
                      Falls back to _DEFAULT_PERSPECTIVES.
    """

    def __init__(
        self,
        swarm_size: int = 4,
        model: str = "gpt-4o",
        perspectives: Optional[list[str]] = None,
    ) -> None:
        self.swarm_size = swarm_size
        self.model = model
        self.perspectives = perspectives or _DEFAULT_PERSPECTIVES
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
        """Execute the swarm and return merged results.

        Args:
            task: The creative or generative task to run.

        Returns:
            Dict with ``merged_answer``, ``individual_responses``,
            ``unique_ideas_count``, and ``diversity_score``.
        """
        # Phase 1: diverse parallel generation
        responses = self._generate_diverse(task)

        # Phase 2: merge
        merged = self._merge(task, responses)

        # Phase 3: quality metrics
        unique_count = self._count_unique_ideas(responses)
        diversity = self._diversity_score(responses)

        result = SwarmResult(
            merged_answer=merged,
            individual_responses=responses,
            unique_ideas_count=unique_count,
            diversity_score=diversity,
        )
        return result.to_dict()

    # ------------------------------------------------------------------
    # Phase 1: Diverse Generation
    # ------------------------------------------------------------------

    def _generate_diverse(self, task: str) -> list[AgentResponse]:
        """Run all agents in parallel; each gets a different perspective."""
        assignments = [
            (i, self.perspectives[i % len(self.perspectives)])
            for i in range(self.swarm_size)
        ]

        responses: list[AgentResponse] = []
        with ThreadPoolExecutor(max_workers=self.swarm_size) as pool:
            futures = {
                pool.submit(self._run_single, task, idx, perspective): idx
                for idx, perspective in assignments
            }
            for future in as_completed(futures):
                result = future.result()
                responses.append(result)

        responses.sort(key=lambda r: r.agent_index)
        return responses

    def _run_single(self, task: str, agent_index: int, perspective: str) -> AgentResponse:
        """One agent's independent run."""
        system_prompt = (
            f"You are a creative and knowledgeable assistant. "
            f"{perspective} "
            "Be specific, concrete, and original. Avoid generic platitudes."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ],
            temperature=0.8,
        )
        return AgentResponse(
            agent_index=agent_index,
            perspective=perspective,
            response=response.choices[0].message.content or "",
        )

    # ------------------------------------------------------------------
    # Phase 2: Merge
    # ------------------------------------------------------------------

    def _merge(self, task: str, responses: list[AgentResponse]) -> str:
        """Consolidate all individual responses into a single answer."""
        separator = "\n" + "─" * 40 + "\n"
        formatted = separator.join(
            f"[Agent {r.agent_index} — {r.perspective}]\n{r.response}"
            for r in responses
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _MERGER_SYSTEM},
                {
                    "role": "user",
                    "content": f"Task: {task}\n\nResponses:\n{formatted}",
                },
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # Phase 3: Quality metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _count_unique_ideas(responses: list[AgentResponse]) -> int:
        """Count approximately unique sentences across all responses.

        Uses a simple normalised-text deduplication heuristic: strip
        punctuation, lower-case, split on sentence boundaries, then deduplicate.
        """
        seen: set[str] = set()
        total = 0
        for r in responses:
            sentences = r.response.replace("\n", " ").split(".")
            for sent in sentences:
                normalised = sent.strip().lower()[:80]
                if len(normalised) > 15 and normalised not in seen:
                    seen.add(normalised)
                    total += 1
        return total

    @staticmethod
    def _diversity_score(responses: list[AgentResponse]) -> float:
        """Estimate diversity as the fraction of responses that differ meaningfully.

        Uses character-set overlap between consecutive pairs. A higher score
        means the agents produced more distinct content.
        """
        if len(responses) < 2:
            return 1.0
        scores = []
        for i in range(len(responses) - 1):
            a = set(responses[i].response.lower().split())
            b = set(responses[i + 1].response.lower().split())
            overlap = len(a & b) / max(len(a | b), 1)
            scores.append(1.0 - overlap)
        return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    agent = SwarmAgent(swarm_size=4)
    task = "Generate innovative feature ideas for a smart home app in 2026"

    print(f"\n{'=' * 60}")
    print(f"Task: {task}")
    print(f"Swarm size: {agent.swarm_size}")
    print("=" * 60)

    result = agent.run(task)

    print("\n--- Individual Agent Responses ---")
    for resp in result["individual_responses"]:
        preview = resp["response"][:200].replace("\n", " ")
        print(f"\n  [Agent {resp['agent']}] {resp['perspective']}")
        print(f"  {preview}…")

    print(f"\n--- Swarm Metrics ---")
    print(f"  Unique ideas: {result['unique_ideas_count']}")
    print(f"  Diversity score: {result['diversity_score']:.2f} (0=identical, 1=fully distinct)")

    print(f"\n{'=' * 60}")
    print("MERGED ANSWER:")
    print("=" * 60)
    print(result["merged_answer"])


if __name__ == "__main__":
    main()
