"""Pattern A: Shared History — all agents append to a single message list.

In the shared-history pattern, every agent reads and writes to the same
conversation thread. This is the simplest approach but has a critical risk:
CONTEXT POLLUTION. Every agent sees all previous agents' internal reasoning,
which can distort their own outputs and rapidly exhaust the context window.

This file demonstrates the pattern and its failure mode.

See also: structured_handoff.py for the safer alternative.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

# WARNING: Shared history causes context pollution. Each agent inherits all
# previous internal reasoning, which can bias responses or exceed the context
# window. Prefer structured_handoff.py for production systems.


class SharedHistoryAgent:
    """A simple agent that reads from and writes to a shared message list.

    Args:
        name:          Agent identifier shown in logs.
        role:          One-sentence role description.
        system_prompt: Full system prompt for this agent.
    """

    def __init__(self, name: str, role: str, system_prompt: str) -> None:
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        return self._client

    def act(self, shared_messages: list[dict], user_request: str) -> str:
        """Append to shared history and return a response.

        Side effect: appends user and assistant messages to ``shared_messages``.

        Args:
            shared_messages: The shared conversation history (mutated in place).
            user_request:    The task this agent should perform.

        Returns:
            The agent's response string.
        """
        # WARNING: The agent sees ALL previous messages, including the internal
        # reasoning of every previous agent. This is the context pollution risk.
        messages = [
            {"role": "system", "content": self.system_prompt},
            *shared_messages,
            {"role": "user", "content": user_request},
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content or ""

        # Both the user request and the response land in the shared history,
        # visible to every subsequent agent.
        shared_messages.append({"role": "user", "content": f"[{self.name}] {user_request}"})
        shared_messages.append({"role": "assistant", "content": answer})
        return answer


# ---------------------------------------------------------------------------
# Demo: two-agent pipeline demonstrating context pollution risk
# ---------------------------------------------------------------------------


def run_shared_history_demo() -> dict[str, Any]:
    """Two-agent pipeline sharing a single message list."""
    shared_history: list[dict] = []

    researcher = SharedHistoryAgent(
        name="researcher",
        role="Collects raw information about a topic",
        system_prompt="""\
You are a research assistant. Gather and present relevant facts about the topic
you are asked about. Be concise — 3-5 bullet points.""",
    )

    analyst = SharedHistoryAgent(
        name="analyst",
        role="Analyses and summarises research findings",
        system_prompt="""\
You are a business analyst. Analyse the research you have been given and produce
a concise executive summary with one key recommendation.

NOTE: You can see all previous messages in this conversation, including the
researcher's internal work. This is the shared-history pattern. Be aware that
large amounts of context may distort your analysis.""",
    )

    topic = "The impact of large language models on enterprise software in 2025"

    print("[shared_history] Step 1: Researcher gathers information")
    research_output = researcher.act(shared_history, f"Research: {topic}")

    print("[shared_history] Step 2: Analyst receives ALL shared context")
    print(f"  (Shared history now has {len(shared_history)} message(s) visible to the analyst)")
    analyst_output = analyst.act(
        shared_history,
        "Summarise the research above and provide one key recommendation.",
    )

    return {
        "research": research_output,
        "analysis": analyst_output,
        "shared_history_length": len(shared_history),
        "pollution_warning": (
            "The analyst received all researcher messages including any internal "
            "reasoning. In larger pipelines this can bias agents and exhaust the "
            "context window. Consider structured_handoff.py instead."
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_shared_history_demo()

    print(f"\nResearch output:\n{result['research']}")
    print(f"\nAnalysis output:\n{result['analysis']}")
    print(f"\nShared history messages: {result['shared_history_length']}")
    print(f"\n⚠ Warning: {result['pollution_warning']}")


if __name__ == "__main__":
    main()
