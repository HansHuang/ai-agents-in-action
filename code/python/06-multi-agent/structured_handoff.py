"""Pattern B: Structured Handoff — agents communicate via explicit data contracts.

In the structured-handoff pattern, agents do NOT share a single message list.
Instead, each agent receives a Handoff object containing only the information
it explicitly needs. The receiving agent validates the handoff structure before
acting, ensuring clear boundaries and preventing context pollution.

This file demonstrates the pattern with a coordinator → specialist → coordinator
pipeline.

See also: shared_history.py for the simpler-but-riskier alternative.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Handoff data contract
# ---------------------------------------------------------------------------


@dataclass
class Handoff:
    """A structured, validated message passed between agents.

    Only the fields listed here cross agent boundaries. The receiving agent
    has no access to the sending agent's internal message history.

    Fields:
        from_agent:  Name of the delegating agent.
        to_agent:    Name of the receiving agent.
        task:        A complete, self-contained description of the work.
        context:     Structured data explicitly shared with the receiver.
                     This is the ONLY information crossing the boundary.
        reply_to:    Optional return address for the result.
    """

    from_agent: str
    to_agent: str
    task: str
    context: dict[str, Any] = field(default_factory=dict)
    reply_to: Optional[str] = None

    def validate(self) -> None:
        """Raise ValueError if the handoff is structurally invalid."""
        if not self.from_agent.strip():
            raise ValueError("Handoff.from_agent must not be empty")
        if not self.to_agent.strip():
            raise ValueError("Handoff.to_agent must not be empty")
        if not self.task.strip():
            raise ValueError("Handoff.task must not be empty")

    def to_user_message(self) -> str:
        """Format the handoff as a user message for the receiving agent."""
        lines = [f"Task from {self.from_agent}: {self.task}"]
        if self.context:
            lines.append("\nContext:")
            for k, v in self.context.items():
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)


@dataclass
class HandoffResult:
    """The structured result returned by a specialist after completing a Handoff."""

    from_agent: str
    status: str  # "complete" | "failed" | "need_clarification"
    result: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# StructuredHandoffAgent
# ---------------------------------------------------------------------------


class StructuredHandoffAgent:
    """An agent that ONLY communicates via Handoff objects.

    Its internal message history is strictly private. It never sees another
    agent's conversation.

    Args:
        name:          Agent identifier.
        system_prompt: System prompt for this agent.
    """

    def __init__(self, name: str, system_prompt: str) -> None:
        self.name = name
        self.system_prompt = system_prompt
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        return self._client

    def receive(self, handoff: Handoff) -> HandoffResult:
        """Validate and act on an incoming handoff.

        The agent's internal messages are NOT shared with the caller.
        Only the HandoffResult crosses the boundary back.

        Args:
            handoff: The structured request from another agent.

        Returns:
            A HandoffResult with status and result string.
        """
        try:
            handoff.validate()
        except ValueError as exc:
            return HandoffResult(
                from_agent=self.name,
                status="failed",
                result="",
                error=f"Invalid handoff: {exc}",
            )

        # Private message history — never leaked outside this method
        private_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": handoff.to_user_message()},
        ]

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=private_messages,
            temperature=0.3,
        )
        answer = response.choices[0].message.content or ""
        return HandoffResult(from_agent=self.name, status="complete", result=answer)


# ---------------------------------------------------------------------------
# Demo: coordinator → specialist → coordinator with explicit boundaries
# ---------------------------------------------------------------------------


def run_structured_handoff_demo() -> dict[str, Any]:
    """Coordinator delegates to a specialist and receives a clean result."""
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    specialist = StructuredHandoffAgent(
        name="research_specialist",
        system_prompt="""\
You are a research specialist. Your job is to answer structured research tasks.
Be concise and factual. 3-5 bullet points per task.""",
    )

    # Step 1: Coordinator determines what to delegate
    coordinator_messages = [
        {
            "role": "system",
            "content": (
                "You are a coordinator. You receive a user goal and decide what "
                "information to collect by delegating to a research_specialist. "
                "Respond with a specific, self-contained research task in one sentence."
            ),
        },
        {
            "role": "user",
            "content": "Goal: Assess whether we should adopt TypeScript for our next project.",
        },
    ]
    coord_resp = client.chat.completions.create(
        model="gpt-4o", messages=coordinator_messages, temperature=0
    )
    research_task = coord_resp.choices[0].message.content or ""

    # Step 2: Build a structured handoff — only explicit context crosses the boundary
    handoff = Handoff(
        from_agent="coordinator",
        to_agent="research_specialist",
        task=research_task,
        context={
            "domain": "enterprise software",
            "constraints": "existing team knows JavaScript",
        },
        reply_to="coordinator",
    )
    handoff.validate()

    # Step 3: Specialist receives only the handoff — no coordinator history visible
    result = specialist.receive(handoff)

    # Step 4: Coordinator synthesises using only the clean HandoffResult
    coordinator_messages.append({"role": "assistant", "content": research_task})
    coordinator_messages.append({
        "role": "user",
        "content": (
            f"The research_specialist returned:\n{result.result}\n\n"
            "Based solely on this, give a one-paragraph recommendation."
        ),
    })
    final_resp = client.chat.completions.create(
        model="gpt-4o", messages=coordinator_messages, temperature=0.3
    )
    recommendation = final_resp.choices[0].message.content or ""

    return {
        "delegated_task": research_task,
        "handoff": {
            "from": handoff.from_agent,
            "to": handoff.to_agent,
            "context_keys": list(handoff.context.keys()),
        },
        "specialist_result": result.result,
        "specialist_status": result.status,
        "coordinator_recommendation": recommendation,
        "boundary_note": (
            "The specialist never saw the coordinator's system prompt or user goal. "
            "The coordinator never saw the specialist's internal message history. "
            "Only the Handoff and HandoffResult objects crossed the boundary."
        ),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = run_structured_handoff_demo()

    print(f"\nDelegated task: {result['delegated_task']}")
    print(f"Handoff context keys: {result['handoff']['context_keys']}")
    print(f"\nSpecialist result:\n{result['specialist_result']}")
    print(f"\nCoordinator recommendation:\n{result['coordinator_recommendation']}")
    print(f"\n✓ Boundary: {result['boundary_note']}")


if __name__ == "__main__":
    main()
