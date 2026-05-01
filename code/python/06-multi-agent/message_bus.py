"""Pattern C: Message Bus — agents communicate via a publish/subscribe bus.

In the message-bus pattern, agents do not call each other directly. Instead,
they publish messages to named topics, and any agent subscribed to that topic
receives them. This decouples producers from consumers and enables fan-out.

This implementation runs in-process (no real threading). It is designed to
demonstrate the pattern clearly without the complexity of actual concurrency.

Topics used:
  tasks.research  — new research tasks to be handled by the research agent
  tasks.analysis  — data ready for the analysis agent
  results.final   — completed results to be collected

Run:
    python message_bus.py
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 30.0


# ---------------------------------------------------------------------------
# Message and Bus
# ---------------------------------------------------------------------------


@dataclass
class BusMessage:
    """A message published to the bus.

    Fields:
        topic:      The topic channel (e.g. "tasks.research").
        sender:     Name of the publishing agent.
        payload:    Arbitrary data; should be JSON-serialisable.
        message_id: Unique identifier assigned by the bus.
        timestamp:  Unix time when the message was published.
    """

    topic: str
    sender: str
    payload: Any
    message_id: str = ""
    timestamp: float = field(default_factory=time.time)


Handler = Callable[[BusMessage], None]


class AgentBus:
    """In-process publish/subscribe message bus.

    Agents can publish to any topic and subscribe to receive messages.
    Messages are delivered synchronously in the order they are published.
    """

    def __init__(self) -> None:
        self._subscriptions: dict[str, list[Handler]] = {}
        self._message_counter = 0
        self._log: list[BusMessage] = []

    def subscribe(self, topic: str, handler: Handler) -> None:
        """Register a handler for a topic (exact match or prefix with '*').

        Args:
            topic:   The topic to subscribe to. Use ``"results.*"`` to
                     receive all topics that start with ``"results."``.
            handler: Callable called with each matching BusMessage.
        """
        self._subscriptions.setdefault(topic, []).append(handler)

    def publish(self, topic: str, sender: str, payload: Any) -> str:
        """Publish a message to a topic and deliver it to all subscribers.

        Args:
            topic:   Target topic channel.
            sender:  Identifier of the publishing agent.
            payload: Data to include in the message.

        Returns:
            The generated message_id.
        """
        self._message_counter += 1
        msg = BusMessage(
            topic=topic,
            sender=sender,
            payload=payload,
            message_id=f"msg-{self._message_counter:04d}",
        )
        self._log.append(msg)
        logger.info("[bus] %s → %s (from %s)", msg.message_id, topic, sender)

        for pattern, handlers in self._subscriptions.items():
            if self._matches(pattern, topic):
                for handler in handlers:
                    handler(msg)

        return msg.message_id

    def request(
        self,
        target_topic: str,
        sender: str,
        payload: Any,
        reply_topic: str,
        timeout_seconds: float = _TIMEOUT_SECONDS,
    ) -> Optional[BusMessage]:
        """Publish a message and wait for a reply on ``reply_topic``.

        In this in-process implementation, the reply is returned synchronously
        if the subscriber handles the message within the same call stack.

        Args:
            target_topic:    The topic to publish the request to.
            sender:          Identifier of the requesting agent.
            payload:         Request data.
            reply_topic:     The topic to listen to for the reply.
            timeout_seconds: Maximum seconds to wait.

        Returns:
            The reply BusMessage, or None if no reply arrived in time.
        """
        reply: list[BusMessage] = []
        deadline = time.monotonic() + timeout_seconds

        def _capture(msg: BusMessage) -> None:
            reply.append(msg)

        self._subscriptions.setdefault(reply_topic, []).insert(0, _capture)
        self.publish(target_topic, sender, payload)

        # In in-process mode, delivery is synchronous, so the reply is
        # already populated after publish() returns.
        self._subscriptions[reply_topic].remove(_capture)
        if reply:
            return reply[0]

        # Graceful timeout for genuinely async scenarios
        while not reply and time.monotonic() < deadline:
            time.sleep(0.01)
        return reply[0] if reply else None

    def message_log(self) -> list[dict[str, Any]]:
        """Return all published messages as dicts for introspection."""
        return [
            {
                "id": m.message_id,
                "topic": m.topic,
                "sender": m.sender,
                "payload_preview": str(m.payload)[:80],
            }
            for m in self._log
        ]

    @staticmethod
    def _matches(pattern: str, topic: str) -> bool:
        """Return True if ``pattern`` matches ``topic``.

        Supports one wildcard: ``"results.*"`` matches any topic starting
        with ``"results."``.
        """
        if pattern == topic:
            return True
        if pattern.endswith(".*") and topic.startswith(pattern[:-1]):
            return True
        return False


# ---------------------------------------------------------------------------
# Bus-connected agents
# ---------------------------------------------------------------------------


class BusAgent:
    """An agent that communicates exclusively via the message bus."""

    def __init__(self, name: str, bus: AgentBus, system_prompt: str) -> None:
        self.name = name
        self.bus = bus
        self.system_prompt = system_prompt
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        return self._client

    def _llm(self, user_content: str, temperature: float = 0.3) -> str:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=temperature,
        )
        return response.choices[0].message.content or ""

    def handle_task(self, msg: BusMessage) -> None:
        """Process a task message and publish the result.

        Subclasses should override this to define agent-specific logic.
        """
        result = self._llm(str(msg.payload))
        self.bus.publish("results.final", self.name, {"source_id": msg.message_id, "result": result})


class ResearchBusAgent(BusAgent):
    def handle_task(self, msg: BusMessage) -> None:
        query = msg.payload.get("query", str(msg.payload))
        result = self._llm(f"Research and summarise: {query}")
        # Publish research output to the analysis topic
        self.bus.publish(
            "tasks.analysis",
            self.name,
            {"source_id": msg.message_id, "research": result, "original_query": query},
        )


class AnalysisBusAgent(BusAgent):
    def handle_task(self, msg: BusMessage) -> None:
        research = msg.payload.get("research", "")
        query = msg.payload.get("original_query", "")
        result = self._llm(
            f"Analyse the following research and produce a concise summary "
            f"with data-driven insights:\n\nQuery: {query}\n\nResearch:\n{research}"
        )
        self.bus.publish(
            "results.final",
            self.name,
            {"source_id": msg.payload.get("source_id", ""), "analysis": result},
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def run_message_bus_demo() -> dict[str, Any]:
    """Demonstrate the message-bus pattern with research → analysis pipeline."""
    bus = AgentBus()
    results: list[dict] = []

    # Agents
    researcher = ResearchBusAgent(
        name="researcher",
        bus=bus,
        system_prompt="You are a research assistant. Provide concise, factual summaries.",
    )
    analyst = AnalysisBusAgent(
        name="analyst",
        bus=bus,
        system_prompt="You are a data analyst. Produce actionable insights from research.",
    )

    # Subscribe agents to their topics
    bus.subscribe("tasks.research", researcher.handle_task)
    bus.subscribe("tasks.analysis", analyst.handle_task)
    bus.subscribe("results.*", lambda msg: results.append(msg.payload))

    # Trigger the pipeline
    bus.publish(
        "tasks.research",
        "orchestrator",
        {"query": "What are the main advantages of Rust over C++ in systems programming?"},
    )

    return {
        "results": results,
        "message_log": bus.message_log(),
        "topics_used": list({m["topic"] for m in bus.message_log()}),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    output = run_message_bus_demo()

    print("\n--- Message Bus Log ---")
    for entry in output["message_log"]:
        print(f"  {entry['id']}  [{entry['topic']:20s}]  from {entry['sender']:15s}  {entry['payload_preview']}")

    print(f"\n--- Topics used: {output['topics_used']} ---")
    print(f"\n--- Results ({len(output['results'])}) ---")
    for r in output["results"]:
        for k, v in r.items():
            preview = str(v)[:200]
            print(f"  {k}: {preview}")
        print()


if __name__ == "__main__":
    main()
