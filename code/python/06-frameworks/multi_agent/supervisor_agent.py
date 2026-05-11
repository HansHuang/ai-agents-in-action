"""Supervisor-Worker multi-agent system.

The supervisor decomposes a goal into subtasks, assigns them to workers,
validates each result, and synthesises the final output. Failed subtasks
can be reassigned with specific feedback (up to MAX_REASSIGNMENTS times).

Implements Pattern 3 from:
docs/02-the-agent-loop/04-multi-agent-patterns.md

Run:
    python supervisor_agent.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_REASSIGNMENTS = 2
MAX_WORKER_ITERATIONS = 5


# ---------------------------------------------------------------------------
# Subtask model
# ---------------------------------------------------------------------------


class SubtaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    FAILED = "failed"


@dataclass
class Subtask:
    """A single unit of work managed by the supervisor."""

    id: str
    description: str
    assigned_worker: str
    dependencies: list[str] = field(default_factory=list)
    status: SubtaskStatus = SubtaskStatus.PENDING
    result: Optional[str] = None
    feedback: Optional[str] = None
    attempts: int = 0
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None

    def elapsed_ms(self) -> int:
        if self.start_ms is None:
            return 0
        end = self.end_ms or int(time.monotonic() * 1000)
        return end - self.start_ms


# ---------------------------------------------------------------------------
# Mock tools for workers
# ---------------------------------------------------------------------------

def _web_search(query: str) -> dict:
    results = {
        "AI code editors": [
            "GitHub Copilot leads with 35% market share among developers.",
            "Cursor AI raised $60M and claims 100k paying users.",
            "Replit Ghostwriter targets educational market with lower price point.",
        ],
        "default": [f"Search results for: {query}"],
    }
    for key in results:
        if key.lower() in query.lower():
            return {"query": query, "results": results[key]}
    return {"query": query, "results": results["default"]}


def _fetch_article(url: str) -> dict:
    return {
        "url": url,
        "title": "Mock article",
        "content": "This mock article contains competitive market analysis data.",
    }


def _calculate(expression: str) -> dict:
    """Evaluate a safe arithmetic expression."""
    # Only allow digits, operators, spaces, dots, and parens
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return {"error": "Expression contains disallowed characters"}
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return {"expression": expression, "result": result}
    except Exception as exc:  # noqa: BLE001
        return {"expression": expression, "error": str(exc)}


def _compare(items: list[dict]) -> dict:
    return {"items": items, "count": len(items), "comparison": "mock comparison"}


_TOOL_DISPATCH = {
    "web_search":   lambda a: _web_search(a["query"]),
    "fetch_article": lambda a: _fetch_article(a["url"]),
    "calculate":    lambda a: _calculate(a["expression"]),
    "compare":      lambda a: _compare(a.get("items", [])),
}

_RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent information about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_article",
            "description": "Fetch the full text of a web article.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL."}
                },
                "required": ["url"],
            },
        },
    },
]

_ANALYSIS_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a safe arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Arithmetic expression, e.g. '(450 * 1.1) / 3'."}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare",
            "description": "Compare a list of items side-by-side.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of objects to compare.",
                    }
                },
                "required": ["items"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# WorkerAgent
# ---------------------------------------------------------------------------


class WorkerAgent:
    """A specialist worker that executes a single subtask.

    Args:
        name:          Worker identifier.
        tools:         Tool definitions available to this worker.
        system_prompt: System prompt scoping the worker's behaviour.
    """

    def __init__(
        self,
        name: str,
        tools: list[dict],
        system_prompt: str,
    ) -> None:
        self.name = name
        self.tools = tools
        self.system_prompt = system_prompt
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        return self._client

    def execute(self, subtask: Subtask, context: str = "") -> str:
        """Execute the subtask and return a result string."""
        user_content = subtask.description
        if context:
            user_content += f"\n\nContext from completed subtasks:\n{context}"
        if subtask.feedback:
            user_content += (
                f"\n\nThis subtask was previously attempted but did not meet requirements.\n"
                f"Feedback: {subtask.feedback}\nPlease address these issues."
            )

        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]

        for _ in range(MAX_WORKER_ITERATIONS):
            kwargs: dict[str, Any] = {"model": "gpt-4o", "messages": messages}
            if self.tools:
                kwargs["tools"] = self.tools
                kwargs["tool_choice"] = "auto"

            response = self.client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            assistant_msg: dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                return msg.content or ""

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                fn = _TOOL_DISPATCH.get(tc.function.name)
                content = json.dumps(fn(args) if fn else {"error": f"Unknown tool: {tc.function.name}"})
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": content})

        return messages[-1].get("content", "")


# ---------------------------------------------------------------------------
# SupervisorAgent
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = """\
You are a project manager. Given a goal, decompose it into subtasks.

Available workers:
- research_worker: web search, fact-finding, data gathering
- analysis_worker: data analysis, comparisons, calculations
- writing_worker:  content creation, summaries, structured reports (no tools)

Output a JSON array of subtasks. Each subtask must have:
{
  "id": "t1",                     // short unique identifier
  "description": "...",           // complete, self-contained task description
  "assigned_worker": "research_worker",
  "dependencies": []              // IDs of subtasks this must wait for
}

Rules:
- Use at least 3 subtasks for non-trivial goals.
- Dependencies must only reference earlier task IDs.
- writing_worker tasks should always depend on research and analysis tasks.
- Make each description specific enough that the worker can act without clarification."""

_VALIDATE_SYSTEM = """\
You are a quality reviewer. Given a subtask description and the worker's result,
decide if the result adequately addresses the subtask.

Respond with JSON:
{
  "passes": true | false,
  "feedback": "Specific, actionable feedback if it fails. Empty string if it passes."
}

Criteria: completeness (did it answer the task?), specificity (are there concrete details?),
accuracy (are claims plausible?). Do not require perfection — accept results that are
substantially complete."""

_SYNTHESIZE_SYSTEM = """\
You are a synthesizer. Combine the results of all completed subtasks into a single,
well-structured final deliverable. Use headings, bullet points, and tables where
appropriate. Produce a polished output suitable for a business audience."""


class SupervisorAgent:
    """Decomposes, assigns, validates, and synthesises multi-worker workflows.

    Args:
        workers: Dict mapping worker names to WorkerAgent instances.
    """

    def __init__(self, workers: dict[str, WorkerAgent]) -> None:
        self.workers = workers
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        return self._client

    def run(self, goal: str) -> dict[str, Any]:
        """Execute the full supervisor workflow.

        Returns:
            A dict with:
                ``answer``    — final synthesised output
                ``workflow``  — list of subtask records with status and timings
        """
        subtasks = self._decompose(goal)
        logger.info("[supervisor] Decomposed into %d subtask(s)", len(subtasks))

        completed: dict[str, Subtask] = {}
        remaining = list(subtasks)
        max_rounds = len(subtasks) * (MAX_REASSIGNMENTS + 1) + 1

        for _ in range(max_rounds):
            if not remaining:
                break
            ready = [
                s for s in remaining
                if all(dep in completed for dep in s.dependencies)
            ]
            if not ready:
                logger.warning("[supervisor] No ready subtasks — unresolvable dependency")
                for s in remaining:
                    s.status = SubtaskStatus.FAILED
                    s.result = "Unresolvable dependency"
                    completed[s.id] = s
                break

            for subtask in ready:
                self._execute_subtask(subtask, completed)
                completed[subtask.id] = subtask
                remaining.remove(subtask)

        answer = self._synthesize(goal, completed)
        workflow = [
            {
                "id": s.id,
                "description": s.description,
                "worker": s.assigned_worker,
                "status": s.status.value,
                "attempts": s.attempts,
                "elapsed_ms": s.elapsed_ms(),
                "result_preview": (s.result or "")[:100],
            }
            for s in subtasks
        ]
        return {"answer": answer, "workflow": workflow}

    # ------------------------------------------------------------------
    # Phase 1: Decompose
    # ------------------------------------------------------------------

    def _decompose(self, goal: str) -> list[Subtask]:
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _DECOMPOSE_SYSTEM},
                {"role": "user", "content": goal},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or "[]"
        try:
            data = json.loads(raw)
            # The model may return {"subtasks": [...]} or a raw array
            if isinstance(data, dict):
                items = data.get("subtasks", data.get("tasks", list(data.values())[0] if data else []))
            else:
                items = data
            return [
                Subtask(
                    id=item["id"],
                    description=item["description"],
                    assigned_worker=item["assigned_worker"],
                    dependencies=item.get("dependencies", []),
                )
                for item in items
                if isinstance(item, dict)
            ]
        except Exception as exc:  # noqa: BLE001
            logger.warning("[supervisor] Decompose parse error: %s; using fallback", exc)
            return [
                Subtask(
                    id="t1",
                    description=goal,
                    assigned_worker="writing_worker",
                )
            ]

    # ------------------------------------------------------------------
    # Phase 2: Execute + Validate
    # ------------------------------------------------------------------

    def _execute_subtask(
        self, subtask: Subtask, completed: dict[str, Subtask]
    ) -> None:
        """Run a subtask with validation; reassign on failure."""
        worker = self.workers.get(subtask.assigned_worker)
        if worker is None:
            logger.warning("[supervisor] Unknown worker '%s'", subtask.assigned_worker)
            subtask.status = SubtaskStatus.FAILED
            subtask.result = f"Unknown worker: {subtask.assigned_worker}"
            return

        # Build context from completed dependencies
        context_parts = []
        for dep_id in subtask.dependencies:
            dep = completed.get(dep_id)
            if dep and dep.result:
                context_parts.append(f"[{dep.id}] {dep.description}\nResult: {dep.result[:500]}")
        context = "\n\n".join(context_parts)

        for attempt in range(1, MAX_REASSIGNMENTS + 2):
            subtask.attempts = attempt
            subtask.status = SubtaskStatus.IN_PROGRESS
            subtask.start_ms = int(time.monotonic() * 1000)
            logger.info(
                "[supervisor] → [%s] attempt %d/%d: %s",
                subtask.assigned_worker, attempt, MAX_REASSIGNMENTS + 1,
                subtask.description[:60],
            )

            result = worker.execute(subtask, context)
            subtask.end_ms = int(time.monotonic() * 1000)

            # Validate result
            passes, feedback = self._validate(subtask, result)
            if passes:
                subtask.status = SubtaskStatus.DONE
                subtask.result = result
                logger.info("[supervisor] ✓ [%s] %s", subtask.assigned_worker, subtask.id)
                return

            subtask.feedback = feedback
            logger.info(
                "[supervisor] ✗ [%s] validation failed: %s",
                subtask.assigned_worker, feedback[:80],
            )
            if attempt >= MAX_REASSIGNMENTS + 1:
                # Accept best effort
                subtask.status = SubtaskStatus.DONE
                subtask.result = result
                logger.info("[supervisor] Accepted best-effort result for %s", subtask.id)
                return

    def _validate(self, subtask: Subtask, result: str) -> tuple[bool, str]:
        """Ask the supervisor LLM to validate a worker's result."""
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _VALIDATE_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Subtask: {subtask.description}\n\nResult:\n{result[:1000]}"
                    ),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            return data.get("passes", True), data.get("feedback", "")
        except Exception:  # noqa: BLE001
            return True, ""

    # ------------------------------------------------------------------
    # Phase 3: Synthesize
    # ------------------------------------------------------------------

    def _synthesize(self, goal: str, completed: dict[str, Subtask]) -> str:
        parts = [f"Goal: {goal}\n"]
        for subtask in completed.values():
            parts.append(f"Subtask {subtask.id}: {subtask.description}")
            if subtask.result:
                parts.append(f"Result: {subtask.result[:800]}")
            parts.append("")

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYNTHESIZE_SYSTEM},
                {"role": "user", "content": "\n".join(parts)},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Default workers
# ---------------------------------------------------------------------------


def build_workers() -> dict[str, WorkerAgent]:
    return {
        "research_worker": WorkerAgent(
            name="research_worker",
            tools=_RESEARCH_TOOLS,
            system_prompt="""\
You are a research specialist. Use your tools to find relevant information.
Summarise key findings clearly. Attribute claims to sources. Report facts only.""",
        ),
        "analysis_worker": WorkerAgent(
            name="analysis_worker",
            tools=_ANALYSIS_TOOLS,
            system_prompt="""\
You are a data analyst. Analyse the provided data, perform calculations,
and draw evidence-based conclusions. Be precise with numbers.""",
        ),
        "writing_worker": WorkerAgent(
            name="writing_worker",
            tools=[],
            system_prompt="""\
You are a professional writer. Turn structured data and findings into polished,
well-organised prose. Use clear headings and bullet points.""",
        ),
    }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _gantt_bar(elapsed_ms: int, max_ms: int, width: int = 20) -> str:
    if max_ms == 0:
        return "░" * width
    filled = max(1, int((elapsed_ms / max_ms) * width))
    return "█" * filled + "░" * (width - filled)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    workers = build_workers()
    supervisor = SupervisorAgent(workers)

    goal = "Create a competitive analysis report for the AI code editor market"

    print(f"\n{'=' * 60}")
    print(f"Goal: {goal}")
    print("=" * 60)

    output = supervisor.run(goal)

    # Gantt chart
    print("\n--- Workflow ---")
    max_ms = max((t["elapsed_ms"] for t in output["workflow"]), default=1) or 1
    for t in output["workflow"]:
        bar = _gantt_bar(t["elapsed_ms"], max_ms)
        deps = f" (after {t.get('dependencies', [])})" if t.get("dependencies") else ""
        attempts_label = f" [attempt {t['attempts']}]" if t["attempts"] > 1 else ""
        print(
            f"  {t['id']:3s} [{t['worker'][:16]:16s}]  {bar}  "
            f"{t['status'].upper()}{attempts_label}{deps}"
        )

    print(f"\n{'=' * 60}")
    print("FINAL ANSWER:")
    print("=" * 60)
    print(output["answer"])


if __name__ == "__main__":
    main()
