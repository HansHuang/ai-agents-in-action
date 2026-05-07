"""LangSmith tracing integration for any agent — LangChain or custom.

Provides :class:`LangSmithTracer`, which wraps LangSmith's Client API
to add tracing to agents that were NOT built with LangChain.

Key insight: LangSmith is an observability tool.  You don't have to use
LangChain to benefit from LangSmith tracing.  This module shows how to
log LLM calls, tool executions, and user feedback from any agent.

Also includes :func:`compare_traces` — run both the from-scratch agent
and the LangGraph agent under the same tracer and compare the resulting
run metadata.

Run:
    LANGSMITH_API_KEY=... python langsmith_tracer.py

See: docs/06-frameworks-in-practice/02-langchain-langgraph.md
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Optional LangSmith import (fail gracefully)
# ---------------------------------------------------------------------------

try:
    import langsmith
    from langsmith import Client as LangSmithClient

    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Internal trace log (used when LangSmith is not available)
# ---------------------------------------------------------------------------


@dataclass
class _TraceEvent:
    """A single event in a local trace log."""

    trace_id: str
    event_type: str   # "llm_call" | "tool_call" | "feedback" | "run_start" | "run_end"
    data: dict
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class _LocalTraceStore:
    """In-memory trace store used when LangSmith is unavailable."""

    def __init__(self) -> None:
        self._traces: dict[str, list[_TraceEvent]] = {}

    def start_trace(self, trace_id: str, data: dict) -> None:
        self._traces[trace_id] = [_TraceEvent(trace_id, "run_start", data)]

    def add_event(self, trace_id: str, event_type: str, data: dict) -> None:
        if trace_id not in self._traces:
            self._traces[trace_id] = []
        self._traces[trace_id].append(_TraceEvent(trace_id, event_type, data))

    def get_trace(self, trace_id: str) -> list[_TraceEvent]:
        return self._traces.get(trace_id, [])

    def all_trace_ids(self) -> list[str]:
        return list(self._traces.keys())


_local_store = _LocalTraceStore()

# ---------------------------------------------------------------------------
# LangSmithTracer
# ---------------------------------------------------------------------------


class LangSmithTracer:
    """Add LangSmith tracing to any agent — LangChain or custom.

    Works in two modes:
      1. **LangSmith mode** (``LANGSMITH_API_KEY`` is set and ``langsmith``
         package is installed): sends real traces to LangSmith.
      2. **Local mode** (fallback): stores traces in memory and prints them;
         no network calls, no API key required.

    Usage::

        tracer = LangSmithTracer(project_name="my-project")

        trace_id = tracer.trace_agent_run("MyAgent", "What's the weather in Tokyo?")
        tracer.log_llm_call(trace_id, "gpt-4o", messages, response, tokens=150, latency_ms=320)
        tracer.log_tool_call(trace_id, "get_weather", {"city": "Tokyo"}, "22°C", latency_ms=12)
        tracer.end_trace(trace_id, answer="It's 22°C in Tokyo.")
        tracer.log_feedback(trace_id, score=0.9, comment="Good answer")

    Args:
        project_name: LangSmith project to log runs under.
    """

    def __init__(self, project_name: str = "ai-agents-in-action") -> None:
        self.project_name = project_name
        self._client: Optional[Any] = None

        if LANGSMITH_AVAILABLE and os.environ.get("LANGSMITH_API_KEY"):
            try:
                self._client = LangSmithClient()
            except Exception:
                # Fall back to local mode silently
                self._client = None

        self._mode = "langsmith" if self._client is not None else "local"

    @property
    def mode(self) -> str:
        """Either ``'langsmith'`` (real API) or ``'local'`` (in-memory)."""
        return self._mode

    # ------------------------------------------------------------------
    # Tracing operations
    # ------------------------------------------------------------------

    def trace_agent_run(self, agent_name: str, user_input: str) -> str:
        """Start a new agent run trace.

        Args:
            agent_name: Identifier for the agent (e.g. ``"LangGraphReActAgent"``).
            user_input: The user's question or instruction.

        Returns:
            A ``trace_id`` string — pass this to all subsequent log calls.
        """
        trace_id = str(uuid.uuid4())
        data = {
            "agent_name": agent_name,
            "user_input": user_input,
            "project": self.project_name,
            "started_at": datetime.now(timezone.utc).isoformat(),
        }

        if self._client is not None:
            try:
                self._client.create_run(
                    name=agent_name,
                    run_type="chain",
                    inputs={"input": user_input},
                    project_name=self.project_name,
                    id=trace_id,
                )
            except Exception as exc:
                # Log locally as fallback
                data["langsmith_error"] = str(exc)

        _local_store.start_trace(trace_id, data)
        return trace_id

    def log_llm_call(
        self,
        trace_id: str,
        model: str,
        input_messages: list,
        output_response: dict,
        tokens_used: int,
        latency_ms: float,
    ) -> None:
        """Log a single LLM call within a trace.

        Args:
            trace_id:       From :meth:`trace_agent_run`.
            model:          Model identifier, e.g. ``"gpt-4o"``.
            input_messages: The messages sent to the LLM.
            output_response:The response object (or dict).
            tokens_used:    Total token count (prompt + completion).
            latency_ms:     Time to receive the response, in milliseconds.
        """
        data = {
            "model": model,
            "tokens": tokens_used,
            "latency_ms": latency_ms,
            "message_count": len(input_messages),
        }

        if self._client is not None:
            try:
                child_id = str(uuid.uuid4())
                self._client.create_run(
                    name=f"llm:{model}",
                    run_type="llm",
                    inputs={"messages": _safe_serialize(input_messages)},
                    outputs={"response": _safe_serialize(output_response)},
                    parent_run_id=trace_id,
                    project_name=self.project_name,
                    id=child_id,
                    extra={"tokens": tokens_used, "latency_ms": latency_ms},
                )
            except Exception as exc:
                data["langsmith_error"] = str(exc)

        _local_store.add_event(trace_id, "llm_call", data)

    def log_tool_call(
        self,
        trace_id: str,
        tool_name: str,
        input_params: dict,
        output_result: str,
        latency_ms: float,
    ) -> None:
        """Log a tool execution within a trace.

        Args:
            trace_id:      From :meth:`trace_agent_run`.
            tool_name:     Name of the tool, e.g. ``"get_weather"``.
            input_params:  Parameters passed to the tool.
            output_result: The tool's return value as a string.
            latency_ms:    Tool execution time in milliseconds.
        """
        data = {
            "tool": tool_name,
            "input": input_params,
            "output": output_result[:200],
            "latency_ms": latency_ms,
        }

        if self._client is not None:
            try:
                child_id = str(uuid.uuid4())
                self._client.create_run(
                    name=f"tool:{tool_name}",
                    run_type="tool",
                    inputs=input_params,
                    outputs={"result": output_result},
                    parent_run_id=trace_id,
                    project_name=self.project_name,
                    id=child_id,
                    extra={"latency_ms": latency_ms},
                )
            except Exception as exc:
                data["langsmith_error"] = str(exc)

        _local_store.add_event(trace_id, "tool_call", data)

    def end_trace(
        self,
        trace_id: str,
        answer: str,
        error: Optional[str] = None,
    ) -> None:
        """Mark a trace as complete.

        Args:
            trace_id: From :meth:`trace_agent_run`.
            answer:   The agent's final answer.
            error:    Error message if the agent failed; None otherwise.
        """
        data = {"answer": answer[:500], "error": error}

        if self._client is not None:
            try:
                self._client.update_run(
                    run_id=trace_id,
                    outputs={"answer": answer},
                    error=error,
                    end_time=datetime.now(timezone.utc),
                )
            except Exception as exc:
                data["langsmith_error"] = str(exc)

        _local_store.add_event(trace_id, "run_end", data)

    def log_feedback(
        self,
        trace_id: str,
        score: float,
        comment: Optional[str] = None,
    ) -> None:
        """Log user feedback on a completed trace.

        Args:
            trace_id: From :meth:`trace_agent_run`.
            score:    Numeric quality score in [0.0, 1.0].
            comment:  Optional human-readable feedback string.

        Raises:
            ValueError: If score is not in [0.0, 1.0].
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"score must be in [0.0, 1.0], got {score}")

        data = {"score": score, "comment": comment}

        if self._client is not None:
            try:
                self._client.create_feedback(
                    run_id=trace_id,
                    key="user_score",
                    score=score,
                    comment=comment,
                )
            except Exception as exc:
                data["langsmith_error"] = str(exc)

        _local_store.add_event(trace_id, "feedback", data)

    # ------------------------------------------------------------------
    # Trace inspection
    # ------------------------------------------------------------------

    def get_trace_summary(self, trace_id: str) -> dict:
        """Return a summary of a completed trace.

        Args:
            trace_id: From :meth:`trace_agent_run`.

        Returns:
            Dict with keys: ``trace_id``, ``llm_calls``, ``tool_calls``,
            ``total_tokens``, ``total_latency_ms``, ``feedback``.
        """
        events = _local_store.get_trace(trace_id)

        llm_calls = [e for e in events if e.event_type == "llm_call"]
        tool_calls = [e for e in events if e.event_type == "tool_call"]
        feedbacks = [e for e in events if e.event_type == "feedback"]

        return {
            "trace_id": trace_id,
            "llm_calls": len(llm_calls),
            "tool_calls": [e.data.get("tool") for e in tool_calls],
            "total_tokens": sum(e.data.get("tokens", 0) for e in llm_calls),
            "total_latency_ms": sum(
                e.data.get("latency_ms", 0) for e in llm_calls + tool_calls
            ),
            "feedback": [e.data for e in feedbacks],
        }

    def compare_traces(self, trace_id_a: str, trace_id_b: str) -> str:
        """Compare two traces and highlight differences.

        Args:
            trace_id_a: First trace ID.
            trace_id_b: Second trace ID.

        Returns:
            Human-readable diff string.
        """
        a = self.get_trace_summary(trace_id_a)
        b = self.get_trace_summary(trace_id_b)

        lines = [
            "Trace Comparison",
            "─" * 50,
            f"{'Metric':<30} {'Trace A':>10} {'Trace B':>10}",
            "─" * 50,
            f"{'LLM calls':<30} {a['llm_calls']:>10} {b['llm_calls']:>10}",
            f"{'Tool calls':<30} {len(a['tool_calls']):>10} {len(b['tool_calls']):>10}",
            f"{'Total tokens':<30} {a['total_tokens']:>10} {b['total_tokens']:>10}",
            f"{'Total latency (ms)':<30} {a['total_latency_ms']:>10.0f} {b['total_latency_ms']:>10.0f}",
            "─" * 50,
            f"Trace A tools: {', '.join(a['tool_calls']) or 'none'}",
            f"Trace B tools: {', '.join(b['tool_calls']) or 'none'}",
        ]

        # Highlight token overhead
        if a["total_tokens"] > 0 and b["total_tokens"] > 0:
            overhead = abs(a["total_tokens"] - b["total_tokens"]) / max(
                a["total_tokens"], b["total_tokens"]
            )
            lines.append(f"\nToken difference: {overhead:.1%}")
            if overhead > 0.10:
                heavier = "A" if a["total_tokens"] > b["total_tokens"] else "B"
                lines.append(f"⚠ Trace {heavier} uses >10% more tokens.")

        if a["feedback"] or b["feedback"]:
            lines.append(f"\nTrace A feedback: {a['feedback']}")
            lines.append(f"Trace B feedback: {b['feedback']}")

        return "\n".join(lines)

    def print_trace(self, trace_id: str) -> None:
        """Print all events in a trace for debugging."""
        events = _local_store.get_trace(trace_id)
        print(f"\nTrace: {trace_id}")
        print("─" * 60)
        for event in events:
            ts = event.timestamp[11:19]  # HH:MM:SS
            etype = event.event_type.upper()
            data_preview = str(event.data)[:80]
            print(f"  [{ts}] {etype:<12} {data_preview}")
        print()


# ---------------------------------------------------------------------------
# Comparison demo
# ---------------------------------------------------------------------------


def compare_agent_traces(query: str, tracer: LangSmithTracer) -> tuple[str, str]:
    """Run *query* through the from-scratch agent and LangGraph agent,
    both under the same :class:`LangSmithTracer`.

    Args:
        query:  User question.
        tracer: Shared tracer instance.

    Returns:
        Tuple of ``(from_scratch_trace_id, langgraph_trace_id)``.
    """
    import sys
    from pathlib import Path

    # -- From-scratch agent --
    scratch_trace = tracer.trace_agent_run("FromScratchAgent", query)
    scratch_dir = Path(__file__).parent.parent / "03-agent-loop"
    if scratch_dir.is_dir():
        sys.path.insert(0, str(scratch_dir))
        try:
            import agent as scratch_agent  # type: ignore
            import tools as scratch_tools  # type: ignore

            messages: list[dict] = [
                {"role": "system", "content": scratch_agent.SYSTEM_PROMPT}
            ]
            t0 = time.monotonic()
            answer = scratch_agent.run_agent(
                query, messages=messages, tools=scratch_tools.TOOLS
            )
            latency = (time.monotonic() - t0) * 1000

            # Log each assistant turn as an LLM call
            for msg in messages:
                if msg["role"] == "assistant":
                    tracer.log_llm_call(
                        scratch_trace,
                        model="gpt-4o",
                        input_messages=messages,
                        output_response={"content": msg.get("content", "")},
                        tokens_used=len(str(messages)) // 4,  # rough estimate
                        latency_ms=latency,
                    )
                if msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        tracer.log_tool_call(
                            scratch_trace,
                            tool_name=tc["function"]["name"],
                            input_params={"args": tc["function"]["arguments"]},
                            output_result="(see tool message)",
                            latency_ms=5.0,
                        )

            tracer.end_trace(scratch_trace, answer=answer)
        except Exception as exc:
            tracer.end_trace(scratch_trace, answer="", error=str(exc))
        finally:
            if str(scratch_dir) in sys.path:
                sys.path.remove(str(scratch_dir))
    else:
        tracer.end_trace(
            scratch_trace,
            answer="(from-scratch agent not found)",
        )

    # -- LangGraph agent --
    lg_trace = tracer.trace_agent_run("LangGraphReActAgent", query)
    try:
        from langgraph_react_agent import LangGraphReActAgent  # type: ignore

        agent = LangGraphReActAgent()
        t0 = time.monotonic()
        result = agent.run(query)
        latency = (time.monotonic() - t0) * 1000

        tracer.log_llm_call(
            lg_trace,
            model="gpt-4o",
            input_messages=[{"role": "user", "content": query}],
            output_response={"content": result.answer},
            tokens_used=len(result.answer) // 3,  # rough estimate
            latency_ms=latency,
        )
        for tc_name in result.tool_calls_made:
            tracer.log_tool_call(
                lg_trace,
                tool_name=tc_name,
                input_params={},
                output_result="(tool output)",
                latency_ms=5.0,
            )

        tracer.end_trace(lg_trace, answer=result.answer)
    except Exception as exc:
        tracer.end_trace(lg_trace, answer="", error=str(exc))

    return scratch_trace, lg_trace


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _safe_serialize(obj: Any) -> Any:
    """Recursively convert objects to JSON-serialisable form."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _safe_serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_serialize(v) for v in obj]
    return str(obj)


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the tracing demo: log both agents and compare traces."""
    tracer = LangSmithTracer(project_name="ai-agents-in-action")

    print(f"LangSmith Tracer Demo (mode: {tracer.mode})")
    print("=" * 50)

    query = "What's the weather in Tokyo?"

    print(f"Query: {query!r}")
    print("Running both agents with tracing...\n")

    scratch_tid, lg_tid = compare_agent_traces(query, tracer)

    # Print individual traces
    tracer.print_trace(scratch_tid)
    tracer.print_trace(lg_tid)

    # Log user feedback on LangGraph trace
    tracer.log_feedback(lg_tid, score=0.9, comment="Good answer")
    print("Logged feedback: score=0.9, comment='Good answer'\n")

    # Compare both traces
    print(tracer.compare_traces(scratch_tid, lg_tid))

    if tracer.mode == "langsmith":
        print(
            f"\nView traces at: https://smith.langchain.com/o/*/projects/p/{tracer.project_name}"
        )
    else:
        print(
            "\nTo send traces to LangSmith:\n"
            "  pip install langsmith\n"
            "  export LANGSMITH_API_KEY=<your-key>"
        )


if __name__ == "__main__":
    main()
