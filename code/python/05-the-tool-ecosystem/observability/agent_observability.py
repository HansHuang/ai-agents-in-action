"""Agent Observability System.

Implements the three pillars of agent observability:
  - Tracing: captures the full decision tree of a single request
  - Logging: structured JSON events at each step
  - Metrics: rolling aggregates for system health dashboards

Additional components:
  - TokenAccountant: per-user/session/model cost tracking
  - DecisionTracer: captures reasoning to explain "why did it do that?"

See: docs/05-the-tool-ecosystem/03-agent-observability.md
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol

# ---------------------------------------------------------------------------
# Pricing defaults (USD per 1 000 tokens, input / output)
# ---------------------------------------------------------------------------

DEFAULT_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o":            {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini":       {"input": 0.00015, "output": 0.0006},
    "claude-3-5-sonnet": {"input": 0.003,  "output": 0.015},
    "claude-3-haiku":    {"input": 0.00025, "output": 0.00125},
    "gemini-1.5-pro":    {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash":  {"input": 0.000075, "output": 0.0003},
    "unknown":           {"input": 0.001,  "output": 0.003},
}


def _compute_cost(model: str, input_tokens: int, output_tokens: int,
                  pricing: dict) -> float:
    """Calculate USD cost for a single LLM call."""
    rates = pricing.get(model) or pricing.get("unknown", {})
    return (input_tokens * rates.get("input", 0.001) +
            output_tokens * rates.get("output", 0.003)) / 1000


# ===========================================================================
# Core data model: Span and Trace
# ===========================================================================

@dataclass
class Span:
    """A single operation within a trace (LLM call, tool call, planning, …)."""

    span_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_span_id: str | None = None
    type: str = ""          # "llm_call" | "tool_call" | "planning" | "execution"
    name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    input_data: dict | None = None
    output_data: dict | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_used: int = 0    # total = input + output (kept for compat)
    cost: float = 0.0
    model: str | None = None
    status: str = "running"  # "running" | "success" | "error"
    error_message: str | None = None
    metadata: dict = field(default_factory=dict)

    # -------------------------------------------------------------------
    def finish(self, output_data: dict | None = None,
               status: str = "success",
               error_message: str | None = None) -> "Span":
        self.end_time = time.time()
        self.output_data = output_data
        self.status = status
        self.error_message = error_message
        return self

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "type": self.type,
            "name": self.name,
            "duration_ms": round(self.duration_ms, 2),
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokens_used": self.tokens_used,
            "cost": round(self.cost, 6),
            "model": self.model,
            "status": self.status,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }


@dataclass
class Trace:
    """The complete record of one agent request."""

    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_query: str = ""
    user_id: str | None = None
    session_id: str | None = None
    spans: list[Span] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    metadata: dict = field(default_factory=dict)

    # -------------------------------------------------------------------
    def add_span(self, span: Span) -> Span:
        self.spans.append(span)
        return span

    def new_span(self, type: str, name: str,
                 parent: Span | None = None, **kwargs) -> Span:
        span = Span(
            type=type,
            name=name,
            parent_span_id=parent.span_id if parent else None,
            **kwargs,
        )
        self.spans.append(span)
        return span

    def finish(self) -> "Trace":
        self.end_time = time.time()
        return self

    @property
    def duration_ms(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) * 1000

    @property
    def total_tokens(self) -> int:
        return sum(s.tokens_used for s in self.spans)

    @property
    def total_cost(self) -> float:
        return sum(s.cost for s in self.spans)

    @property
    def llm_call_count(self) -> int:
        return sum(1 for s in self.spans if s.type == "llm_call")

    @property
    def tool_call_count(self) -> int:
        return sum(1 for s in self.spans if s.type == "tool_call")

    @property
    def has_error(self) -> bool:
        return any(s.status == "error" for s in self.spans)

    @property
    def status(self) -> str:
        return "error" if self.has_error else "success"

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "user_query": self.user_query,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "llm_calls": self.llm_call_count,
            "tool_calls": self.tool_call_count,
            "total_tokens": self.total_tokens,
            "total_cost": round(self.total_cost, 6),
            "spans": [s.to_dict() for s in self.spans],
            "metadata": self.metadata,
        }


# ===========================================================================
# Exporter protocol and built-in console exporter
# ===========================================================================

class TraceExporter(Protocol):
    """Any object that can receive and persist a Trace."""

    def export(self, trace: Trace) -> None: ...


class ConsoleExporter:
    """Print a compact tree view of a trace to stdout."""

    def export(self, trace: Trace) -> None:
        status_icon = "✓" if not trace.has_error else "✗"
        print(f"\n{'='*60}")
        print(f"{status_icon} Trace {trace.trace_id[:8]}  "
              f"query='{trace.user_query[:60]}'")
        print(f"  duration={trace.duration_ms:.0f}ms  "
              f"tokens={trace.total_tokens}  cost=${trace.total_cost:.4f}")
        print(f"  llm_calls={trace.llm_call_count}  "
              f"tool_calls={trace.tool_call_count}  "
              f"status={trace.status}")
        print()
        for span in trace.spans:
            indent = "    " if span.parent_span_id else "  "
            icon = "✗" if span.status == "error" else "·"
            err = f"  ERROR: {span.error_message}" if span.error_message else ""
            print(f"{indent}{icon} [{span.type}] {span.name}"
                  f"  {span.duration_ms:.0f}ms"
                  f"  tokens={span.tokens_used}{err}")
        print(f"{'='*60}\n")


# ===========================================================================
# TraceCollector
# ===========================================================================

class TraceCollector:
    """Lifecycle manager for agent traces.

    Usage::

        collector = TraceCollector(exporter=JSONFileExporter("./traces/"))
        trace = collector.start_trace("Compare AAPL vs MSFT", user_id="u1")
        span  = trace.new_span("llm_call", "plan")
        ...
        span.finish(output_data={"plan": [...]})
        collector.end_trace(trace)
    """

    def __init__(self, exporter: TraceExporter | None = None):
        self.exporter: TraceExporter = exporter or ConsoleExporter()
        self._traces: dict[str, Trace] = {}

    # -------------------------------------------------------------------
    def start_trace(self, user_query: str,
                    user_id: str | None = None,
                    session_id: str | None = None,
                    metadata: dict | None = None) -> Trace:
        """Create and register a new trace."""
        trace = Trace(
            user_query=user_query,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata or {},
        )
        self._traces[trace.trace_id] = trace
        return trace

    def end_trace(self, trace: Trace) -> None:
        """Finalize and export a trace."""
        if trace.end_time is None:
            trace.finish()
        self.exporter.export(trace)

    def get_trace(self, trace_id: str) -> Trace | None:
        """Retrieve a trace by ID."""
        return self._traces.get(trace_id)

    def query_traces(
        self,
        user_id: str | None = None,
        status: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[Trace]:
        """Filter stored traces by user, status, or time window."""
        since_ts = since.timestamp() if since else 0.0
        results = [
            t for t in self._traces.values()
            if (user_id is None or t.user_id == user_id)
            and (status is None or t.status == status)
            and t.start_time >= since_ts
        ]
        results.sort(key=lambda t: t.start_time, reverse=True)
        return results[:limit]


# ===========================================================================
# AgentMetrics
# ===========================================================================

@dataclass
class MetricsSummary:
    """Snapshot of rolling metric statistics."""
    requests: int
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    avg_tokens: float
    avg_cost: float
    total_cost: float
    error_rate_pct: float
    avg_llm_calls: float
    avg_tool_calls: float

    def to_dict(self) -> dict:
        return {
            "requests": self.requests,
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "p95_latency_ms": round(self.p95_latency_ms, 1),
            "p99_latency_ms": round(self.p99_latency_ms, 1),
            "avg_tokens": round(self.avg_tokens, 1),
            "avg_cost": round(self.avg_cost, 4),
            "total_cost": round(self.total_cost, 4),
            "error_rate_pct": round(self.error_rate_pct, 2),
            "avg_llm_calls": round(self.avg_llm_calls, 2),
            "avg_tool_calls": round(self.avg_tool_calls, 2),
        }


@dataclass
class Alert:
    """A triggered anomaly alert."""
    kind: str          # "error_rate" | "latency" | "cost"
    message: str
    value: float
    threshold: float
    severity: str = "warning"  # "warning" | "critical"

    def __str__(self) -> str:
        return f"[{self.severity.upper()}] {self.message}"


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * pct / 100)
    return sorted_v[min(idx, len(sorted_v) - 1)]


class AgentMetrics:
    """Collect and aggregate agent performance metrics over a rolling window.

    All deques are bounded to ``window_size`` entries so memory stays fixed.
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.metrics: dict[str, deque] = {
            "llm_latency_ms":       deque(maxlen=window_size),
            "tool_latency_ms":      deque(maxlen=window_size),
            "total_latency_ms":     deque(maxlen=window_size),
            "tokens_per_request":   deque(maxlen=window_size),
            "cost_per_request":     deque(maxlen=window_size),
            "tool_calls_per_request": deque(maxlen=window_size),
            "llm_calls_per_request":  deque(maxlen=window_size),
            "errors_per_request":   deque(maxlen=window_size),
        }
        # Store timestamps alongside each sample for time-window queries
        self._timestamps: deque[float] = deque(maxlen=window_size)

    # -------------------------------------------------------------------
    def record(self, trace: Trace) -> None:
        """Ingest all metrics from a completed trace."""
        self._timestamps.append(time.time())
        self.metrics["total_latency_ms"].append(trace.duration_ms)
        self.metrics["tokens_per_request"].append(trace.total_tokens)
        self.metrics["cost_per_request"].append(trace.total_cost)
        self.metrics["tool_calls_per_request"].append(trace.tool_call_count)
        self.metrics["llm_calls_per_request"].append(trace.llm_call_count)
        self.metrics["errors_per_request"].append(1 if trace.has_error else 0)

        for span in trace.spans:
            if span.type == "llm_call" and span.end_time:
                self.metrics["llm_latency_ms"].append(span.duration_ms)
            elif span.type == "tool_call" and span.end_time:
                self.metrics["tool_latency_ms"].append(span.duration_ms)

    # -------------------------------------------------------------------
    def get_summary(self, window_minutes: int = 60) -> MetricsSummary:
        """Compute summary statistics over the last ``window_minutes``."""
        cutoff = time.time() - window_minutes * 60
        # Find indices within the time window
        indices = [
            i for i, ts in enumerate(self._timestamps) if ts >= cutoff
        ]

        def _windowed(key: str) -> list[float]:
            all_vals = list(self.metrics[key])
            if not indices or len(all_vals) != len(list(self._timestamps)):
                return all_vals  # fallback: use everything
            return [all_vals[i] for i in indices if i < len(all_vals)]

        latency = _windowed("total_latency_ms") or [0.0]
        tokens  = _windowed("tokens_per_request") or [0.0]
        costs   = _windowed("cost_per_request") or [0.0]
        errors  = _windowed("errors_per_request") or [0.0]
        llm_c   = _windowed("llm_calls_per_request") or [0.0]
        tool_c  = _windowed("tool_calls_per_request") or [0.0]

        n = len(latency)
        avg = lambda lst: sum(lst) / max(len(lst), 1)

        return MetricsSummary(
            requests=n,
            avg_latency_ms=avg(latency),
            p95_latency_ms=_percentile(latency, 95),
            p99_latency_ms=_percentile(latency, 99),
            avg_tokens=avg(tokens),
            avg_cost=avg(costs),
            total_cost=sum(costs),
            error_rate_pct=avg(errors) * 100,
            avg_llm_calls=avg(llm_c),
            avg_tool_calls=avg(tool_c),
        )

    # -------------------------------------------------------------------
    def detect_anomalies(self) -> list[Alert]:
        """Detect anomalies: error spikes, latency spikes, cost spikes."""
        alerts: list[Alert] = []
        summary = self.get_summary()

        # 1. Error rate > 5 %
        if summary.error_rate_pct > 5.0:
            alerts.append(Alert(
                kind="error_rate",
                message=f"High error rate: {summary.error_rate_pct:.1f}%",
                value=summary.error_rate_pct,
                threshold=5.0,
                severity="critical" if summary.error_rate_pct > 20 else "warning",
            ))

        # 2. P95 latency > 2× average (only meaningful with ≥10 requests)
        if summary.requests >= 10 and summary.avg_latency_ms > 0:
            if summary.p95_latency_ms > 2 * summary.avg_latency_ms:
                alerts.append(Alert(
                    kind="latency",
                    message=(f"P95 latency {summary.p95_latency_ms:.0f}ms "
                             f"> 2× average {summary.avg_latency_ms:.0f}ms"),
                    value=summary.p95_latency_ms,
                    threshold=2 * summary.avg_latency_ms,
                ))

        # 3. Avg cost > $0.50 per request
        if summary.avg_cost > 0.50:
            alerts.append(Alert(
                kind="cost",
                message=f"High cost per request: ${summary.avg_cost:.2f}",
                value=summary.avg_cost,
                threshold=0.50,
                severity="critical" if summary.avg_cost > 2.0 else "warning",
            ))

        return alerts

    # -------------------------------------------------------------------
    def export_prometheus(self) -> str:
        """Return metrics in Prometheus text format."""
        summary = self.get_summary()
        lines = [
            "# HELP agent_requests_total Rolling window request count",
            "# TYPE agent_requests_total gauge",
            f"agent_requests_total {summary.requests}",
            "",
            "# HELP agent_latency_avg_ms Average request latency in milliseconds",
            "# TYPE agent_latency_avg_ms gauge",
            f"agent_latency_avg_ms {summary.avg_latency_ms:.2f}",
            "",
            "# HELP agent_latency_p95_ms P95 request latency in milliseconds",
            "# TYPE agent_latency_p95_ms gauge",
            f"agent_latency_p95_ms {summary.p95_latency_ms:.2f}",
            "",
            "# HELP agent_error_rate_pct Error rate as a percentage",
            "# TYPE agent_error_rate_pct gauge",
            f"agent_error_rate_pct {summary.error_rate_pct:.2f}",
            "",
            "# HELP agent_avg_tokens_per_request Average tokens consumed per request",
            "# TYPE agent_avg_tokens_per_request gauge",
            f"agent_avg_tokens_per_request {summary.avg_tokens:.1f}",
            "",
            "# HELP agent_avg_cost_per_request Average cost in USD per request",
            "# TYPE agent_avg_cost_per_request gauge",
            f"agent_avg_cost_per_request {summary.avg_cost:.6f}",
            "",
            "# HELP agent_total_cost_usd Total cost in USD (rolling window)",
            "# TYPE agent_total_cost_usd gauge",
            f"agent_total_cost_usd {summary.total_cost:.6f}",
        ]
        return "\n".join(lines)


# ===========================================================================
# AgentLogger
# ===========================================================================

class AgentLogger:
    """Structured JSON logging for agent operations.

    Every log line is a valid JSON object so it can be ingested by any log
    aggregation system (Datadog, CloudWatch, Loki, …).
    """

    _REDACTED_KEYS = frozenset({
        "api_key", "apikey", "secret", "password", "token",
        "authorization", "credential", "private_key",
    })

    def __init__(self, log_level: str = "INFO"):
        self.logger = logging.getLogger("agent")
        self.logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.logger.addHandler(handler)

    # -------------------------------------------------------------------
    def _emit(self, level: int, payload: dict) -> None:
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        self.logger.log(level, json.dumps(self._redact(payload)))

    def _redact(self, obj: Any) -> Any:
        """Recursively redact sensitive keys from dicts."""
        if isinstance(obj, dict):
            return {
                k: "[REDACTED]" if k.lower() in self._REDACTED_KEYS
                else self._redact(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [self._redact(v) for v in obj]
        return obj

    # -------------------------------------------------------------------
    def log_llm_call(self, trace_id: str, span_id: str,
                     model: str, messages_count: int,
                     estimated_tokens: int) -> None:
        self._emit(logging.INFO, {
            "event": "llm_call_start",
            "trace_id": trace_id,
            "span_id": span_id,
            "model": model,
            "messages_count": messages_count,
            "estimated_tokens": estimated_tokens,
        })

    def log_llm_response(self, trace_id: str, span_id: str,
                         model: str, input_tokens: int,
                         output_tokens: int, latency_ms: float,
                         has_tool_calls: bool) -> None:
        self._emit(logging.INFO, {
            "event": "llm_call_complete",
            "trace_id": trace_id,
            "span_id": span_id,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "latency_ms": round(latency_ms, 2),
            "has_tool_calls": has_tool_calls,
        })

    def log_tool_execution(self, trace_id: str, span_id: str,
                           tool_name: str, params_summary: str) -> None:
        self._emit(logging.INFO, {
            "event": "tool_call_start",
            "trace_id": trace_id,
            "span_id": span_id,
            "tool_name": tool_name,
            "params_summary": params_summary,
        })

    def log_tool_result(self, trace_id: str, span_id: str,
                        tool_name: str, success: bool,
                        result_summary: str, latency_ms: float) -> None:
        level = logging.INFO if success else logging.WARNING
        self._emit(level, {
            "event": "tool_call_complete",
            "trace_id": trace_id,
            "span_id": span_id,
            "tool_name": tool_name,
            "success": success,
            "result_summary": result_summary[:200],
            "latency_ms": round(latency_ms, 2),
        })

    def log_context_management(self, trace_id: str,
                               action: str,
                               original_tokens: int,
                               result_tokens: int) -> None:
        """Log context-window management events (truncation, summarisation …)."""
        self._emit(logging.WARNING, {
            "event": "context_management",
            "trace_id": trace_id,
            "action": action,
            "original_tokens": original_tokens,
            "result_tokens": result_tokens,
            "tokens_removed": original_tokens - result_tokens,
        })

    def log_error(self, trace_id: str, error: Exception,
                  context: dict) -> None:
        self._emit(logging.ERROR, {
            "event": "agent_error",
            "trace_id": trace_id,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context,
        })


# ===========================================================================
# TokenAccountant
# ===========================================================================

@dataclass
class TokenRecord:
    """One cost record tied to a single trace."""
    trace_id: str
    user_id: str
    session_id: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: float = field(default_factory=time.time)


class TokenAccountant:
    """Track token usage and costs across users, sessions, and models.

    Budget alerts are checked on every ``record()`` call.
    """

    def __init__(self, pricing: dict | None = None):
        self.pricing = pricing or DEFAULT_PRICING
        self.records: list[TokenRecord] = []
        self._budget_alerts: dict[str, float] = {}   # user_id → max_cost_usd

    # -------------------------------------------------------------------
    def record(self, trace: Trace, user_id: str, session_id: str) -> None:
        """Extract token usage from a trace and persist a record."""
        for span in trace.spans:
            if span.type != "llm_call":
                continue
            model = span.model or "unknown"
            r = TokenRecord(
                trace_id=trace.trace_id,
                user_id=user_id,
                session_id=session_id,
                model=model,
                input_tokens=span.input_tokens,
                output_tokens=span.output_tokens,
                cost=span.cost,
            )
            self.records.append(r)

            # Check budget alert
            if user_id in self._budget_alerts:
                total = self.get_user_cost(user_id)
                budget = self._budget_alerts[user_id]
                if total > budget:
                    print(f"[BUDGET ALERT] User '{user_id}' exceeded "
                          f"budget ${budget:.2f}: current ${total:.4f}")

    # -------------------------------------------------------------------
    def get_user_cost(self, user_id: str, days: int = 30) -> float:
        cutoff = time.time() - days * 86400
        return sum(
            r.cost for r in self.records
            if r.user_id == user_id and r.timestamp >= cutoff
        )

    def get_session_cost(self, session_id: str) -> float:
        return sum(r.cost for r in self.records if r.session_id == session_id)

    def get_model_usage(self, model: str, days: int = 30) -> dict:
        cutoff = time.time() - days * 86400
        recs = [r for r in self.records
                if r.model == model and r.timestamp >= cutoff]
        return {
            "model": model,
            "calls": len(recs),
            "input_tokens": sum(r.input_tokens for r in recs),
            "output_tokens": sum(r.output_tokens for r in recs),
            "total_cost": sum(r.cost for r in recs),
        }

    def get_daily_cost_report(self) -> dict:
        """Aggregate cost for the last 24 hours, broken down by model."""
        cutoff = time.time() - 86400
        recent = [r for r in self.records if r.timestamp >= cutoff]

        by_model: dict[str, dict] = {}
        for r in recent:
            m = by_model.setdefault(r.model, {
                "calls": 0, "input_tokens": 0,
                "output_tokens": 0, "cost": 0.0,
            })
            m["calls"] += 1
            m["input_tokens"] += r.input_tokens
            m["output_tokens"] += r.output_tokens
            m["cost"] += r.cost

        unique_users = len({r.user_id for r in recent})
        return {
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "total_requests": len(recent),
            "total_input_tokens": sum(r.input_tokens for r in recent),
            "total_output_tokens": sum(r.output_tokens for r in recent),
            "total_cost": sum(r.cost for r in recent),
            "unique_users": unique_users,
            "by_model": by_model,
        }

    def set_budget_alert(self, user_id: str, max_cost: float) -> None:
        """Trigger a console alert whenever ``user_id`` exceeds ``max_cost``."""
        self._budget_alerts[user_id] = max_cost


# ===========================================================================
# DecisionTracer
# ===========================================================================

@dataclass
class Decision:
    """A single agent decision captured for later debugging."""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    step: str = ""
    context: dict = field(default_factory=dict)
    options: list[str] = field(default_factory=list)
    chosen: str = ""
    reasoning: str = ""


class DecisionTracer:
    """Capture agent decisions for debugging "why did the agent do that?"

    Particularly useful for routing/planning decisions where the agent chose
    tool A instead of tool B.
    """

    def __init__(self):
        self.decisions: list[Decision] = []

    # -------------------------------------------------------------------
    def capture(self, step: str, context: dict,
                options: list[str], chosen: str,
                reasoning: str) -> Decision:
        """Record one decision."""
        d = Decision(
            step=step,
            context=context,
            options=options,
            chosen=chosen,
            reasoning=reasoning,
        )
        self.decisions.append(d)
        return d

    def replay(self, trace_id: str | None = None) -> str:
        """Return a human-readable decision trail."""
        if not self.decisions:
            return "(no decisions recorded)"
        lines = [f"# Agent Decision Trail{' — trace ' + trace_id if trace_id else ''}\n"]
        for i, d in enumerate(self.decisions, 1):
            lines.append(f"## Step {i}: {d.step}")
            if d.context:
                lines.append(f"  Context:   {d.context}")
            lines.append(f"  Options:   {', '.join(d.options) or '(none)'}")
            lines.append(f"  Chosen:    {d.chosen}")
            lines.append(f"  Reasoning: {d.reasoning}")
            lines.append("")
        return "\n".join(lines)

    def find_divergence(self, expected_choice: str) -> str:
        """Find the step where the agent diverged from an expected choice."""
        for i, d in enumerate(self.decisions, 1):
            if expected_choice in d.options and d.chosen != expected_choice:
                return (
                    f"Divergence at step {i} ({d.step}):\n"
                    f"  Expected:  '{expected_choice}'\n"
                    f"  Chosen:    '{d.chosen}'\n"
                    f"  Reasoning: {d.reasoning}\n"
                    f"  Context:   {d.context}"
                )
        return f"No divergence found — '{expected_choice}' was either chosen or not in options."


# ===========================================================================
# Simulated observable agent (for demos)
# ===========================================================================

class SimulatedLLM:
    """Fake LLM that returns deterministic responses for demo purposes."""

    MODEL = "gpt-4o-mini"
    INPUT_TOKENS = 512
    OUTPUT_TOKENS = 128

    def __init__(self, fail_on: str | None = None):
        self._fail_on = fail_on  # query substring that triggers a failure

    def chat(self, query: str) -> dict:
        if self._fail_on and self._fail_on in query:
            raise ValueError(f"Simulated LLM failure for query: {query!r}")
        time.sleep(0.05)  # simulate network latency
        return {
            "model": self.MODEL,
            "input_tokens": self.INPUT_TOKENS,
            "output_tokens": self.OUTPUT_TOKENS,
            "content": f"Simulated answer for: {query[:40]}",
            "tool_calls": None,
        }


class ObservableAgent:
    """A demo agent wired up with all observability components."""

    def __init__(
        self,
        collector: TraceCollector,
        metrics: AgentMetrics,
        logger: AgentLogger,
        accountant: TokenAccountant,
        decision_tracer: DecisionTracer,
        llm: SimulatedLLM | None = None,
        pricing: dict | None = None,
    ):
        self.collector = collector
        self.metrics = metrics
        self.log = logger
        self.accountant = accountant
        self.dt = decision_tracer
        self.llm = llm or SimulatedLLM()
        self.pricing = pricing or DEFAULT_PRICING

    # -------------------------------------------------------------------
    def _llm_span(self, trace: Trace, parent: Span,
                  call_name: str, query: str) -> Span:
        span = trace.new_span("llm_call", call_name, parent=parent)
        self.log.log_llm_call(
            trace_id=trace.trace_id,
            span_id=span.span_id,
            model=SimulatedLLM.MODEL,
            messages_count=1,
            estimated_tokens=SimulatedLLM.INPUT_TOKENS,
        )
        resp = self.llm.chat(query)
        span.model = resp["model"]
        span.input_tokens = resp["input_tokens"]
        span.output_tokens = resp["output_tokens"]
        span.tokens_used = resp["input_tokens"] + resp["output_tokens"]
        span.cost = _compute_cost(
            resp["model"], resp["input_tokens"],
            resp["output_tokens"], self.pricing,
        )
        span.finish(output_data={"content": resp["content"]})
        self.log.log_llm_response(
            trace_id=trace.trace_id,
            span_id=span.span_id,
            model=resp["model"],
            input_tokens=resp["input_tokens"],
            output_tokens=resp["output_tokens"],
            latency_ms=span.duration_ms,
            has_tool_calls=bool(resp.get("tool_calls")),
        )
        return span

    def run(self, query: str,
            user_id: str = "demo_user",
            session_id: str = "demo_session") -> dict:
        trace = self.collector.start_trace(
            user_query=query,
            user_id=user_id,
            session_id=session_id,
        )
        try:
            # --- Planning span -------------------------------------------
            plan_span = trace.new_span("planning", "generate_plan")
            self.dt.capture(
                step="plan",
                context={"query": query},
                options=["direct_answer", "tool_use"],
                chosen="tool_use",
                reasoning="Query requires external data lookup",
            )
            plan_span.finish(output_data={"steps": 2})

            # --- Execution span (parent for tool + second LLM call) -------
            exec_span = trace.new_span("execution", "execute_plan",
                                       parent=plan_span)

            # Tool call
            tool_span = trace.new_span("tool_call", "get_data",
                                       parent=exec_span)
            self.log.log_tool_execution(
                trace_id=trace.trace_id,
                span_id=tool_span.span_id,
                tool_name="get_data",
                params_summary=f"query={query[:30]}",
            )
            time.sleep(0.02)
            tool_result = {"data": "mock_result"}
            tool_span.finish(output_data=tool_result)
            self.log.log_tool_result(
                trace_id=trace.trace_id,
                span_id=tool_span.span_id,
                tool_name="get_data",
                success=True,
                result_summary=str(tool_result),
                latency_ms=tool_span.duration_ms,
            )

            # Second LLM call to synthesise answer
            synth_span = self._llm_span(trace, exec_span, "synthesise", query)
            exec_span.finish(output_data={"tool_calls": 1, "llm_calls": 1})

            # --- Generation span -----------------------------------------
            gen_span = self._llm_span(trace, exec_span, "generate_answer", query)
            gen_span.finish()

            answer = f"Answer to '{query[:40]}': {synth_span.output_data['content']}"
            trace.finish()
            self.collector.end_trace(trace)
            self.metrics.record(trace)
            self.accountant.record(trace, user_id, session_id)
            return {"answer": answer, "trace_id": trace.trace_id}

        except Exception as exc:
            # Ensure partial trace is still exported
            if trace.spans:
                trace.spans[-1].finish(status="error",
                                       error_message=str(exc))
            self.log.log_error(
                trace_id=trace.trace_id,
                error=exc,
                context={"query": query, "user_id": user_id},
            )
            trace.finish()
            self.collector.end_trace(trace)
            self.metrics.record(trace)
            raise


# ===========================================================================
# Demo
# ===========================================================================

def run_demo() -> None:  # noqa: C901
    print("=" * 60)
    print("AGENT OBSERVABILITY DEMO")
    print("=" * 60)

    # Wire up all components
    collector = TraceCollector(exporter=ConsoleExporter())
    metrics = AgentMetrics()
    logger = AgentLogger(log_level="WARNING")  # quiet for demo clarity
    accountant = TokenAccountant()
    dt = DecisionTracer()

    accountant.set_budget_alert("alice", max_cost=0.01)

    # Agent that fails on the word "CRASH"
    agent = ObservableAgent(
        collector=collector,
        metrics=metrics,
        logger=logger,
        accountant=accountant,
        decision_tracer=dt,
        llm=SimulatedLLM(fail_on="CRASH"),
    )

    queries = [
        ("Compare AAPL vs MSFT stock performance", "alice", "s1"),
        ("Summarise the latest earnings report",    "bob",   "s2"),
        ("CRASH: trigger a deliberate failure",     "alice", "s1"),
    ]

    failing_trace_id: str | None = None

    for query, uid, sid in queries:
        print(f"\n>>> Running: {query!r}")
        try:
            result = agent.run(query, user_id=uid, session_id=sid)
            print(f"    ✓ {result['answer'][:80]}")
        except Exception as exc:
            # Collect the trace_id for the failure so we can inspect it
            traces = collector.query_traces(user_id=uid, status="error")
            if traces:
                failing_trace_id = traces[0].trace_id
            print(f"    ✗ Error: {exc}")

    # ----- Full trace for the failing query -----
    if failing_trace_id:
        print("\n" + "─" * 60)
        print("FULL TRACE FOR FAILING QUERY")
        print("─" * 60)
        t = collector.get_trace(failing_trace_id)
        if t:
            print(json.dumps(t.to_dict(), indent=2))

    # ----- Decision trail -----
    print("\n" + "─" * 60)
    print("DECISION TRAIL")
    print("─" * 60)
    print(dt.replay())

    divergence = dt.find_divergence("direct_answer")
    print("Divergence check:", divergence)

    # ----- Metrics summary -----
    print("\n" + "─" * 60)
    print("METRICS SUMMARY (all requests)")
    print("─" * 60)
    summary = metrics.get_summary()
    print(json.dumps(summary.to_dict(), indent=2))

    # Simulate an error spike for anomaly detection demo
    print("\n--- Simulating error spike for anomaly detection ---")
    for _ in range(20):
        bad_trace = collector.start_trace("bad", user_id="test")
        bad_span = bad_trace.new_span("llm_call", "fail")
        bad_span.finish(status="error", error_message="simulated")
        bad_trace.finish()
        metrics.record(bad_trace)

    alerts = metrics.detect_anomalies()
    print(f"\nAlerts triggered: {len(alerts)}")
    for alert in alerts:
        print(f"  {alert}")

    # ----- Daily cost report -----
    print("\n" + "─" * 60)
    print("DAILY COST REPORT")
    print("─" * 60)
    report = accountant.get_daily_cost_report()
    print(json.dumps(report, indent=2))

    print("\n" + "=" * 60)
    print("Demo complete.")


if __name__ == "__main__":
    run_demo()
