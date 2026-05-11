"""Real-time harness monitoring and alerting.

Collects metrics from every harness response, maintains sliding-window
aggregates, and fires alerts when health thresholds are breached.

Key concepts:
    HarnessMetrics  — raw collection: latencies, tokens, costs, decisions
    HarnessMonitor  — computes dashboard data from metrics
    HarnessAlerter  — checks thresholds and dispatches Alert objects
    AlertHandler    — pluggable sink (console, Slack, PagerDuty, …)

See: docs/07-harness-engineering/01-the-harness-mindset.md
"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Import HarnessResponse — fall back to a local stub if module absent
# ---------------------------------------------------------------------------

try:
    from harness_state_machine import HarnessResponse  # type: ignore
except ImportError:
    @dataclass
    class HarnessResponse:  # type: ignore[no-redef]
        content: str = ""
        state_trace: list[str] = field(default_factory=list)
        decisions_made: list[dict] = field(default_factory=list)
        tokens_used: int = 0
        cost: float = 0.0
        duration_ms: float = 0.0
        final_state: str = "respond"


# ---------------------------------------------------------------------------
# Alert model
# ---------------------------------------------------------------------------

class AlertSeverity(Enum):
    INFO     = "info"
    WARNING  = "warning"
    CRITICAL = "critical"


@dataclass
class Alert:
    severity: AlertSeverity
    title: str
    detail: str
    metric: str
    value: float
    threshold: float
    fired_at: float = field(default_factory=time.time)

    def __str__(self) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(self.fired_at))
        return (f"[{self.severity.value.upper()}] {ts} "
                f"{self.title}: {self.detail} "
                f"(value={self.value:.3f}, threshold={self.threshold:.3f})")


# ---------------------------------------------------------------------------
# Alert handlers
# ---------------------------------------------------------------------------

class AlertHandler:
    def handle(self, alert: Alert) -> None:
        raise NotImplementedError


class ConsoleAlertHandler(AlertHandler):
    def handle(self, alert: Alert) -> None:
        print(f"  🚨 {alert}")


class LogAlertHandler(AlertHandler):
    def handle(self, alert: Alert) -> None:
        level = {
            AlertSeverity.INFO:     logging.INFO,
            AlertSeverity.WARNING:  logging.WARNING,
            AlertSeverity.CRITICAL: logging.CRITICAL,
        }[alert.severity]
        logger.log(level, str(alert))


# ---------------------------------------------------------------------------
# Per-state latency tracker (used inside HarnessMetrics)
# ---------------------------------------------------------------------------

class _Percentiles:
    """Keep a bounded window of float samples and compute percentiles."""

    def __init__(self, maxlen: int = 1000) -> None:
        self._data: deque[float] = deque(maxlen=maxlen)

    def add(self, value: float) -> None:
        self._data.append(value)

    def percentile(self, p: float) -> float:
        """Return the p-th percentile (0–100) or 0.0 if empty."""
        if not self._data:
            return 0.0
        sorted_data = sorted(self._data)
        idx = (p / 100.0) * (len(sorted_data) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return sorted_data[lo]
        return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])

    def mean(self) -> float:
        return sum(self._data) / len(self._data) if self._data else 0.0

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Metrics collector
# ---------------------------------------------------------------------------

class HarnessMetrics:
    """Collect and aggregate harness metrics.

    All windows are bounded (deque with maxlen) so memory use is constant.
    """

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = window_size

        # Latency
        self.request_latencies = _Percentiles(window_size)
        self.state_latencies: dict[str, _Percentiles] = defaultdict(
            lambda: _Percentiles(window_size)
        )

        # Decision counters
        self.decisions: dict[str, int] = defaultdict(int)

        # Provider outcomes
        self.provider_successes: dict[str, int] = defaultdict(int)
        self.provider_failures: dict[str, int] = defaultdict(int)

        # Token / cost
        self.tokens: deque[int] = deque(maxlen=window_size)
        self.costs: deque[float] = deque(maxlen=window_size)

        # Final-state distribution
        self.final_states: dict[str, int] = defaultdict(int)

        # Handler distribution
        self.handler_calls: dict[str, int] = defaultdict(int)

        # Timestamps for throughput calculation
        self.request_timestamps: deque[float] = deque(maxlen=window_size)

    def record(self, response: HarnessResponse) -> None:
        now = time.time()
        self.request_timestamps.append(now)
        self.request_latencies.add(response.duration_ms)
        self.tokens.append(response.tokens_used)
        self.costs.append(response.cost)
        self.final_states[response.final_state] += 1

        # Count decision event types
        for dec in response.decisions_made:
            event = dec.get("event", "unknown")
            self.decisions[event] += 1

            # Provider info (from execution events)
            if event == "execution":
                handler = dec.get("handler", "unknown")
                self.handler_calls[handler] += 1

            # Input/output guardrail outcomes
            if event in ("input_validation", "output_validation"):
                result = dec.get("result", "unknown")
                self.decisions[f"{event}.{result}"] += 1

    def get_summary(self, window_seconds: int = 300) -> dict:
        """Compute summary statistics over the most recent *window_seconds*."""
        now = time.time()
        cutoff = now - window_seconds

        # Requests in window
        recent_count = sum(
            1 for t in self.request_timestamps if t >= cutoff
        )
        rpm = (recent_count / window_seconds) * 60 if window_seconds else 0

        # Cost in window
        # (We approximate by taking the last N items matching recent_count)
        recent_costs = list(self.costs)[-recent_count:] if recent_count else []
        cost_per_min = (sum(recent_costs) / window_seconds) * 60 \
            if window_seconds and recent_costs else 0.0

        total_requests = sum(self.final_states.values()) or 1
        reject_count   = (self.final_states.get("reject", 0) +
                           self.final_states.get("error", 0))
        timeout_count  = self.final_states.get("timeout", 0)

        return {
            "throughput": {
                "requests_per_minute": round(rpm, 1),
                "total_requests": sum(self.final_states.values()),
                "handler_distribution": dict(self.handler_calls),
            },
            "latency_ms": {
                "p50":  round(self.request_latencies.percentile(50), 1),
                "p95":  round(self.request_latencies.percentile(95), 1),
                "p99":  round(self.request_latencies.percentile(99), 1),
                "mean": round(self.request_latencies.mean(), 1),
            },
            "guardrails": {
                "input_rejection_rate": round(
                    self.decisions.get("input_validation.rejected", 0)
                    / total_requests, 4),
                "output_block_rate": round(
                    self.decisions.get("output_validation.blocked", 0)
                    / total_requests, 4),
                "human_approval_count": self.decisions.get("human_approval", 0),
            },
            "reliability": {
                "timeout_rate": round(timeout_count / total_requests, 4),
                "error_rate":   round(reject_count  / total_requests, 4),
                "final_states": dict(self.final_states),
            },
            "cost": {
                "cost_per_minute": round(cost_per_min, 4),
                "total_cost":      round(sum(self.costs), 4),
                "avg_tokens_per_request": round(
                    sum(self.tokens) / len(self.tokens), 1
                ) if self.tokens else 0.0,
            },
        }


# ---------------------------------------------------------------------------
# Alerter
# ---------------------------------------------------------------------------

# Thresholds
_ALERT_THRESHOLDS = {
    # CRITICAL
    "output_block_rate":           (0.05,   AlertSeverity.CRITICAL),
    "timeout_rate":                (0.10,   AlertSeverity.CRITICAL),
    # WARNING
    "input_rejection_rate":        (0.02,   AlertSeverity.WARNING),
    "error_rate":                  (0.05,   AlertSeverity.WARNING),
    "p95_latency_ms":              (5000.0, AlertSeverity.WARNING),
    "cost_per_minute":             (1.0,    AlertSeverity.WARNING),
}


class HarnessAlerter:
    """Check metric thresholds and dispatch Alert objects to handlers."""

    def __init__(self, handlers: list[AlertHandler] | None = None) -> None:
        self.handlers = handlers or [ConsoleAlertHandler()]
        self._fired: set[str] = set()   # Deduplicate within a session

    def check(self, metrics: HarnessMetrics) -> list[Alert]:
        """Evaluate all alert conditions; return newly fired alerts."""
        summary = metrics.get_summary()
        alerts: list[Alert] = []

        checks = {
            "output_block_rate":    summary["guardrails"]["output_block_rate"],
            "timeout_rate":         summary["reliability"]["timeout_rate"],
            "input_rejection_rate": summary["guardrails"]["input_rejection_rate"],
            "error_rate":           summary["reliability"]["error_rate"],
            "p95_latency_ms":       summary["latency_ms"]["p95"],
            "cost_per_minute":      summary["cost"]["cost_per_minute"],
        }

        for metric, value in checks.items():
            if metric not in _ALERT_THRESHOLDS:
                continue
            threshold, severity = _ALERT_THRESHOLDS[metric]
            if value > threshold:
                alert = Alert(
                    severity=severity,
                    title=f"Harness alert: {metric}",
                    detail=f"{metric} is {value:.4f}",
                    metric=metric,
                    value=value,
                    threshold=threshold,
                )
                alerts.append(alert)

        return alerts

    def fire(self, alert: Alert) -> None:
        """Dispatch *alert* to all registered handlers."""
        for handler in self.handlers:
            handler.handle(alert)

    def check_and_fire(self, metrics: HarnessMetrics) -> list[Alert]:
        """Convenience: check and immediately fire any triggered alerts."""
        alerts = self.check(metrics)
        for alert in alerts:
            self.fire(alert)
        return alerts


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class HarnessMonitor:
    """Orchestrates metrics collection and alerting.

    Usage::

        monitor = HarnessMonitor()
        # After each harness call:
        monitor.record_request(response)
        # Periodically:
        dashboard = monitor.get_dashboard_data()
        alerts = monitor.check_alerts()
    """

    def __init__(
        self,
        alert_handlers: list[AlertHandler] | None = None,
        window_size: int = 1000,
    ) -> None:
        self.metrics  = HarnessMetrics(window_size)
        self.alerter  = HarnessAlerter(alert_handlers)

    def record_request(self, response: HarnessResponse) -> None:
        self.metrics.record(response)

    def get_dashboard_data(self) -> dict:
        return self.metrics.get_summary()

    def check_alerts(self) -> list[Alert]:
        return self.alerter.check_and_fire(self.metrics)

    def print_dashboard(self) -> None:
        """Print a formatted text dashboard to stdout."""
        data = self.get_dashboard_data()
        width = 60

        def _section(title: str) -> None:
            print(f"\n{'─' * width}")
            print(f"  {title}")
            print(f"{'─' * width}")

        def _row(label: str, value: Any, unit: str = "") -> None:
            val_str = f"{value}{unit}"
            print(f"  {label:<36s} {val_str:>18s}")

        print("=" * width)
        print(f"  HARNESS MONITOR DASHBOARD")
        print(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * width)

        tp = data["throughput"]
        _section("THROUGHPUT")
        _row("Requests / minute",    tp["requests_per_minute"], " rpm")
        _row("Total requests",       tp["total_requests"])
        for handler, count in tp["handler_distribution"].items():
            _row(f"  └─ {handler}", count)

        lat = data["latency_ms"]
        _section("LATENCY")
        _row("P50",  lat["p50"],  " ms")
        _row("P95",  lat["p95"],  " ms")
        _row("P99",  lat["p99"],  " ms")
        _row("Mean", lat["mean"], " ms")

        gr = data["guardrails"]
        _section("GUARDRAILS")
        _row("Input rejection rate",
             f"{gr['input_rejection_rate']*100:.2f}", "%")
        _row("Output block rate",
             f"{gr['output_block_rate']*100:.2f}", "%")
        _row("Human approvals (total)", gr["human_approval_count"])

        rel = data["reliability"]
        _section("RELIABILITY")
        _row("Timeout rate", f"{rel['timeout_rate']*100:.2f}", "%")
        _row("Error rate",   f"{rel['error_rate']*100:.2f}", "%")
        for state, count in rel["final_states"].items():
            _row(f"  └─ {state}", count)

        cost = data["cost"]
        _section("COST")
        _row("Cost / minute",           f"${cost['cost_per_minute']:.4f}")
        _row("Total cost",              f"${cost['total_cost']:.4f}")
        _row("Avg tokens / request",    cost["avg_tokens_per_request"])

        print("=" * width)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _simulate_request(
    *,
    final_state: str = "respond",
    duration_ms: float | None = None,
    tokens: int | None = None,
    rejected_input: bool = False,
    blocked_output: bool = False,
    timeout: bool = False,
) -> HarnessResponse:
    """Generate a synthetic HarnessResponse for simulation."""
    rng = random.random
    decisions: list[dict] = []

    if rejected_input:
        decisions.append({"event": "input_validation", "result": "rejected"})
    elif blocked_output:
        decisions.append({"event": "input_validation", "result": "passed"})
        decisions.append({"event": "route_decision", "route": "agent"})
        decisions.append({"event": "execution", "handler": "agent"})
        decisions.append({"event": "output_validation", "result": "blocked"})
    else:
        decisions.append({"event": "input_validation", "result": "passed"})
        decisions.append({
            "event": "route_decision",
            "route": random.choice(["simple_chat", "agent", "rag"]),
        })
        decisions.append({
            "event": "execution",
            "handler": random.choice(["simple_chat", "agent", "rag"]),
        })
        decisions.append({"event": "output_validation", "result": "passed"})

    trace: list[str] = ["validate_input", "route", "execute",
                        "validate_output", final_state]

    return HarnessResponse(
        content="Simulated response.",
        state_trace=trace,
        decisions_made=decisions,
        tokens_used=tokens if tokens is not None else int(rng() * 300 + 50),
        cost=(tokens or 150) / 1_000_000 * 5.0,
        duration_ms=duration_ms if duration_ms is not None else rng() * 500 + 50,
        final_state=final_state,
    )


def _run_demo() -> None:
    logging.basicConfig(level=logging.WARNING)
    random.seed(42)

    print("=" * 60)
    print("HARNESS MONITOR DEMO")
    print("=" * 60)
    print("\nSimulating 500 normal requests …")

    monitor = HarnessMonitor()

    # Phase 1: normal operation
    for _ in range(500):
        r = _simulate_request(duration_ms=random.gauss(300, 80))
        monitor.record_request(r)

    print("\n[Phase 1] Normal operation dashboard:")
    monitor.print_dashboard()
    alerts = monitor.check_alerts()
    if not alerts:
        print("\n  No alerts — system healthy.")

    # Phase 2: inject anomalies
    print("\n\nSimulating anomalies: timeouts, rejected inputs, blocked outputs …")
    for i in range(500):
        roll = random.random()
        if roll < 0.12:      # 12% timeout
            r = _simulate_request(final_state="timeout", duration_ms=30_000,
                                   timeout=True)
        elif roll < 0.18:    # 6% input rejection
            r = _simulate_request(final_state="reject", rejected_input=True,
                                   duration_ms=5)
        elif roll < 0.26:    # 8% output blocked
            r = _simulate_request(final_state="reject", blocked_output=True,
                                   duration_ms=200)
        else:
            r = _simulate_request()
        monitor.record_request(r)

    print("\n[Phase 2] After anomalies dashboard:")
    monitor.print_dashboard()

    print("\n[Phase 2] Alerts:")
    alerts = monitor.check_alerts()
    if not alerts:
        print("  No alerts.")
    else:
        print(f"  {len(alerts)} alert(s) fired.")

    print("\n" + "=" * 60)
    print("Demo complete.")


if __name__ == "__main__":
    _run_demo()
