"""Real-time terminal dashboard for agent observability.

Uses the ``rich`` library to render a live, auto-refreshing metrics panel.

Install::

    pip install rich

Usage::

    python dashboard.py

See: docs/05-the-tool-ecosystem/03-agent-observability.md
"""

from __future__ import annotations

import sys
import os
import time
import random
import threading
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from agent_observability import (
    AgentMetrics,
    TokenAccountant,
    TraceCollector,
    DecisionTracer,
    AgentLogger,
    ObservableAgent,
    SimulatedLLM,
    Trace,
    Span,
    DEFAULT_PRICING,
    _compute_cost,
)


def _try_import_rich():
    try:
        import rich                    # noqa: F401
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        from rich.layout import Layout
        from rich.live import Live
        from rich.text import Text
        from rich import box
        return Console, Table, Panel, Layout, Live, Text, box
    except ImportError:
        return None


# ===========================================================================
# AgentDashboard
# ===========================================================================

class AgentDashboard:
    """Real-time terminal dashboard for agent observability.

    Renders a live panel every ``refresh_interval`` seconds showing:

    * Rolling request/latency/token/cost metrics
    * Per-model cost breakdown
    * Recent trace history with status icons
    * Active anomaly alerts
    """

    def __init__(
        self,
        metrics: AgentMetrics,
        token_accountant: TokenAccountant,
        collector: TraceCollector,
        refresh_interval: float = 2.0,
    ):
        self.metrics = metrics
        self.accountant = token_accountant
        self.collector = collector
        self.refresh_interval = refresh_interval
        self._running = False
        self._recent_traces: list[Trace] = []
        self._lock = threading.Lock()

    # -------------------------------------------------------------------
    def push_trace(self, trace: Trace) -> None:
        """Register a trace so it appears in the dashboard's recent list."""
        with self._lock:
            self._recent_traces.append(trace)
            # Keep only the 10 most recent
            if len(self._recent_traces) > 10:
                self._recent_traces = self._recent_traces[-10:]

    # -------------------------------------------------------------------
    def render(self) -> str:
        """Return a plain-text snapshot of the current dashboard state.

        Used when ``rich`` is not installed, or for testing.
        """
        summary = self.metrics.get_summary()
        report = self.accountant.get_daily_cost_report()
        alerts = self.metrics.detect_anomalies()

        width = 51
        border = "─" * width
        lines = [
            f"┌{border}┐",
            f"│{'AGENT OBSERVABILITY DASHBOARD':^{width}}│",
            f"├{border}┤",
            f"│ Requests: {summary.requests:<6}  Error rate: {summary.error_rate_pct:>5.1f}%{'':<8}│",
            f"│ Avg Latency: {summary.avg_latency_ms:>7.0f}ms  P95: {summary.p95_latency_ms:>7.0f}ms{'':<4}│",
            f"│ Avg Tokens:  {summary.avg_tokens:>7.0f}   Avg Cost: ${summary.avg_cost:>7.4f}{'':<4}│",
            f"│ Total Cost Today: ${report['total_cost']:>8.4f}{'':<18}│",
            f"├{border}┤",
            f"│{'Model Usage':^{width}}│",
        ]

        by_model = report.get("by_model", {})
        if by_model:
            for model, stats in list(by_model.items())[:4]:
                short = model[:20]
                row = f"│  {short:<22} {stats['calls']:>5} calls  ${stats['cost']:>7.4f}   │"
                lines.append(row)
        else:
            lines.append(f"│{'(no data yet)':^{width}}│")

        lines.append(f"├{border}┤")
        lines.append(f"│{'Recent Traces':^{width}}│")

        with self._lock:
            recent = list(self._recent_traces)[-5:]

        if recent:
            for t in reversed(recent):
                icon = "✓" if not t.has_error else "✗"
                tid = t.trace_id[:8]
                dur = f"{t.duration_ms:.0f}ms"
                err = "ERROR" if t.has_error else dur
                row = (f"│  {icon} {tid}  {t.llm_call_count} LLM  "
                       f"{t.tool_call_count} tools  {err:<8}   │")
                lines.append(row)
        else:
            lines.append(f"│{'(no traces yet)':^{width}}│")

        lines.append(f"├{border}┤")
        alert_str = ", ".join(str(a) for a in alerts) if alerts else "None"
        lines.append(f"│ Alerts: {alert_str[:width-9]:<{width-9}}│")
        lines.append(f"└{border}┘")

        return "\n".join(lines)

    # -------------------------------------------------------------------
    def _render_rich(self):
        """Build a Rich renderable for live display."""
        rich_imports = _try_import_rich()
        if not rich_imports:
            return self.render()

        Console, Table, Panel, Layout, Live, Text, box = rich_imports

        summary = self.metrics.get_summary()
        report  = self.accountant.get_daily_cost_report()
        alerts  = self.metrics.detect_anomalies()

        # ── Header metrics ──────────────────────────────────────────────
        stats_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
        stats_table.add_column("Key", style="bold cyan")
        stats_table.add_column("Value", style="white")
        stats_table.add_row("Requests", str(summary.requests))
        stats_table.add_row("Error rate", f"{summary.error_rate_pct:.1f}%")
        stats_table.add_row("Avg latency", f"{summary.avg_latency_ms:.0f}ms")
        stats_table.add_row("P95 latency", f"{summary.p95_latency_ms:.0f}ms")
        stats_table.add_row("Avg tokens", f"{summary.avg_tokens:.0f}")
        stats_table.add_row("Avg cost", f"${summary.avg_cost:.4f}")
        stats_table.add_row("Total cost today", f"${report['total_cost']:.4f}")

        # ── Model breakdown ──────────────────────────────────────────────
        model_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        model_table.add_column("Model", style="bold")
        model_table.add_column("Calls", justify="right")
        model_table.add_column("Cost", justify="right", style="green")
        for model, stats in report.get("by_model", {}).items():
            model_table.add_row(model, str(stats["calls"]),
                                f"${stats['cost']:.4f}")

        # ── Recent traces ────────────────────────────────────────────────
        traces_table = Table(box=box.SIMPLE, show_header=True, padding=(0, 1))
        traces_table.add_column("", width=2)
        traces_table.add_column("Trace ID", style="dim")
        traces_table.add_column("LLM", justify="right")
        traces_table.add_column("Tools", justify="right")
        traces_table.add_column("Latency", justify="right")
        traces_table.add_column("Status")

        with self._lock:
            recent = list(self._recent_traces)[-5:]

        for t in reversed(recent):
            icon = Text("✓", style="green") if not t.has_error else Text("✗", style="red")
            status = Text("ERROR", style="red") if t.has_error else Text("ok", style="green")
            traces_table.add_row(icon, t.trace_id[:8],
                                 str(t.llm_call_count),
                                 str(t.tool_call_count),
                                 f"{t.duration_ms:.0f}ms",
                                 status)

        # ── Alerts ───────────────────────────────────────────────────────
        if alerts:
            alert_text = "\n".join(str(a) for a in alerts)
            alert_style = "bold red"
        else:
            alert_text = "None"
            alert_style = "green"

        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        from rich.columns import Columns  # noqa: PLC0415
        from rich.rule import Rule        # noqa: PLC0415
        from rich import print as rprint  # noqa: PLC0415

        layout = (
            Panel(stats_table,   title="Metrics",        border_style="blue") ,
            Panel(model_table,   title="Model Usage",    border_style="cyan"),
            Panel(traces_table,  title="Recent Traces",  border_style="magenta"),
            Panel(Text(alert_text, style=alert_style),
                  title="Alerts", border_style="red" if alerts else "green"),
        )

        from rich.console import Group  # noqa: PLC0415
        return Panel(
            Group(*layout),
            title=f"[bold]AGENT OBSERVABILITY[/bold]  [dim]{now}[/dim]",
            border_style="bright_blue",
        )

    # -------------------------------------------------------------------
    def start(self, duration_seconds: float | None = None) -> None:
        """Start the live dashboard.

        Blocks until ``duration_seconds`` elapses (for demos) or until
        the user presses Ctrl-C.
        """
        self._running = True
        rich_imports = _try_import_rich()

        if not rich_imports:
            # Fallback: plain-text polling loop
            print("(rich not installed — using plain-text mode)\n")
            start = time.time()
            while self._running:
                print("\033[2J\033[H")  # clear screen
                print(self.render())
                time.sleep(self.refresh_interval)
                if duration_seconds and (time.time() - start) >= duration_seconds:
                    break
            return

        Console, Table, Panel, Layout, Live, Text, box = rich_imports
        from rich.live import Live as RichLive  # noqa: PLC0415

        start = time.time()
        try:
            with RichLive(
                self._render_rich(),
                refresh_per_second=max(1, int(1 / self.refresh_interval)),
                screen=False,
            ) as live:
                while self._running:
                    time.sleep(self.refresh_interval)
                    live.update(self._render_rich())
                    if duration_seconds and (time.time() - start) >= duration_seconds:
                        break
        except KeyboardInterrupt:
            pass
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False


# ===========================================================================
# Demo
# ===========================================================================

def run_demo() -> None:
    """
    Simulates 100 agent requests in a background thread while the dashboard
    renders live. After 50 normal requests an error spike is injected to
    demonstrate anomaly detection.
    """
    metrics    = AgentMetrics(window_size=1000)
    accountant = TokenAccountant()
    collector  = TraceCollector()
    logger     = AgentLogger(log_level="WARNING")
    dt         = DecisionTracer()

    dashboard = AgentDashboard(
        metrics=metrics,
        token_accountant=accountant,
        collector=collector,
        refresh_interval=1.0,
    )

    models = ["gpt-4o", "gpt-4o-mini"]
    users  = ["alice", "bob", "charlie"]

    def _make_trace(fail: bool = False, model: str = "gpt-4o-mini") -> Trace:
        trace = collector.start_trace(
            user_query="demo query",
            user_id=random.choice(users),
            session_id="dashboard_demo",
        )
        span = trace.new_span("llm_call", "generate")
        span.model = model
        span.input_tokens  = random.randint(400, 1200)
        span.output_tokens = random.randint(50, 300)
        span.tokens_used   = span.input_tokens + span.output_tokens
        span.cost = _compute_cost(
            model, span.input_tokens, span.output_tokens, DEFAULT_PRICING
        )
        if fail:
            span.finish(status="error", error_message="simulated error")
        else:
            # Simulate variable latency
            time.sleep(random.uniform(0.01, 0.05))
            span.finish()
        trace.finish()
        return trace

    TOTAL_REQUESTS = 80
    ERROR_SPIKE_AT = 60   # inject errors after request #60

    def _background_load():
        for i in range(TOTAL_REQUESTS):
            model = random.choice(models)
            fail  = i >= ERROR_SPIKE_AT and random.random() < 0.50
            t = _make_trace(fail=fail, model=model)
            metrics.record(t)
            accountant.record(t, t.user_id or "unknown", t.session_id or "s")
            dashboard.push_trace(t)
            time.sleep(0.15)

    thread = threading.Thread(target=_background_load, daemon=True)
    thread.start()

    print("Starting dashboard (Ctrl-C to exit, or waits ~15 seconds)…\n")
    dashboard.start(duration_seconds=15.0)
    thread.join(timeout=1)

    print("\n--- Final metrics summary ---")
    print(dashboard.render())

    alerts = metrics.detect_anomalies()
    if alerts:
        print(f"\nAlerts triggered ({len(alerts)}):")
        for alert in alerts:
            print(f"  {alert}")
    else:
        print("\nNo anomalies detected.")


if __name__ == "__main__":
    run_demo()
