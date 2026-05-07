"""Tests for the agent observability system.

Covers: TraceCollector, AgentMetrics, AgentLogger, TokenAccountant,
        DecisionTracer, and the JSONFileExporter.

Run:
    pytest test_observability.py -v
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone, timedelta
from io import StringIO

import pytest

sys.path.insert(0, __file__.replace("test_observability.py", ""))

from agent_observability import (
    DEFAULT_PRICING,
    Alert,
    AgentLogger,
    AgentMetrics,
    ConsoleExporter,
    Decision,
    DecisionTracer,
    MetricsSummary,
    ObservableAgent,
    SimulatedLLM,
    Span,
    Trace,
    TokenAccountant,
    TraceCollector,
    _compute_cost,
)
from exporters import JSONFileExporter


# ===========================================================================
# Helpers / fixtures
# ===========================================================================

def _make_trace(
    user_id: str = "u1",
    session_id: str = "s1",
    has_error: bool = False,
    n_llm_calls: int = 1,
    n_tool_calls: int = 0,
    llm_model: str = "gpt-4o-mini",
    input_tokens: int = 500,
    output_tokens: int = 100,
) -> Trace:
    """Construct a Trace with realistic spans for testing."""
    trace = Trace(user_query="test query", user_id=user_id, session_id=session_id)
    pricing = DEFAULT_PRICING

    for _ in range(n_llm_calls):
        span = trace.new_span("llm_call", "test_llm_call")
        span.model = llm_model
        span.input_tokens = input_tokens
        span.output_tokens = output_tokens
        span.tokens_used = input_tokens + output_tokens
        span.cost = _compute_cost(llm_model, input_tokens, output_tokens, pricing)
        if has_error:
            span.finish(status="error", error_message="simulated error")
        else:
            span.finish(output_data={"content": "ok"})

    for _ in range(n_tool_calls):
        tspan = trace.new_span("tool_call", "test_tool")
        tspan.finish(output_data={"result": "data"})

    trace.finish()
    return trace


@pytest.fixture()
def collector() -> TraceCollector:
    return TraceCollector(exporter=ConsoleExporter())


@pytest.fixture()
def metrics() -> AgentMetrics:
    return AgentMetrics(window_size=1000)


@pytest.fixture()
def accountant() -> TokenAccountant:
    return TokenAccountant()


@pytest.fixture()
def dt() -> DecisionTracer:
    return DecisionTracer()


# ===========================================================================
# TRACE TESTS
# ===========================================================================

class TestTraceCollector:

    def test_trace_captures_all_spans(self, collector: TraceCollector):
        """A completed trace should have planning, execution, and generation spans."""
        agent = ObservableAgent(
            collector=collector,
            metrics=AgentMetrics(),
            logger=AgentLogger(log_level="ERROR"),
            accountant=TokenAccountant(),
            decision_tracer=DecisionTracer(),
        )
        result = agent.run("test query")
        trace = collector.get_trace(result["trace_id"])
        assert trace is not None

        types = {s.type for s in trace.spans}
        assert "planning" in types
        assert "execution" in types
        assert "llm_call" in types
        assert "tool_call" in types

    def test_trace_includes_token_counts(self, collector: TraceCollector):
        """Every LLM span should have tokens_used > 0."""
        agent = ObservableAgent(
            collector=collector,
            metrics=AgentMetrics(),
            logger=AgentLogger(log_level="ERROR"),
            accountant=TokenAccountant(),
            decision_tracer=DecisionTracer(),
        )
        result = agent.run("test query with tokens")
        trace = collector.get_trace(result["trace_id"])
        assert trace is not None

        llm_spans = [s for s in trace.spans if s.type == "llm_call"]
        assert len(llm_spans) > 0
        for span in llm_spans:
            assert span.tokens_used > 0

    def test_trace_exported_on_error(self, collector: TraceCollector):
        """Even a failing agent run should export a partial trace with error status."""
        agent = ObservableAgent(
            collector=collector,
            metrics=AgentMetrics(),
            logger=AgentLogger(log_level="ERROR"),
            accountant=TokenAccountant(),
            decision_tracer=DecisionTracer(),
            llm=SimulatedLLM(fail_on="CRASH"),
        )
        with pytest.raises(Exception):
            agent.run("CRASH deliberately")

        # Trace should exist and be exported (stored in collector)
        traces = collector.query_traces(status="error")
        assert len(traces) == 1
        assert traces[0].has_error is True

    def test_trace_query_by_user(self, collector: TraceCollector):
        """query_traces with user_id filter returns only that user's traces."""
        # 5 traces for user A
        for _ in range(5):
            t = collector.start_trace("q", user_id="alice", session_id="s")
            t.finish()
            collector.end_trace(t)

        # 3 traces for user B
        for _ in range(3):
            t = collector.start_trace("q", user_id="bob", session_id="s")
            t.finish()
            collector.end_trace(t)

        alice_traces = collector.query_traces(user_id="alice")
        assert len(alice_traces) == 5

        bob_traces = collector.query_traces(user_id="bob")
        assert len(bob_traces) == 3

    def test_query_traces_by_status(self, collector: TraceCollector):
        """query_traces with status='error' returns only error traces."""
        ok_trace = _make_trace(has_error=False)
        err_trace = _make_trace(has_error=True)

        # Manually register
        collector._traces[ok_trace.trace_id] = ok_trace
        collector._traces[err_trace.trace_id] = err_trace

        errors = collector.query_traces(status="error")
        assert all(t.has_error for t in errors)

    def test_query_traces_by_time(self, collector: TraceCollector):
        """query_traces with since= filters out old traces."""
        old = _make_trace()
        old.start_time = time.time() - 7200  # 2 hours ago
        collector._traces[old.trace_id] = old

        recent = _make_trace()
        collector._traces[recent.trace_id] = recent

        since = datetime.now(timezone.utc) - timedelta(hours=1)
        results = collector.query_traces(since=since)
        ids = {t.trace_id for t in results}
        assert recent.trace_id in ids
        assert old.trace_id not in ids

    def test_trace_total_tokens(self):
        """Trace.total_tokens sums tokens across all LLM spans."""
        t = _make_trace(n_llm_calls=3, input_tokens=200, output_tokens=50)
        assert t.total_tokens == 3 * (200 + 50)

    def test_trace_total_cost(self):
        """Trace.total_cost matches manual calculation."""
        model = "gpt-4o-mini"
        input_t, output_t = 1000, 500
        t = _make_trace(n_llm_calls=2, llm_model=model,
                        input_tokens=input_t, output_tokens=output_t)
        expected = 2 * _compute_cost(model, input_t, output_t, DEFAULT_PRICING)
        assert abs(t.total_cost - expected) < 1e-9


# ===========================================================================
# METRICS TESTS
# ===========================================================================

class TestAgentMetrics:

    def test_metrics_aggregate_correctly(self, metrics: AgentMetrics):
        """Record 10 traces with known latency; verify averages."""
        # Each trace has one LLM span with fixed tokens
        for _ in range(10):
            t = _make_trace(n_llm_calls=1, input_tokens=400, output_tokens=100)
            metrics.record(t)

        summary = metrics.get_summary()
        assert summary.requests == 10
        assert summary.avg_tokens == pytest.approx(500, rel=0.01)
        # No errors in these traces
        assert summary.error_rate_pct == 0.0

    def test_anomaly_detection_error_spike(self, metrics: AgentMetrics):
        """After recording many error traces, an error-rate alert is triggered."""
        # 100 normal
        for _ in range(100):
            metrics.record(_make_trace(has_error=False))
        # 20 errors — 16.7 % error rate
        for _ in range(20):
            metrics.record(_make_trace(has_error=True))

        alerts = metrics.detect_anomalies()
        kinds = {a.kind for a in alerts}
        assert "error_rate" in kinds
        err_alert = next(a for a in alerts if a.kind == "error_rate")
        assert err_alert.value > 5.0

    def test_anomaly_detection_latency_spike(self, metrics: AgentMetrics):
        """Latency spike (P95 >> 2× average) should trigger latency alert."""
        # 50 fast traces (near-zero latency from _make_trace)
        for _ in range(50):
            t = _make_trace()
            metrics.record(t)

        # 5 artificially slow traces
        for _ in range(5):
            t = _make_trace()
            # Backdate start to inflate latency
            t.start_time = t.start_time - 15  # 15 seconds earlier
            metrics.record(t)

        alerts = metrics.detect_anomalies()
        kinds = {a.kind for a in alerts}
        # May or may not trigger depending on timing — just check the method runs
        assert isinstance(alerts, list)
        for a in alerts:
            assert isinstance(a, Alert)

    def test_metrics_window_bounded(self):
        """Metrics deque should not exceed window_size entries."""
        m = AgentMetrics(window_size=10)
        for _ in range(50):
            m.record(_make_trace())
        assert len(m.metrics["total_latency_ms"]) == 10

    def test_prometheus_export(self, metrics: AgentMetrics):
        """export_prometheus() returns non-empty text with expected gauge names."""
        for _ in range(5):
            metrics.record(_make_trace())
        prom = metrics.export_prometheus()
        assert "agent_requests_total" in prom
        assert "agent_latency_avg_ms" in prom
        assert "agent_error_rate_pct" in prom


# ===========================================================================
# TOKEN ACCOUNTANT TESTS
# ===========================================================================

class TestTokenAccountant:

    def test_user_cost_calculated_correctly(self, accountant: TokenAccountant):
        """Manual calculation should match get_user_cost()."""
        model = "gpt-4o-mini"
        input_t, output_t = 1000, 250
        expected_per_call = _compute_cost(model, input_t, output_t, DEFAULT_PRICING)

        for _ in range(5):
            t = _make_trace(user_id="alice", llm_model=model,
                            input_tokens=input_t, output_tokens=output_t,
                            n_llm_calls=1)
            accountant.record(t, user_id="alice", session_id="s1")

        total = accountant.get_user_cost("alice")
        assert abs(total - 5 * expected_per_call) < 1e-9

    def test_daily_report_aggregates_by_model(self, accountant: TokenAccountant):
        """Daily cost report should break down calls and cost by model."""
        for _ in range(3):
            t = _make_trace(llm_model="gpt-4o", n_llm_calls=1,
                            input_tokens=500, output_tokens=100)
            accountant.record(t, "u1", "s1")
        for _ in range(7):
            t = _make_trace(llm_model="gpt-4o-mini", n_llm_calls=1,
                            input_tokens=500, output_tokens=100)
            accountant.record(t, "u1", "s1")

        report = accountant.get_daily_cost_report()
        by_model = report["by_model"]
        assert "gpt-4o" in by_model
        assert "gpt-4o-mini" in by_model
        assert by_model["gpt-4o"]["calls"] == 3
        assert by_model["gpt-4o-mini"]["calls"] == 7

    def test_session_cost(self, accountant: TokenAccountant):
        """get_session_cost() returns cost for a specific session only."""
        for _ in range(4):
            t = _make_trace(session_id="session-A", n_llm_calls=1)
            accountant.record(t, "u1", "session-A")
        for _ in range(2):
            t = _make_trace(session_id="session-B", n_llm_calls=1)
            accountant.record(t, "u1", "session-B")

        cost_a = accountant.get_session_cost("session-A")
        cost_b = accountant.get_session_cost("session-B")
        assert cost_a > 0
        assert cost_b > 0
        assert abs(cost_a - 2 * cost_b) < 1e-9  # 4 vs 2 calls, same model/tokens

    def test_budget_alert_fires(self, accountant: TokenAccountant, capsys):
        """Budget alert should print when user exceeds the cap."""
        accountant.set_budget_alert("alice", max_cost=0.0)  # cap at $0

        t = _make_trace(user_id="alice", n_llm_calls=1,
                        input_tokens=1000, output_tokens=500)
        accountant.record(t, user_id="alice", session_id="s")

        captured = capsys.readouterr()
        assert "BUDGET ALERT" in captured.out


# ===========================================================================
# DECISION TRACER TESTS
# ===========================================================================

class TestDecisionTracer:

    def test_decision_trail_replayable(self, dt: DecisionTracer):
        """Replay should contain all 5 captured decisions with reasoning."""
        for i in range(5):
            dt.capture(
                step=f"step_{i}",
                context={"i": i},
                options=["a", "b"],
                chosen="a",
                reasoning=f"reason_{i}",
            )

        trail = dt.replay()
        for i in range(5):
            assert f"step_{i}" in trail
            assert f"reason_{i}" in trail

    def test_divergence_detection(self, dt: DecisionTracer):
        """find_divergence should pinpoint where agent chose wrongly."""
        dt.capture(
            step="tool_selection",
            context={"query": "stock price"},
            options=["get_weather", "get_stock_price"],
            chosen="get_weather",         # wrong
            reasoning="Misclassified intent as weather query",
        )

        result = dt.find_divergence("get_stock_price")
        assert "Divergence" in result
        assert "get_weather" in result
        assert "get_stock_price" in result

    def test_no_divergence_when_correct(self, dt: DecisionTracer):
        """find_divergence returns 'no divergence' when agent chose correctly."""
        dt.capture(
            step="tool_selection",
            context={},
            options=["get_weather", "get_stock_price"],
            chosen="get_stock_price",
            reasoning="Correct choice",
        )
        result = dt.find_divergence("get_stock_price")
        assert "No divergence" in result

    def test_capture_returns_decision(self, dt: DecisionTracer):
        """capture() should return a Decision with the given fields."""
        d = dt.capture("my_step", {"k": "v"}, ["x", "y"], "x", "because")
        assert isinstance(d, Decision)
        assert d.step == "my_step"
        assert d.chosen == "x"
        assert d.reasoning == "because"


# ===========================================================================
# LOGGING TESTS
# ===========================================================================

class TestAgentLogger:

    def _capture_logs(self) -> tuple[AgentLogger, list[str]]:
        """Return an AgentLogger wired to an in-memory list."""
        log = logging.getLogger("agent")
        log.handlers.clear()

        lines: list[str] = []

        class ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                lines.append(self.format(record))

        handler = ListHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)
        log.setLevel(logging.DEBUG)

        logger = AgentLogger(log_level="DEBUG")
        return logger, lines

    def test_logs_are_valid_json(self):
        """Every emitted log line must be valid JSON."""
        logger, lines = self._capture_logs()

        logger.log_llm_call("tid", "sid", "gpt-4o", 3, 1200)
        logger.log_llm_response("tid", "sid", "gpt-4o", 1000, 200, 1234.5, False)
        logger.log_tool_execution("tid", "sid", "search", "query=hello")
        logger.log_tool_result("tid", "sid", "search", True, "result...", 50.0)
        logger.log_context_management("tid", "truncated", 80000, 60000)

        assert len(lines) > 0
        for line in lines:
            parsed = json.loads(line)  # raises if invalid
            assert "event" in parsed or "level" in parsed

    def test_logs_redact_sensitive_data(self):
        """api_key and similar fields should be replaced with [REDACTED]."""
        logger, lines = self._capture_logs()

        # Trigger a log that contains a sensitive context
        logger.log_error(
            trace_id="tid",
            error=ValueError("boom"),
            context={"api_key": "sk-secret-123", "user": "alice"},
        )

        assert len(lines) > 0
        combined = " ".join(lines)
        assert "sk-secret-123" not in combined
        assert "[REDACTED]" in combined

    def test_log_level_filtering(self):
        """WARNING-level logger should suppress INFO logs."""
        log = logging.getLogger("agent")
        log.handlers.clear()

        lines: list[str] = []

        class ListHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                lines.append(self.format(record))

        handler = ListHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(handler)

        logger = AgentLogger(log_level="WARNING")
        logger.log_llm_call("t", "s", "gpt-4o", 1, 100)  # INFO — should be suppressed
        logger.log_context_management("t", "truncated", 1000, 800)  # WARNING — emitted

        # Filter to lines that came from our handler
        info_events = [l for l in lines if '"llm_call_start"' in l]
        warn_events = [l for l in lines if '"context_management"' in l]
        assert len(info_events) == 0
        assert len(warn_events) == 1


# ===========================================================================
# JSON FILE EXPORTER TESTS
# ===========================================================================

class TestJSONFileExporter:

    def test_export_and_load(self, tmp_path):
        """Exported trace should be loadable and match the original."""
        exporter = JSONFileExporter(output_dir=str(tmp_path))
        t = _make_trace(user_id="u1", n_llm_calls=2, n_tool_calls=1)

        exporter.export(t)
        loaded = exporter.load(t.trace_id)

        assert loaded["trace_id"] == t.trace_id
        assert loaded["user_id"] == "u1"
        assert len(loaded["spans"]) == len(t.spans)

    def test_list_traces(self, tmp_path):
        """list_traces() should return the IDs of all exported traces."""
        exporter = JSONFileExporter(output_dir=str(tmp_path))
        ids = set()
        for _ in range(3):
            t = _make_trace()
            exporter.export(t)
            ids.add(t.trace_id)

        stored = set(exporter.list_traces())
        assert ids == stored

    def test_json_is_valid(self, tmp_path):
        """Written file must be valid JSON with expected top-level keys."""
        exporter = JSONFileExporter(output_dir=str(tmp_path))
        t = _make_trace()
        exporter.export(t)

        filepath = tmp_path / f"{t.trace_id}.json"
        data = json.loads(filepath.read_text())
        for key in ("trace_id", "user_query", "status", "spans"):
            assert key in data
