"""Trace exporters for the agent observability system.

Each exporter implements the TraceExporter protocol: a single ``export(trace)``
method that persists or forwards the trace to a backend.

Included exporters:
  ConsoleExporter       — formatted tree view to stdout (default, zero deps)
  JSONFileExporter      — one JSON file per trace, for offline analysis
  LangFuseExporter      — push to LangFuse (open-source tracing, optional dep)
  OpenTelemetryExporter — OTLP spans via OpenTelemetry SDK (optional dep)

See: docs/05-the-tool-ecosystem/03-agent-observability.md
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_observability import Span, Trace


# ===========================================================================
# ConsoleExporter — zero dependencies
# ===========================================================================

class ConsoleExporter:
    """Pretty-print a trace as an ASCII decision tree.

    Produces output like::

        ══════════════════════════════════════════
        ✓ Trace a1b2c3d4  query='Compare AAPL vs MSFT'
          duration=4823ms  tokens=8330  cost=$0.0021
          llm_calls=3  tool_calls=2  status=success

          · [planning]   generate_plan       12ms  tokens=0
          · [execution]  execute_plan      4800ms  tokens=0
              · [tool_call]  get_stock_price(AAPL)  302ms  tokens=0
              · [llm_call]   synthesise     1501ms  tokens=2890
              · [tool_call]  get_stock_price(MSFT)  198ms  tokens=0
              · [llm_call]   generate_answer 1797ms  tokens=3100
        ══════════════════════════════════════════
    """

    def export(self, trace: "Trace") -> None:
        icon = "✓" if not trace.has_error else "✗"
        border = "═" * 56

        print(f"\n{border}")
        print(f"{icon} Trace {trace.trace_id[:8]}  "
              f"query='{trace.user_query[:55]}'")
        print(f"  duration={trace.duration_ms:.0f}ms  "
              f"tokens={trace.total_tokens}  "
              f"cost=${trace.total_cost:.4f}")
        print(f"  llm_calls={trace.llm_call_count}  "
              f"tool_calls={trace.tool_call_count}  "
              f"status={trace.status}")
        print()

        # Build parent → children map for tree rendering
        children: dict[str | None, list["Span"]] = {}
        for span in trace.spans:
            children.setdefault(span.parent_span_id, []).append(span)

        def _render(span_id: str | None, depth: int) -> None:
            for span in children.get(span_id, []):
                indent = "  " + "    " * depth
                s_icon = "✗" if span.status == "error" else "·"
                err = (f"  ERROR: {span.error_message}"
                       if span.error_message else "")
                print(f"{indent}{s_icon} [{span.type:<10}] {span.name:<22} "
                      f"{span.duration_ms:>7.0f}ms  "
                      f"tokens={span.tokens_used}{err}")
                _render(span.span_id, depth + 1)

        _render(None, 0)
        print(f"{border}\n")


# ===========================================================================
# JSONFileExporter
# ===========================================================================

class JSONFileExporter:
    """Export each trace as a pretty-printed JSON file.

    Files are named ``<trace_id>.json`` and written to ``output_dir``.
    Useful for post-mortem debugging: you can open any file and inspect the
    complete trace including every span, token count, and error.
    """

    def __init__(self, output_dir: str = "./traces/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, trace: "Trace") -> None:
        filepath = self.output_dir / f"{trace.trace_id}.json"
        filepath.write_text(
            json.dumps(trace.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )

    def load(self, trace_id: str) -> dict:
        """Read a previously exported trace back from disk."""
        filepath = self.output_dir / f"{trace_id}.json"
        return json.loads(filepath.read_text(encoding="utf-8"))

    def list_traces(self) -> list[str]:
        """Return all trace IDs stored in the output directory."""
        return [p.stem for p in sorted(self.output_dir.glob("*.json"))]


# ===========================================================================
# MultiExporter — fan-out to multiple backends
# ===========================================================================

class MultiExporter:
    """Fan traces out to multiple exporters simultaneously.

    Example — console during development, JSON file for debugging,
    and LangFuse in production::

        exporter = MultiExporter([
            ConsoleExporter(),
            JSONFileExporter("./traces/"),
            LangFuseExporter(public_key=..., secret_key=...),
        ])
    """

    def __init__(self, exporters: list):
        self._exporters = exporters

    def export(self, trace: "Trace") -> None:
        for exporter in self._exporters:
            try:
                exporter.export(trace)
            except Exception as exc:  # noqa: BLE001
                # Never let one failed exporter break the agent
                print(f"[WARN] Exporter {type(exporter).__name__} failed: {exc}")


# ===========================================================================
# LangFuseExporter (requires: pip install langfuse)
# ===========================================================================

class LangFuseExporter:
    """Push traces to LangFuse — open-source, self-hostable tracing.

    Requires the ``langfuse`` Python package::

        pip install langfuse

    Sign up at https://langfuse.com or self-host the Docker image.
    """

    def __init__(self, public_key: str, secret_key: str,
                 host: str = "https://cloud.langfuse.com"):
        try:
            import langfuse  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "LangFuseExporter requires 'langfuse'. "
                "Install it with: pip install langfuse"
            ) from exc

        self._lf = langfuse.Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )

    def export(self, trace: "Trace") -> None:
        """Convert internal Trace → LangFuse trace + observations."""
        lf_trace = self._lf.trace(
            id=trace.trace_id,
            name="agent_request",
            input={"query": trace.user_query},
            user_id=trace.user_id,
            session_id=trace.session_id,
            metadata=trace.metadata,
        )

        span_map: dict[str, object] = {}

        for span in trace.spans:
            parent = (span_map.get(span.parent_span_id)
                      if span.parent_span_id else lf_trace)

            if span.type == "llm_call":
                obs = lf_trace.generation(
                    name=span.name,
                    model=span.model,
                    input=span.input_data,
                    output=span.output_data,
                    usage={
                        "input": span.input_tokens,
                        "output": span.output_tokens,
                        "total": span.tokens_used,
                    },
                    metadata={"cost": span.cost, "latency_ms": span.duration_ms},
                )
            else:
                obs = lf_trace.span(
                    name=span.name,
                    input=span.input_data,
                    output=span.output_data,
                    metadata={"type": span.type, "latency_ms": span.duration_ms},
                )
            span_map[span.span_id] = obs

        self._lf.flush()


# ===========================================================================
# OpenTelemetryExporter (requires: pip install opentelemetry-sdk
#                                           opentelemetry-exporter-otlp)
# ===========================================================================

class OpenTelemetryExporter:
    """Export traces using the OpenTelemetry standard.

    This makes agent traces compatible with any OTLP-capable backend:
    Jaeger, Zipkin, Grafana Tempo, Honeycomb, Datadog, …

    Requires::

        pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http

    The OTLP endpoint defaults to localhost:4318 (local collector).
    Set OTEL_EXPORTER_OTLP_ENDPOINT in the environment to override.
    """

    def __init__(self, service_name: str = "ai-agent",
                 otlp_endpoint: str = "http://localhost:4318"):
        try:
            from opentelemetry import trace as otel_trace  # noqa: PLC0415
            from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
            from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
            from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
                OTLPSpanExporter,
            )
        except ImportError as exc:
            raise ImportError(
                "OpenTelemetryExporter requires the OTel SDK. "
                "Install with: pip install opentelemetry-sdk "
                "opentelemetry-exporter-otlp-proto-http"
            ) from exc

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=otlp_endpoint))
        )
        otel_trace.set_tracer_provider(provider)
        self._tracer = otel_trace.get_tracer(service_name)
        self._otel_trace = otel_trace

    def export(self, trace: "Trace") -> None:
        """Convert internal Trace → OpenTelemetry spans."""
        # Map internal span IDs to OTel context managers so children can nest
        otel_spans: dict[str, object] = {}

        with self._tracer.start_as_current_span(
            "agent_request",
            attributes={
                "agent.trace_id": trace.trace_id,
                "agent.user_query": trace.user_query[:256],
                "agent.user_id": trace.user_id or "",
                "agent.session_id": trace.session_id or "",
                "agent.total_tokens": trace.total_tokens,
                "agent.total_cost": trace.total_cost,
                "agent.status": trace.status,
            },
        ) as root_span:
            for span in trace.spans:
                # Context: use parent OTel span if available
                parent_ctx = (
                    otel_spans.get(span.parent_span_id)
                    if span.parent_span_id
                    else None
                )
                with self._tracer.start_as_current_span(
                    span.name,
                    context=parent_ctx,
                    attributes={
                        "span.type": span.type,
                        "span.tokens_used": span.tokens_used,
                        "span.cost": span.cost,
                        "span.model": span.model or "",
                        "span.status": span.status,
                        "span.latency_ms": span.duration_ms,
                    },
                ) as otel_span:
                    if span.status == "error" and span.error_message:
                        otel_span.set_attribute(
                            "exception.message", span.error_message
                        )
                    otel_spans[span.span_id] = (
                        self._otel_trace.get_current_span().get_span_context()
                    )


# ===========================================================================
# Demo
# ===========================================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from agent_observability import (
        TraceCollector, AgentMetrics, AgentLogger,
        TokenAccountant, DecisionTracer, ObservableAgent,
    )

    print("=== ConsoleExporter demo ===")
    collector = TraceCollector(exporter=ConsoleExporter())
    agent = ObservableAgent(
        collector=collector,
        metrics=AgentMetrics(),
        logger=AgentLogger(log_level="WARNING"),
        accountant=TokenAccountant(),
        decision_tracer=DecisionTracer(),
    )
    agent.run("What is the capital of France?", user_id="u1", session_id="s1")

    print("\n=== JSONFileExporter demo ===")
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        json_exp = JSONFileExporter(output_dir=tmpdir)
        collector2 = TraceCollector(exporter=json_exp)
        agent2 = ObservableAgent(
            collector=collector2,
            metrics=AgentMetrics(),
            logger=AgentLogger(log_level="WARNING"),
            accountant=TokenAccountant(),
            decision_tracer=DecisionTracer(),
        )
        result = agent2.run("Explain quantum entanglement", "u2", "s2")
        trace_id = result["trace_id"]
        loaded = json_exp.load(trace_id)
        print(f"Trace exported to JSON: trace_id={loaded['trace_id']}")
        print(f"Spans: {len(loaded['spans'])}")
        all_ids = json_exp.list_traces()
        print(f"All stored traces: {all_ids}")

    print("\n=== MultiExporter demo (Console + JSON) ===")
    with tempfile.TemporaryDirectory() as tmpdir2:
        multi = MultiExporter([ConsoleExporter(), JSONFileExporter(tmpdir2)])
        col3 = TraceCollector(exporter=multi)
        agent3 = ObservableAgent(
            collector=col3,
            metrics=AgentMetrics(),
            logger=AgentLogger(log_level="WARNING"),
            accountant=TokenAccountant(),
            decision_tracer=DecisionTracer(),
        )
        agent3.run("Multi-exporter test", "u3", "s3")
        stored = JSONFileExporter(tmpdir2).list_traces()
        print(f"JSON files written: {len(stored)}")
