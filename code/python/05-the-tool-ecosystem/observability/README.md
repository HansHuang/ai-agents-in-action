# Agent Observability

> *"If you can't trace it, you can't debug it. If you can't measure it, you can't improve it."*

This folder demonstrates a complete agent observability system — the instrumentation layer that turns the agent's black-box decision process into something you can inspect, debug, and monitor in production.

## What This Demonstrates

Agents fail in ways that are invisible without the right tooling: wrong tool choice, context truncation, runaway token costs, subtle reasoning errors. This folder shows how to see inside every step.

## The Three Pillars

| Pillar | File | Answers |
|---|---|---|
| **Tracing** | `agent_observability.py` | "What happened during this specific request?" |
| **Logging** | `agent_observability.py` | "What was the state at each step?" |
| **Metrics** | `agent_observability.py` | "Is the system healthy right now?" |

## Files

| File | Purpose |
|---|---|
| `agent_observability.py` | Core: `TraceCollector`, `AgentMetrics`, `AgentLogger`, `TokenAccountant`, `DecisionTracer` |
| `exporters.py` | Trace exporters: `ConsoleExporter`, `JSONFileExporter`, `LangFuseExporter`, `OpenTelemetryExporter`, `MultiExporter` |
| `dashboard.py` | Real-time terminal dashboard (plain-text or `rich`) |
| `test_observability.py` | 20 pytest tests covering all components |

## Quick Start

```bash
# Install deps (rich is optional but recommended for the dashboard)
pip install rich

# Run the main demo: 3 requests, one failing, full trace + decision trail + metrics
python agent_observability.py

# Run the exporters demo: Console, JSON file, MultiExporter
python exporters.py

# Run the live dashboard (simulates 80 requests with an error spike at request 60)
python dashboard.py

# Run tests
pytest test_observability.py -v
```

## Architecture

```
User Query
    │
    ▼
ObservableAgent
    │
    ├─── TraceCollector ──► TraceExporter ──► Console / JSON / LangFuse / OTLP
    │         │
    │         └── Trace { Span, Span, Span … }
    │
    ├─── AgentMetrics   ──► Prometheus / Dashboard
    ├─── AgentLogger    ──► Structured JSON (stdout / log aggregator)
    ├─── TokenAccountant ─► Cost per user / session / model
    └─── DecisionTracer ──► Human-readable "why did it do that?"
```

## Supported Exporters

| Exporter | When to Use |
|---|---|
| `ConsoleExporter` | Development — zero deps, pretty tree view |
| `JSONFileExporter` | Debugging — one file per trace, inspect offline |
| `LangFuseExporter` | Production — open-source, self-hostable (`pip install langfuse`) |
| `OpenTelemetryExporter` | Enterprise — Jaeger, Grafana Tempo, Datadog, Honeycomb (`pip install opentelemetry-sdk opentelemetry-exporter-otlp-proto-http`) |
| `MultiExporter` | Fan out to multiple backends simultaneously |

## Related Implementations

- [Node.js / TypeScript](../../../nodejs/05-the-tool-ecosystem/observability/agent_observability.ts)
- [Go](../../../go/05-the-tool-ecosystem/observability/agent_observability.go)

## Reference

→ [Agent Observability](../../../../docs/05-the-tool-ecosystem/03-agent-observability.md)
