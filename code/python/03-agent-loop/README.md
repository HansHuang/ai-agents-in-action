# 03 · Agent Loop

A working ReAct agent with a complete tool pipeline, plus three planning
strategies — Plan-and-Execute, Reflection, and a side-by-side comparison script.

```
ReAct             Plan-and-Execute           Reflection
──────────        ─────────────────          ───────────────
Reason → Act      Plan ──► Execute           Generate
→ Observe         (parallel)  ──► Synth      → Critique
→ Answer          structured JSON plan       → Revise (repeat)
```

## Files

| File | Purpose |
|------|---------|
| `agent.py` | ReAct loop — `run_agent()`, `MAX_ITERATIONS`, `SYSTEM_PROMPT` |
| `tools.py` | Weather and stock tool implementations + schemas |
| `tool_builder.py` | `ToolDef`/`Param` — programmatic schema building + validation |
| `tool_registry.py` | `ToolRegistry` — registration, dispatch, structured errors |
| `tool_dispatcher.py` | Standalone dispatcher (used directly by the loop) |
| `tool_validator.py` | Static best-practice checker for tool definitions |
| `plan_schema.py` | Pydantic models: `PlanStep`, `AgentPlan`, `StepResult`, quality scorer |
| `plan_execute_agent.py` | Plan-and-Execute agent with parallel step execution |
| `reflection_agent.py` | Generate → Critique → Revise loop with `CritiqueResult` |
| `strategy_comparison.py` | Run all three strategies on the same task; prints metrics table |
| `llm_provider.py` | Provider abstraction — OpenAI, Anthropic, Google, Ollama, Together, Fallback |
| `model_router.py` | Task-based router — scores providers by cost, latency, capability |
| `provider_benchmark.py` | Latency, cost, and capability benchmarking with ASCII report |
| `test_plan_schema.py` | 23 tests for plan models and quality scorer |
| `test_planning_strategies.py` | 10 tests for Plan-and-Execute and Reflection (MockLLM) |
| `test_tool_builder.py` | 27 unit tests for `ToolDef`/`Param` |
| `test_tool_registry.py` | 10 unit tests for `ToolRegistry` |
| `test_tool_registry_edge_cases.py` | 10 edge-case tests (concurrency, None, exceptions) |
| `test_agent.py` | 7 agent-loop unit tests |
| `test_agent_integration.py` | 3 lifecycle tests with `MockLLM` |
| `test_llm_provider.py` | 13 provider, fallback, router, and factory tests |

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
# Optional, to unlock additional providers:
export ANTHROPIC_API_KEY=...
export GOOGLE_API_KEY=...
```

## Multi-Provider Support

Your agent shouldn't know which provider it's talking to.

`llm_provider.py` gives every provider the same interface. Switch providers with a single config change:

```python
from llm_provider import LLMFactory

# Start with OpenAI
provider = LLMFactory.create("openai", api_key=..., model="gpt-4o")

# One line to switch to Anthropic — agent code stays the same
provider = LLMFactory.create("anthropic", api_key=..., model="claude-3-5-sonnet-20241022")

# Or use a local model with Ollama (no API key needed)
provider = LLMFactory.create("ollama", model="llama3.1:8b")
```

Add automatic fallback via config:

```python
provider = LLMFactory.create_from_config({
    "provider": "openai", "api_key": "...", "model": "gpt-4o",
    "fallback": {"provider": "anthropic", "api_key": "...", "model": "claude-3-haiku-20240307"},
})
```

Route tasks to the optimal provider:

```python
from model_router import ModelRouter, RouterConfig, RoutingTask

router = ModelRouter(RouterConfig(priority_order=["cheap", "fast", "smart"]))
router.register_provider("gpt-4o-mini", mini_provider, capabilities=["chat"],
                         cost_per_1k_input=0.00015, cost_per_1k_output=0.0006,
                         typical_latency_ms=800)
# …register more providers…

task = RoutingTask(messages=messages, task_type="simple_chat",
                   estimated_input_tokens=100, estimated_output_tokens=200,
                   priority="cheap")
name, provider = router.route(task)
```

Benchmark providers before committing:

```python
from provider_benchmark import ProviderBenchmark
benchmark = ProviderBenchmark({"GPT-4o": gpt4o, "Claude-Haiku": haiku})
results = benchmark.run_all(messages, iterations=5)
print(benchmark.generate_report(results))
```

**Architecture:**

```
LLMFactory / RouterConfig
       │
       ▼
  LLMProvider (abstract)
  ├─ OpenAIProvider    ← gpt-4o, gpt-4o-mini, gpt-3.5-turbo
  ├─ AnthropicProvider ← claude-3-5-sonnet, claude-3-haiku
  ├─ GoogleProvider    ← gemini-1.5-pro, gemini-1.5-flash
  ├─ OllamaProvider    ← local models via OpenAI-compatible API
  ├─ TogetherProvider  ← hosted open-source models
  └─ FallbackProvider  ← wraps any two providers

  ModelRouter
  └─ scores candidates by cost · latency · capability
       │
       ▼
  Agent / Tools  ← never depends on a specific SDK
```

## Quick Start

```bash
python main.py                  # ReAct demo (uses $OPENAI_API_KEY)
python plan_execute_agent.py    # Plan-and-Execute demo
python reflection_agent.py      # Reflection demo
python strategy_comparison.py   # Side-by-side comparison table
python llm_provider.py          # Provider comparison demo
python provider_benchmark.py    # Cross-provider benchmark
```

## Run Tests

```bash
pytest -v                              # all tests
pytest test_llm_provider.py -v         # 13 provider tests (no API key needed)
pytest test_planning_strategies.py -v  # 10 strategy tests (no API key needed)
pytest test_plan_schema.py -v          # 23 schema tests
```

## Which Strategy Should I Use?

| Situation | Recommended strategy |
|-----------|---------------------|
| Simple lookup or short Q&A | **ReAct** — low latency, lowest cost |
| Multi-source research with parallel fetching | **Plan-and-Execute** — structured, parallelisable |
| High-stakes writing, analysis, or code generation | **Reflection** — best quality via self-critique |
| Complex research *and* high quality required | Wrap Plan-and-Execute inside Reflection |

## Architecture Notes

- **`plan_schema.py`** enforces sequential numbering, backward-only dependencies,
  cycle detection (DFS), and a quality scorer that deducts for vague steps.
- **`plan_execute_agent.py`** executes independent steps in parallel via
  `ThreadPoolExecutor`; dependency resolution uses a topological round loop.
- **`reflection_agent.py`** uses `response_format={"type":"json_object"}` for
  structured critique so scores and feedback are always machine-readable.
- **`strategy_comparison.py`** monkey-patches `OpenAI()` to count calls and
  tokens without modifying agent internals.
- **`llm_provider.py`** uses lazy SDK imports (`from openai import OpenAI` inside
  `__init__`) so the module loads without every package installed.
- **`model_router.py`** evaluates routing rules with a restricted `eval()` scope
  (no builtins) and scores candidates across cost, latency, and quality axes.

## Related Docs

- [01-anatomy-of-an-agent.md](../../../docs/02-the-agent-loop/01-anatomy-of-an-agent.md)
- [02-tool-design-patterns.md](../../../docs/02-the-agent-loop/02-tool-design-patterns.md)
- [03-planning-strategies.md](../../../docs/02-the-agent-loop/03-planning-strategies.md)
- [01-model-providers.md](../../../docs/05-the-tool-ecosystem/01-model-providers.md)

## Same Implementation in Other Languages

- [Node.js / TypeScript](../../nodejs/03-agent-loop/) — `llm_provider.ts` + `plan_execute_agent.ts`
- [Go](../../go/03-agent-loop/) — `llm_provider.go` + `plan_execute_agent.go`
