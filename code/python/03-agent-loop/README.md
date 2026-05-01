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
| `test_plan_schema.py` | 23 tests for plan models and quality scorer |
| `test_planning_strategies.py` | 10 tests for Plan-and-Execute and Reflection (MockLLM) |
| `test_tool_builder.py` | 27 unit tests for `ToolDef`/`Param` |
| `test_tool_registry.py` | 10 unit tests for `ToolRegistry` |
| `test_tool_registry_edge_cases.py` | 10 edge-case tests (concurrency, None, exceptions) |
| `test_agent.py` | 7 agent-loop unit tests |
| `test_agent_integration.py` | 3 lifecycle tests with `MockLLM` |

## Quick Start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

python main.py                  # ReAct demo
python plan_execute_agent.py    # Plan-and-Execute demo
python reflection_agent.py      # Reflection demo
python strategy_comparison.py   # Side-by-side comparison table
```

## Run Tests

```bash
pytest -v                              # all 91 tests
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

## Related Docs

- [01-anatomy-of-an-agent.md](../../../docs/02-the-agent-loop/01-anatomy-of-an-agent.md)
- [02-tool-design-patterns.md](../../../docs/02-the-agent-loop/02-tool-design-patterns.md)
- [03-planning-strategies.md](../../../docs/02-the-agent-loop/03-planning-strategies.md)

## Same Implementation in Other Languages

- [Node.js / TypeScript](../../nodejs/03-agent-loop/) — `plan_execute_agent.ts` with Zod
- [Go](../../go/03-agent-loop/) — `plan_execute_agent.go`
