# 07-harness — Harness Engineering

Production-grade request lifecycle control for AI agents: **validate, classify, route, handle, guard, and make every call resilient.**

> **Key insight:** Every response is guilty until proven innocent. Validate before the user sees it.

## Files

### Resilience Layer

| File | Description |
|------|-------------|
| `resilience_layer.py` | Retry + Fallback + Circuit Breaker combined into one resilience wrapper |
| `failure_scenarios.py` | Simulator for 10 realistic failure/recovery scenarios |
| `resilience_config.py` | Configuration builder for different system profiles (user-facing, background, SLO-derived) |
| `test_resilience.py` | Pytest suite — 34 tests covering all resilience components |

### Routing and Intent Classification

| File | Description |
|------|-------------|
| `hybrid_router.py` | Two-stage router (deterministic + LLM) with 12+ intents, metrics, and escalation |
| `routing_test_suite.py` | 100+ labelled test cases with accuracy reporting and CI guard |
| `handlers.py` | Specialized handler implementations: chat, RAG, agent loop, escalation |
| `test_routing.py` | Pytest suite — 35 tests covering all routing components |

### Input Guardrails

| File | Description |
|------|-------------|
| `input_guardrail_pipeline.py` | Complete six-layer validation pipeline |
| `injection_test_suite.py` | 60+ prompt injection payloads with precision/recall metrics |
| `pii_benchmark.py` | PII detection accuracy benchmark (1 000 test inputs) |
| `test_input_guardrails.py` | Pytest test suite (33 tests) |

### Output Guardrails

| File | Description |
|------|-------------|
| `output_guardrail_pipeline.py` | Complete six-layer output validation pipeline (schema → PII → safety → leakage → hallucination → facts) |
| `hallucination_test_suite.py` | 52 labelled test cases across 5 hallucination categories with per-category precision/recall |
| `safety_regression_suite.py` | 105 labelled safety cases (8 harmful categories + adversarial bypass attempts) with FPR/FNR metrics |
| `test_output_guardrails.py` | Pytest suite — 20 tests covering all six output guardrail layers |

### Human-in-the-Loop (HITL)

| File | Description |
|------|-------------|
| `human_in_the_loop.py` | Core async HITL approval system: `ApprovalPolicy`, `ApprovalInterface`, `ApprovalExecutor`, `ApprovalMetrics` |
| `approval_dashboard.py` | Rich CLI reviewer dashboard — scripted demo mode and live interactive mode |
| `approval_policy_optimizer.py` | Analyzes historical approval records and generates policy improvement recommendations |
| `test_human_in_the_loop.py` | Pytest suite — 28 tests: policy (8), interface (6), executor (4), metrics (4), integration (6) |

## Architecture: Complete Request Lifecycle

```
User Input
    │
    ▼  Input Guardrails     (rate limit → structural → PII → content policy → injection → sanitise)
    │
    ▼  Router               (deterministic regex → LLM fallback)
    │                        ↓ ~75% handled instantly    ↓ ~25% need LLM
    │                 DeterministicRouter            LLMRouter
    │
    ▼  Handler dispatch      (wrapped in ResilienceLayer for every external call)
    │
    ├──▶ simple_chat         (fast model, 512 tokens, 15 s timeout)
    ├──▶ knowledge_question  (RAG: retrieve docs → grounded answer)
    ├──▶ agent_task          (tool-calling loop, up to 10 iterations)
    ├──▶ support_request     (ticket creation, human escalation)
    └──▶ out_of_scope        (polite refusal)
    │
    ▼  Output Guardrails     (schema → PII → safety → leakage → hallucination → fact-check)
    │
    ▼  Human Approval        (ApprovalPolicy evaluates risk → ApprovalInterface routes to reviewer)
    │                          ↓ approved / approved_with_edits / rejected / timed_out
    │                        ApprovalExecutor runs action or notifies user
    │
    ▼
Final Response to User
```

> **Key insight:** "One agent can't handle everything. Route to specialists."  
> **Hybrid philosophy:** "Deterministic for the obvious, LLM for the ambiguous."

---

## Resilience Triad

| Pattern | Handles | Analogy |
|---------|---------|---------|
| **Retry** | Transient failures (packet loss, 429, 503) | "Let me try that one more time" |
| **Fallback** | Persistent failures (provider outage, quota) | "Plan B" |
| **Circuit Breaker** | Cascading failure (stop calling broken services) | "Give it a break and try later" |

### Circuit Breaker States

```
CLOSED ──(N failures in window)──▶ OPEN ──(timeout)──▶ HALF_OPEN
  ▲                                                          │
  └──────────────── probe succeeds ◀────────────────────────┘
                    probe fails → back to OPEN
```

### Fallback Hierarchy

```
Full capability (provider A)  ← Primary
    │ fails
Full capability (provider B)  ← Cross-provider fallback
    │ fails
Reduced capability (smaller model)
    │ fails
Static response (pre-computed, zero cost)
    │ fails
SystemUnavailableError → show error page, alert on-call
```

### Exponential Backoff Formula

$$delay = base \times multiplier^{attempt}$$

With ±10 % jitter and a 60 s cap:

| Attempt | Without jitter | With jitter (±10 %) |
|---------|---------------|---------------------|
| 0 | 1.0 s | ~0.9 – 1.1 s |
| 1 | 2.0 s | ~1.8 – 2.2 s |
| 2 | 4.0 s | ~3.6 – 4.4 s |
| 3 | 8.0 s | ~7.2 – 8.8 s |

Jitter prevents the [thundering herd](https://en.wikipedia.org/wiki/Thundering_herd_problem): when 1 000 clients all retry at the same moment they overwhelm the recovering service.

---

## Quick Start

### Resilience layer

```python
import asyncio
from resilience_layer import (
    CircuitBreaker, FallbackExecutor, FallbackLevel,
    ResilienceLayer, RetryConfig,
)

async def my_llm_call():
    # Replace with your actual provider call
    return {"answer": "42"}

async def my_fallback_call():
    return {"answer": "cached answer"}

layer = ResilienceLayer(
    name="llm_call",
    circuit_breaker=CircuitBreaker("openai", failure_threshold=5),
    retry_config=RetryConfig(max_retries=3, base_delay_seconds=1.0),
    fallback_executor=FallbackExecutor([
        FallbackLevel("fallback", my_fallback_call, timeout_seconds=10, capability="reduced"),
    ]),
)

async def main():
    result = await layer.execute(my_llm_call)
    print(result.path)       # "primary" or "fallback_level_0"
    print(result.result)

asyncio.run(main())
```

### Run failure scenarios

```bash
python failure_scenarios.py   # 10 annotated failure/recovery scenarios
```

### Get a profile-based config

```python
from resilience_config import ResilienceConfigBuilder

builder = ResilienceConfigBuilder()

cfg = builder.for_user_facing_api()   # fast, fail-fast
cfg = builder.for_background_job()   # patient, retry-heavy
cfg = builder.for_critical_path()    # balanced, multi-fallback
cfg = builder.for_cost_sensitive()   # minimal retries, cheap fallback

# Or derive from SLO targets
cfg = builder.from_slo(target_availability=0.999, target_latency_p99=5.0)
print(cfg.max_retries, cfg.circuit_failure_threshold)
```

### Router

```python
import asyncio
from hybrid_router import HybridRouter

router = HybridRouter()

async def main():
    result = await router.route("Where is my order #12345?")
    print(result.intent)      # agent_task
    print(result.method)      # llm
    print(result.confidence)  # 0.94

asyncio.run(main())
```

### Register handlers and process requests

```python
from handlers import build_handler_registry
from hybrid_router import EscalatingRouter, HybridRouter

router = HybridRouter()
registry = build_handler_registry()
escalating = EscalatingRouter(router, registry)

async def main():
    response = await escalating.handle("What's your return policy?")
    print(response.content)
    print(response.handler_used)   # rag
    print(response.metadata)

asyncio.run(main())
```

### Input guardrails

```python
from input_guardrail_pipeline import InputGuardrailPipeline

pipeline = InputGuardrailPipeline()

result = pipeline.process(
    user_input="What is the capital of France?",
    user_id="user_42",
)

if result.passed:
    print(result.cleaned_input)
else:
    print(result.rejection_layer, result.rejection_reason)
```

### Output guardrails

```python
import asyncio
from output_guardrail_pipeline import OutputGuardrailConfig, OutputGuardrailPipeline

schema = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "confidence": {"type": "number"}},
    "required": ["answer", "confidence"],
}

pipeline = OutputGuardrailPipeline(
    config=OutputGuardrailConfig(expected_schema=schema, expected_type="json")
)
pipeline.set_system_prompt("You are a helpful assistant.")

async def main():
    model_output = '{"answer": "Paris is the capital of France.", "confidence": 0.99}'
    context = {
        "retrieved_documents": [{"text": "Paris is the capital of France."}],
    }
    result = await pipeline.validate(model_output, context)

    if result.passed:
        print(result.cleaned_output)        # possibly redacted
    else:
        print(result.rejection_layer)       # e.g. "safety", "hallucination"
        print(result.rejection_reason)

asyncio.run(main())
```

### Human-in-the-loop

```python
import asyncio
from human_in_the_loop import (
    ApprovalInterface, ApprovalPolicy, ApprovalExecutor,
    HumanReviewerInterface, Reviewer,
)

policy = ApprovalPolicy.with_defaults()
iface  = ApprovalInterface(channels=["dashboard"])
iface.register_reviewer(Reviewer("r-1", "Alice", is_senior=True))
reviewer  = HumanReviewerInterface("r-1")

async def main():
    from human_in_the_loop import ApprovalRequest
    import time, uuid

    req = ApprovalRequest(
        request_id=str(uuid.uuid4()),
        agent_id="my-agent", session_id="s1",
        proposed_action="send_email",
        proposed_params={"to": "user@example.com", "subject": "Your order"},
        reasoning="User asked to be notified",
        conversation_summary="Order #42 was shipped.",
        evidence=[], risk_level="medium", estimated_cost=0.0,
        affected_systems=["email_service"], created_at=time.time(),
    )

    decision = policy.requires_approval(req.proposed_action, req.proposed_params, {})
    if decision.requires_approval:
        # In production this waits for a real human; here we auto-approve after 0.1 s
        asyncio.get_event_loop().call_later(
            0.1, lambda: iface.submit_response(reviewer.approve(req.request_id))
        )
        resp = await iface.request_approval(req, timeout_seconds=30)
        print(resp.decision)   # "approved"

asyncio.run(main())
```

### Dashboard demo

```bash
python approval_dashboard.py            # scripted 6-step demo
python approval_dashboard.py --interactive  # keyboard-driven interactive mode
```

### Policy optimizer

```bash
python approval_policy_optimizer.py     # analyzes 500 synthetic records, prints report
```

## Running the demos

```bash
python hybrid_router.py        # routing demo with 20 test requests
python routing_test_suite.py   # 100+ test cases with accuracy report
python handlers.py             # compare handlers side-by-side
python input_guardrail_pipeline.py   # 10-case input demo
python injection_test_suite.py
python pii_benchmark.py
python output_guardrail_pipeline.py  # 10-case output demo
python hallucination_test_suite.py   # 52-case hallucination report
python safety_regression_suite.py    # 105-case safety regression report
python failure_scenarios.py    # 10 failure/recovery scenarios
python resilience_config.py    # configuration builder with comparison table
python human_in_the_loop.py    # 8-scenario HITL demo
python approval_dashboard.py   # Rich reviewer dashboard
python approval_policy_optimizer.py  # policy optimization report

pytest test_routing.py -v            # 35 routing tests
pytest test_input_guardrails.py -v   # 33 input guardrail tests
pytest test_resilience.py -v         # 34 resilience tests
pytest test_output_guardrails.py -v  # 20 output guardrail tests
pytest test_human_in_the_loop.py -v  # 28 HITL tests
```

## Routing Accuracy Targets

| Intent | Target accuracy |
|---|---|
| simple_chat / greeting | ≥ 95% |
| knowledge_question | ≥ 87% |
| agent_task | ≥ 90% |
| support_request | ≥ 85% |
| human_escalation | ≥ 95% |
| **Overall** | **≥ 90%** |

The `routing_test_suite.py` exits with a non-zero code if overall accuracy drops below 85%, making it safe to run in CI.

## Escalation Paths

When a handler fails or returns a low-quality response, the `EscalatingRouter` automatically tries the next handler:

```
simple_chat → knowledge_question → agent_task → human_escalation
support_request                  → human_escalation
```

Escalation triggers: no RAG documents found, agent reaches max iterations, or response contains uncertainty phrases.

## Injection Detection Benchmark

| Category | Expected F1 |
|---|---|
| Direct Override | ~0.90 |
| Delimiter Abuse | ~1.00 |
| Roleplay Attacks | ~0.80 |
| Token Smuggling | ~0.50–0.65 |

---

## The Assembled ProductionHarness (Chapter 7 Capstone)

The files below wire all five layers into a single production-ready entry point.

### Chapter 7 files

| File | Description |
|------|-------------|
| `production_harness.py` | `ProductionHarness` class — assembles all five layers; 10-request demo |
| `harness_config.py` | Standalone config system: env/YAML/JSON loading, presets, diff, export |
| `harness_test_framework.py` | `HarnessTestFramework` — unit, integration, chaos, regression, perf, security |
| `harness_runbook.py` | `HarnessRunbook` — 6 operational scenarios with auto-remediation actions |
| `test_production_harness.py` | pytest integration suite — 22 test functions, all deps mocked |

### Quick start (3 lines)

```python
from production_harness import ProductionHarness, HarnessConfig

harness = ProductionHarness(HarnessConfig.production())
response = await harness.process("What's your return policy?", user_id="alice")
```

> **Key insight:** "The harness is not a framework — it is a set of promises: every request will be validated, routed, handled, checked, and approved or rejected. When any promise breaks the harness surfaces the failure clearly instead of hiding it."

### The Harness Manifesto

1. **Validate early, validate cheaply.** Reject bad input at Layer 1 before spending tokens.
2. **Route to specialists.** One handler cannot serve every intent well.
3. **Every external call is a failure waiting to happen.** Wrap it in retry + fallback + circuit breaker.
4. **Every response is guilty until proven innocent.** Run output guardrails before the user sees anything.
5. **Humans are the final safety net.** Route high-risk actions to a reviewer, not the void.
6. **Measure everything.** A harness without metrics is a black box you cannot debug.
7. **Runbooks are code.** Automate your incident response so humans focus on judgment, not procedure.

### Running chapter 7

```bash
# Assembled harness demo (10 requests, all layers active)
python production_harness.py

# Full test framework (unit + integration + chaos + regression + perf + security)
python harness_test_framework.py

# Operational runbook (diagnose + 3 scenario simulations)
python harness_runbook.py

# pytest integration tests (all external deps mocked)
pytest test_production_harness.py -v
```

### Config loading hierarchy

```
env vars → yaml/json file → code defaults → preset overrides
```

```python
from harness_config import HarnessConfig

cfg = HarnessConfig.from_env()            # production
cfg = HarnessConfig.from_yaml("cfg.yaml") # file-based
cfg = HarnessConfig.development()         # preset: permissive
cfg = HarnessConfig.production()          # preset: strict
cfg = HarnessConfig.cost_optimized()      # preset: cheap models
cfg = HarnessConfig.high_security()       # preset: maximum guardrails

warnings = cfg.validate()  # ["approval_default_timeout > critical_timeout", ...]
diff = cfg.diff(other_cfg) # human-readable diff of two configs
```

---

## Cross-References

- Node.js: [code/nodejs/07-harness/hybrid_router.ts](../../nodejs/07-harness/hybrid_router.ts) · [input_guardrail_pipeline.ts](../../nodejs/07-harness/input_guardrail_pipeline.ts) · [output_guardrail_pipeline.ts](../../nodejs/07-harness/output_guardrail_pipeline.ts) · [resilience_layer.ts](../../nodejs/07-harness/resilience_layer.ts) · [human_in_the_loop.ts](../../nodejs/07-harness/human_in_the_loop.ts) · [production_harness.ts](../../nodejs/07-harness/production_harness.ts)
- Go: [code/go/07-harness/hybrid_router.go](../../go/07-harness/hybrid_router.go) · [input_guardrail_pipeline.go](../../go/07-harness/input_guardrail_pipeline.go) · [output_guardrail_pipeline.go](../../go/07-harness/output_guardrail_pipeline.go) · [resilience_layer.go](../../go/07-harness/resilience_layer.go) · [human_in_the_loop.go](../../go/07-harness/human_in_the_loop.go) · [production_harness.go](../../go/07-harness/production_harness.go)
- Docs: [docs/07-harness-engineering/](../../../docs/07-harness-engineering/) · [07-building-a-reliable-harness.md](../../../docs/07-harness-engineering/07-building-a-reliable-harness.md)
