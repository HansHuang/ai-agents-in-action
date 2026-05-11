# 08 — Evaluating Agents

> **Key insight:** "It looks right" is not evaluation. Measure everything.

This directory demonstrates a three-level framework for measuring whether an agent actually works — not just whether it runs without crashing.

---

## Three Levels of Evaluation

### 1. Retrieval Quality
Does the retriever surface the right documents?

| Metric | Target | Formula |
|---|---|---|
| Hit Rate | > 90% | ≥1 relevant doc in top-k |
| Precision@K | > 70% | relevant / retrieved |
| Recall@K | > 80% | relevant retrieved / total relevant |
| MRR | > 0.60 | mean of 1/rank(first relevant) |
| NDCG@K | > 70% | graded relevance: relevant=2, partial=1 |

### 2. Generation Quality
Is the generated answer accurate and useful?

Two complementary approaches run on every response:

- **Rule-based checks** — fast, deterministic: required phrases, forbidden phrases, length bounds, source citations. Zero API cost.
- **LLM-as-judge** — catches what rules miss: faithfulness (unsupported claims), relevance (on-topic), completeness (all parts answered). Use a different, capable model as judge. Budget ~100 tokens per call.

**LLM-as-judge caveats:** Judges inherit model biases and can themselves hallucinate. Run them on a sampled subset (10–20%) of queries for cost control. Always pair with rule-based checks.

### 3. End-to-End Task Success
Does the agent actually solve the user's problem?

Tracks task success rate (target > 85%), turns to resolution, and tool usage across full multi-turn conversations.

---

## Quick Start

```bash
cd code/python/08-evaluation
python agent_evaluator.py        # full demo with regression simulation
python test_set_builder.py       # CSV/JSON/log import demo
python evaluation_dashboard.py   # rich terminal dashboard

pytest test_agent_evaluator.py -v    # 43 tests, no API keys required
```

---

## Continuous Evaluation

```python
pipeline = ContinuousEvaluationPipeline(...)
await pipeline.set_baseline()          # run on current main branch

# Run in CI on every PR:
check = await pipeline.check_regression()
if check.has_regressions:
    raise SystemExit(f"Regressions: {check.regressions}")
```

The regression threshold is **5%** — a drop larger than this on any tracked metric fails the build.

---

## See Also

- Concept doc: [docs/08-evaluation-and-guardrails/01-evaluating-agents.md](../../../docs/08-evaluation-and-guardrails/01-evaluating-agents.md)
- Node.js port: [code/nodejs/08-evaluation/agent_evaluator.ts](../../nodejs/08-evaluation/agent_evaluator.ts)
- Go port: [code/go/08-evaluation/agent_evaluator.go](../../go/08-evaluation/agent_evaluator.go)
