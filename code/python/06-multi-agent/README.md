# 06 · Multi-Agent Patterns

Practical implementations of four multi-agent collaboration patterns, communication primitives, and cross-language ports. No external orchestration frameworks required.

> **Important:** Multi-agent increases complexity significantly. Start with a single agent; only add agents when a single agent cannot handle the task reliably.

---

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```bash
# Run any demo (requires OPENAI_API_KEY)
python delegation_agent.py
python debate_agent.py
python supervisor_agent.py
python swarm_agent.py

# Run tests (no API key needed)
python -m pytest test_multi_agent.py -v
```

---

## Pattern Decision Guide

| Pattern | Use when | Caution |
|---|---|---|
| **Delegation** | Task requires multiple specialist domains | Coordinator can loop; cap delegations |
| **Debate** | Quality matters more than speed | Each round doubles LLM calls |
| **Supervisor-Worker** | Complex workflow with sequential dependencies | Validation adds latency |
| **Swarm** | Creative tasks benefiting from diversity | High token usage at scale |

---

## Files

| File | Pattern | Description |
|---|---|---|
| `delegation_agent.py` | Delegation | Coordinator delegates to Finance, Research, Writing specialists |
| `debate_agent.py` | Debate | Generator + Critic improve iteratively through adversarial review |
| `supervisor_agent.py` | Supervisor-Worker | Supervisor decomposes, assigns, validates, and synthesises |
| `swarm_agent.py` | Swarm | 4 independent agents generate in parallel; merger consolidates |
| `shared_history.py` | Communication | Shared message list — demonstrates context pollution risk |
| `structured_handoff.py` | Communication | Explicit `Handoff` objects — safe agent boundaries |
| `message_bus.py` | Communication | Pub/sub bus with topics and wildcard subscriptions |
| `test_multi_agent.py` | — | 10 tests covering all four patterns (MockLLM, no API calls) |

---

## Cross-Language Ports

- Node.js/TypeScript: [code/nodejs/06-multi-agent/delegation_agent.ts](../../nodejs/06-multi-agent/delegation_agent.ts)
- Go: [code/go/06-multi-agent/delegation_agent.go](../../go/06-multi-agent/delegation_agent.go)

---

## Related Documentation

- [docs/02-the-agent-loop/04-multi-agent-patterns.md](../../../../docs/02-the-agent-loop/04-multi-agent-patterns.md)
