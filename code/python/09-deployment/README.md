# 09 — Deploying AI Agents to Production

This folder demonstrates how to deploy an AI agent safely: gradual rollout, canary evaluation, streaming infrastructure, cost control, multi-region routing, rollback, and load testing.

Reference: [docs/09-from-dev-to-production/01-deployment-strategies.md](../../../docs/09-from-dev-to-production/01-deployment-strategies.md)

## Files

| File | Purpose |
|------|---------|
| `deployment_manager.py` | Gradual rollout, canary deployment, cost control, multi-region routing, rollback, health checking |
| `agent_service.py` | FastAPI service: REST chat, SSE streaming, `/health`, `/metrics` endpoints |
| `load_tester.py` | Async load testing: sustained, burst, streaming, mixed workload, concurrent users |
| `test_deployment.py` | 37 pytest tests for all deployment components |
| `Dockerfile` | Multi-stage build, non-root user, health check |
| `docker-compose.yml` | Local dev stack: agent + Redis + Qdrant |
| `k8s/deployment.yaml` | K8s Deployment (3 replicas, readiness/liveness probes, anti-affinity) |
| `k8s/service.yaml` | ClusterIP service with ClientIP session affinity for streaming |
| `k8s/hpa.yaml` | HPA: min 3, max 20 replicas; CPU, memory, and custom RPS metrics |
| `k8s/ingress.yaml` | NGINX ingress: proxy-buffering off, 5-minute read timeout, rate limiting |
| `k8s/network-policy.yaml` | Default-deny network policy (ingress controller + LLM/Redis/Qdrant egress only) |
| `k8s/pdb.yaml` | PodDisruptionBudget: always keep ≥ 2 pods running |

## Architecture options

- **Serverless (Lambda)**: zero ops, auto-scaling; cold starts and 15-minute timeout limit
- **Containerized (K8s)**: full control, stateful sessions, no timeouts; requires ops
- **Hybrid**: route simple queries to Lambda and complex agent tasks to the container service

## The rollout playbook

```
Stage 0  Internal team only       Day 0 — run eval suite & red team
Stage 1  1%  canary               Day 1–2 — monitor error rate, latency, cost
Stage 2  5%  extended canary      Day 3–4 — A/B on task success rate
Stage 3  25% beta                 Day 5–7 — collect user feedback
Stage 4  100% full rollout        Day 8   — keep previous version warm
```

> **Key rule:** Never deploy to 100% on Friday. Gradual rollout saves weekends.

## Kubernetes setup

Three replicas minimum, autoscaled to 20 via HPA. The ingress has `proxy-buffering: off` and a 300-second read timeout so SSE streams aren't severed mid-token. The PodDisruptionBudget ensures at least 2 pods survive node drains.

## Cost control

Every request is checked against a per-user daily budget ($10 default) and a total daily cap ($1000 default). Alerts fire at 70%, 90%, and 100% of the total budget. The free tier limit is $0.50/day; enterprise users get $50/day.

## Multi-region routing

| Region | Primary LLM | LLM Latency |
|--------|-------------|-------------|
| us-east | OpenAI | ~50ms |
| eu-west | OpenAI | ~80ms (GDPR-compliant) |
| ap-southeast | Anthropic | ~120ms |

EU users are always routed to `eu-west` for GDPR data residency. If a region's circuit breaker is open, the nearest healthy region is used automatically.

## Rollback

A rollback is not just `git revert`. The `RollbackManager` handles six artifacts in order of speed: config (10s) → prompt (10s) → model (30s) → tools (30s) → code (60s) → documents (300s).

## Running

```bash
# Demo (no server needed)
python deployment_manager.py

# Run the FastAPI service
pip install fastapi uvicorn pydantic-settings
uvicorn agent_service:app --reload

# Load test (requires running service)
python load_tester.py --url http://localhost:8000 --scenario sustained

# Tests
pip install pytest pytest-asyncio
pytest test_deployment.py -v

# Docker
docker compose up
```

See also: [Node.js port](../../nodejs/09-deployment/deployment_manager.ts) · [Go port](../../go/09-deployment/deployment_manager.go)

---

## 12-Factor Agent Self-Assessment

This folder also includes a complete production-readiness assessment toolkit based on the 12-Factor Agent framework. Use these tools to measure where your agent sits on the maturity scale and get an actionable roadmap to the next level.

Reference: [docs/09-from-dev-to-production/02-the-12-factor-agent.md](../../../docs/09-from-dev-to-production/02-the-12-factor-agent.md)

### Assessment tools

| File | Purpose |
|------|---------|
| `twelve_factor_assessor.py` | Core assessment engine — evaluates all 12 factors, computes maturity level, generates markdown/HTML reports |
| `twelve_factor_validator.py` | Static-analysis engine — scans your actual codebase and returns pass/fail/warning per factor |
| `maturity_dashboard.py` | Visual terminal dashboard (rich) + standalone HTML export |
| `ci_twelve_factor_check.py` | CLI gate for CI/CD — returns exit 0/1/2 with GitHub Actions, GitLab, and Jenkins annotations |
| `test_twelve_factor.py` | 42 pytest tests covering all four tools and edge cases |

### The 12 factors at a glance

| # | Factor | Why it matters |
|---|--------|----------------|
| I | Prompt as Code | Prompts are logic — treat them like source code |
| II | Explicit State | Conversation state must survive context window truncation |
| III | Provider Agnostic | A single provider dependency is a single point of failure |
| IV | Token Budgeting | Unbounded LLM calls will bankrupt you |
| V | Structured Everything | Raw string parsing breaks at the boundary |
| VI | Context Is a Resource | The context window is finite — budget every byte |
| VII | Defense in Depth | Every trust boundary needs independent guardrail layers |
| VIII | Graceful Degradation | Partial failure must not become total failure |
| IX | Observability First | You can't debug what you can't observe |
| X | Human in the Loop | High-stakes actions need a human checkpoint |
| XI | Continuous Evaluation | Quality regressions happen silently without measurement |
| XII | Dev-Prod Parity | Surprises in production come from gaps in parity |

### Maturity model

| Level | Name | Score | Requirement |
|-------|------|-------|-------------|
| 1 | Prototype | 12-24 | Starting point |
| 2 | Development | 25-36 | Factors I, II, V, IX ≥ 3 |
| 3 | Staging | 37-48 | Factors I-VI, VIII, IX ≥ 3 |
| 4 | Production | 49-59 | Factors I-XI ≥ 3 |
| 5 | Elite | 60 | All factors ≥ 4 |

### Quick start

```bash
# Run the self-assessment demo
python twelve_factor_assessor.py

# Scan your codebase
python twelve_factor_validator.py --codebase-path /path/to/your/agent

# Visual dashboard
python maturity_dashboard.py

# CI/CD gate
python ci_twelve_factor_check.py \
  --codebase-path . \
  --minimum-level staging \
  --output-format github-actions

# Run tests
pytest test_twelve_factor.py -v
```

### CI/CD integration

Add a quality gate to your pipeline:

```yaml
# GitHub Actions
- name: 12-Factor Agent Check
  run: |
    python ci_twelve_factor_check.py \
      --codebase-path . \
      --minimum-level ${{ env.MIN_MATURITY_LEVEL }} \
      --output-format github-actions \
      --baseline-file .twelve_factor_baseline.json
```

Exit codes: `0` = all checks pass · `1` = blocking failures (deployment blocked) · `2` = warnings only

> **Key insight:** The 12 factors are not a checklist you complete once. They are a continuous discipline. Every new feature can break parity; every new provider can bypass abstraction; every new tool call can skip a guardrail.

See also: [Node.js assessor](../../nodejs/09-deployment/twelve_factor_assessor.ts) · [Go assessor](../../go/09-deployment/twelve_factor_assessor.go)
