# Code Examples

Companion code for *AI Agents in Action*. Every concept from the book has working implementations in three languages — choose the one that matches your stack.

## Structure

```
code/
├── python/     # Canonical reference implementations
├── go/         # Idiomatic Go ports
└── nodejs/     # TypeScript / Node.js ports
```

Each numbered directory maps to a chapter and its corresponding `docs/` section.

---

## Module Overview

| Directory | Topic | Docs |
|-----------|-------|------|
| `01-basic-llm-call` | First LLM call, streaming, retries | [01-foundations](../docs/01-foundations/) |
| `02-structured-output` | JSON mode, Zod/Pydantic schemas, validation | [01-foundations/03-structured-output](../docs/01-foundations/03-structured-output.md) |
| `03-agent-loop` | Tool registry, plan-execute, reflection | [02-the-agent-loop](../docs/02-the-agent-loop/) |
| `04-rag-pipeline` | Embeddings, vector store, chunking, retrieval | [03-memory-and-retrieval](../docs/03-memory-and-retrieval/) |
| `05-context-assembly` | Context window management, dynamic prompts | [04-context-engineering](../docs/04-context-engineering/) |
| `05-the-tool-ecosystem` | Model providers, vector DBs, observability | [05-the-tool-ecosystem](../docs/05-the-tool-ecosystem/) |
| `06-frameworks` | LangChain, LangGraph, CrewAI, Vercel AI SDK | [06-frameworks-in-practice](../docs/06-frameworks-in-practice/) |
| `07-harness` | Input guardrails, routing, PII, policy engine | [07-harness-engineering](../docs/07-harness-engineering/) |
| `08-evaluation` | Retrieval eval, LLM-as-judge, dashboards | [08-evaluation-and-guardrails](../docs/08-evaluation-and-guardrails/) |
| `09-deployment` | 12-factor agents, CI checks, maturity model | [09-from-dev-to-production](../docs/09-from-dev-to-production/) |
| `09-skills` | Skill registry, composable capabilities | [02-the-agent-loop/05-skills](../docs/02-the-agent-loop/05-skills-composing-capabilities.md) |
| `10-mcp-server` | Model Context Protocol server and marketplace | [05-the-tool-ecosystem/04-mcp-protocol](../docs/05-the-tool-ecosystem/04-mcp-protocol.md) |
| `11-eval-suite` | End-to-end evaluation suite | [08-evaluation-and-guardrails](../docs/08-evaluation-and-guardrails/) |

---

## Setup

### Prerequisites

All examples require an OpenAI API key:

```bash
export OPENAI_API_KEY=sk-...
```

### Python

Requires Python 3.11+.

```bash
cd python/01-basic-llm-call
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Most Python examples use `openai`, `pydantic`, and `python-dotenv`. RAG examples also use `numpy` and `tiktoken`.

### Go

Requires Go 1.21+.

```bash
cd go/01-basic-llm-call
go mod tidy
go run main.go
```

### Node.js (TypeScript)

Requires Node.js 18+ and npm.

```bash
cd nodejs/01-basic-llm-call
npm install
npx tsx main.ts
```

All Node.js examples are written in TypeScript with strict mode enabled. Source files use `.js` extensions in imports for ESM compatibility.

---

## Running Tests

### Node.js

Each directory with tests includes a `test_*.ts` file. Run with:

```bash
# Using tsx directly (recommended — avoids ESM cycle issues)
./node_modules/.bin/tsx --test test_*.ts

# Or using Node's built-in test runner with tsx loader
node --import tsx/esm --test test_*.ts
```

Directories with passing test suites:

| Directory | Tests | Command |
|-----------|-------|---------|
| `nodejs/03-agent-loop` | 8 tests | `./node_modules/.bin/tsx --test test_agent.ts` |
| `nodejs/04-rag-pipeline` | 9 tests | `node --import tsx/esm --test test_rag.ts` |
| `nodejs/07-harness` | 17 tests | `./node_modules/.bin/tsx --test test_harness.ts` |
| `nodejs/08-evaluation` | 4 tests | `./node_modules/.bin/tsx --test test_evaluation.ts` |
| `nodejs/09-deployment` | 9 tests | `node --import tsx/esm --test test_deployment.ts` |
| `nodejs/09-skills` | 13 tests | `node --import tsx/esm --test test_skills.ts` |
| `nodejs/10-mcp-server` | 12 tests | `node --import tsx/esm --test test_mcp.ts` |

### Python

```bash
cd python/07-harness
pytest
```

### Go

```bash
cd go/03-agent-loop
go test ./...
```

---

## Key Files by Directory

### `01-basic-llm-call`
- `main.py` / `main.go` / `main.ts` — synchronous and streaming LLM calls
- `retry_client.py` / `retry_client.ts` — exponential backoff retry wrapper

### `02-structured-output`
- `structured_output.py` / `structured_output.ts` — JSON mode and schema validation
- `schema_validator.py` — Pydantic model validation

### `03-agent-loop`
- `agent.py` / `agent.ts` — core ReAct agent loop
- `tool_registry.py` / `tool_registry.ts` — tool registration and execution
- `plan_execute_agent.py` / `plan_execute_agent.ts` — two-phase planning
- `reflection_agent.py` / `reflection_agent.ts` — self-critique and retry

### `04-rag-pipeline`
- `simple_vector_store.py` / `simple_vector_store.ts` — in-memory cosine similarity store
- `document_chunker.py` / `document_chunker.ts` — token-aware text chunking
- `embedding_generator.py` / `embedding_generator.ts` — OpenAI embeddings wrapper
- `rag_pipeline.py` — end-to-end retrieval-augmented generation

### `05-the-tool-ecosystem`
- `model_provider.py` / `model_provider.ts` — provider abstraction (OpenAI, Anthropic)
- `vector_database.py` / `vector_database.ts` — pluggable vector DB interface
- `observability.py` / `observability.ts` — tracing and metrics

### `06-frameworks`
- `langchain_agent.py` / `langchain_agent.ts` — LangChain tool-calling agent
- `langgraph_agent.py` / `langgraph_agent.ts` — LangGraph stateful graph agent
- `crewai_agent.py` — CrewAI multi-agent crew
- `vercel_ai_agent.ts` — Vercel AI SDK streaming agent

### `07-harness`
- `input_guardrail_pipeline.py` / `input_guardrail_pipeline.ts` — entry-point guardrail
- `harness_policy.py` / `harness_policy.ts` — rule-based policy engine
- `pii_benchmark.py` / `pii_benchmark.ts` — PII detection with precision/recall metrics
- `safety_regression_suite.py` / `safety_regression_suite.ts` — safety test suite
- `fallback_chain.py` / `fallback_chain.ts` — circuit-breaking provider fallback

### `08-evaluation`
- `agent_evaluator.py` / `agent_evaluator.ts` — retrieval, generation, and E2E evaluation
- `evaluation_dashboard.py` / `evaluation_dashboard.ts` — ASCII evaluation dashboard
- `llm_judge.py` — LLM-as-judge for faithfulness/relevance scoring

### `09-deployment`
- `twelve_factor_assessor.py` / `twelve_factor_assessor.ts` — 12-factor agent checklist
- `twelve_factor_validator.py` / `twelve_factor_validator.ts` — automated file-based checks
- `ci_twelve_factor_check.py` / `ci_twelve_factor_check.ts` — CI/CD integration CLI
- `maturity_dashboard.py` / `maturity_dashboard.ts` — production readiness dashboard
- `deployment_manager.py` / `deployment_manager.ts` — deployment lifecycle management

### `09-skills`
- `skill_base.py` / `skill_base.ts` — `Skill`, `SkillRegistry`, dependency resolution
- `skilled_agent.py` / `skilled_agent.ts` — agent that loads skills dynamically
- `main.py` / `main.ts` — sample skills (weather, calculator, time)

### `10-mcp-server`
- `mcp_agent.py` / `mcp_agent.ts` — agent with MCP tool integration
- `simple_mcp_server.py` / `simple_mcp_server.ts` — lightweight MCP server wrapper
- `mcp_marketplace.py` / `mcp_marketplace.ts` — catalog of public MCP servers

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for all LLM calls |
| `ANTHROPIC_API_KEY` | No | Needed for multi-provider examples |
| `PINECONE_API_KEY` | No | Needed for Pinecone vector DB examples |
| `LANGCHAIN_API_KEY` | No | Needed for LangSmith tracing |
| `LANGCHAIN_TRACING_V2` | No | Set to `true` to enable LangSmith |

---

## Docker

Several deployment examples include Docker support:

```bash
cd nodejs/09-deployment
docker build -t my-agent .
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY my-agent
```

Kubernetes manifests are available in `09-deployment/k8s/`.
