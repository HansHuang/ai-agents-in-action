# ai-agents-in-action

A hands-on, knowledge-first guide to building AI agents — from raw LLM calls to production-grade systems.

## What This Is

`ai-agent-in-action` is both a **blog series** and a **polyglot codebase**. The `docs/` folder walks you through every concept, pattern, and ecosystem consideration for building AI agents. The `code/` folder provides runnable, minimal implementations in Python, Node.js, and Go so you can learn the universal patterns in the language you're most comfortable with.

## Structure

docs/ → The knowledge base. Start here.
code/ → Runnable implementations by language.
    python/
    nodejs/
    go/

## Table of Contents

- Foundations
  - [01-how-llms-work](docs/01-foundations/01-how-llms-work.md)
  - [02-prompt-engineering](docs/01-foundations/02-prompt-engineering.md)
  - [03-structured-output](docs/01-foundations/03-structured-output.md)
- The agent loop
  - [01-anatomy-of-an-agent](docs/02-the-agent-loop/01-anatomy-of-an-agent.md)
  - [02-tool-design-patterns](docs/02-the-agent-loop/02-tool-design-patterns.md)
  - [03-planning-strategies](docs/02-the-agent-loop/03-planning-strategies.md)
  - [04-multi-agent-patterns](docs/02-the-agent-loop/04-multi-agent-patterns.md)
  - [05-skills-composing-capabilities](docs/02-the-agent-loop/05-skills-composing-capabilities.md)
- Memory and retrieval
  - [01-short-term-memory](docs/03-memory-and-retrieval/01-short-term-memory.md)
  - [02-embeddings-and-vectors](docs/03-memory-and-retrieval/02-embeddings-and-vectors.md)
  - [03-rag-from-scratch](docs/03-memory-and-retrieval/03-rag-from-scratch.md)
- Context engineering
  - [01-the-context-window-as-a-resource](docs/04-context-engineering/01-the-context-window-as-a-resource.md)
  - [02-dynamic-prompt-assembly](docs/04-context-engineering/02-dynamic-prompt-assembly.md)
  - [03-context-compression-and-filtering](docs/04-context-engineering/03-context-compression-and-filtering.md)
  - [04-multi-turn-context-management](docs/04-context-engineering/04-multi-turn-context-management.md)
- The tool ecosystem
  - [01-model-providers](docs/05-the-tool-ecosystem/01-model-providers.md)
  - [02-vector-databases](docs/05-the-tool-ecosystem/02-vector-databases.md)
  - [03-agent-observability](docs/05-the-tool-ecosystem/03-agent-observability.md)
  - [04-mcp-protocol](docs/05-the-tool-ecosystem/04-mcp-protocol.md)
- Frameworks in practice
  - [01-when-to-use-frameworks](docs/06-frameworks-in-practice/01-when-to-use-frameworks.md)
  - [02-langchain-langgraph](docs/06-frameworks-in-practice/02-langchain-langgraph.md)
  - [03-crewai-autogen](docs/06-frameworks-in-practice/03-crewai-autogen.md)
  - [04-vercel-ai-sdk](docs/06-frameworks-in-practice/04-vercel-ai-sdk.md)
- Harness engineering
  - [01-the-harness-mindset](docs/07-harness-engineering/01-the-harness-mindset.md)
  - [02-input-guardrails-and-validation](docs/07-harness-engineering/02-input-guardrails-and-validation.md)
  - [03-routing-and-intent-classification](docs/07-harness-engineering/03-routing-and-intent-classification.md)
  - [04-retry-fallback-and-circuit-breakers](docs/07-harness-engineering/04-retry-fallback-and-circuit-breakers.md)
  - [05-output-guardrails-and-fact-checking](docs/07-harness-engineering/05-output-guardrails-and-fact-checking.md)
  - [06-human-in-the-loop](docs/07-harness-engineering/06-human-in-the-loop.md)
  - [07-building-a-reliable-harness](docs/07-harness-engineering/07-building-a-reliable-harness.md)
- Evaluation and guardrails
  - [01-evaluating-agents](docs/08-evaluation-and-guardrails/01-evaluating-agents.md)
  - [02-guardrails-and-safety](docs/08-evaluation-and-guardrails/02-guardrails-and-safety.md)
- From dev to production
  - [01-deployment-strategies](docs/09-from-dev-to-production/01-deployment-strategies.md)
  - [02-the-12-factor-agent](docs/09-from-dev-to-production/02-the-12-factor-agent.md)

## Getting Started

1. Clone the repo
2. Copy `.env.example` to `.env` and add your API keys
3. Pick your language, enter `code/<language>/01-basic-llm-call/`, and run

## Philosophy

- **From scratch first, frameworks later.** Understand what's happening under the hood.
- **Concepts over code.** The markdown docs are the primary artifact; the code proves the concepts.
- **Language shouldn't be a barrier.** Identical logic across Python, Node.js, and Go.
- **Production realism.** Evaluation, guardrails, observability — not just prototypes.
