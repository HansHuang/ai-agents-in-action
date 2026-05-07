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


## Getting Started

1. Clone the repo
2. Copy `.env.example` to `.env` and add your API keys
3. Pick your language, enter `code/<language>/01-basic-llm-call/`, and run

## Philosophy

- **From scratch first, frameworks later.** Understand what's happening under the hood.
- **Concepts over code.** The markdown docs are the primary artifact; the code proves the concepts.
- **Language shouldn't be a barrier.** Identical logic across Python, Node.js, and Go.
- **Production realism.** Evaluation, guardrails, observability — not just prototypes.

## Roadmap

- [x] Foundations: LLMs, prompts, structured output
- [x] The agent loop: anatomy, tools, planning, multi-agent
- [x] Memory and retrieval: short-term, embeddings, RAG
- [ ] The tool ecosystem: providers, vector DBs, observability
- [ ] Frameworks in practice: LangChain, LangGraph, CrewAI, Vercel AI SDK
- [ ] Evaluation and guardrails
- [ ] From dev to production