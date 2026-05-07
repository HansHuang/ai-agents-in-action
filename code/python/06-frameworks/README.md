# 06 — Frameworks in Practice

This folder demonstrates the **build vs. buy decision** for AI agent frameworks — including multi-agent frameworks. Every script runs the same task, letting you compare tradeoffs by reading code instead of blog posts.

## The Hybrid Philosophy

> Framework for commodities. Custom code for differentiation.  
> **Never let a framework own your agent's brain.**

## Files

### Multi-Agent Frameworks (CrewAI & AutoGen)

| File | What it demonstrates |
|:-----|:---------------------|
| `crewai_research_crew.py` | 4-agent research crew (Researcher → Analyst → FactChecker → Writer) using CrewAI's role-based orchestration; from-scratch equivalent included for comparison |
| `autogen_design_team.py` | 5-agent product design conversation using AutoGen's group-chat pattern; `CustomConversationalTeam` fallback runs without AutoGen installed |
| `multi_agent_comparison.py` | **Same task, three ways**: from-scratch, CrewAI, AutoGen — measured across code complexity, execution time, token cost, output quality, and developer control |
| `over_engineering_detector.py` | Rule-based + optional LLM-as-judge tool that warns when multi-agent is overkill; includes a cost comparison (5-agent vs. 1-agent) |

### LangChain & LangGraph

| File | What it demonstrates |
|:-----|:---------------------|
| `langchain_rag_pipeline.py` | RAG pipeline built with LangChain; includes a four-step extraction path back to zero-dependency code |
| `langgraph_react_agent.py` | The Chapter 02 ReAct agent rebuilt as a LangGraph StateGraph; side-by-side comparison with the from-scratch agent |
| `langgraph_multi_agent.py` | Research → Fact-Check → Write → Edit workflow with a writer/editor revision loop |
| `langsmith_tracer.py` | Add LangSmith tracing to any agent — LangChain or custom |

### Framework Decision Tools

| File | What it demonstrates |
|:-----|:---------------------|
| `framework_comparison.py` | Same RAG agent built from-scratch, LangChain, and LangGraph — side-by-side metrics |
| `hybrid_rag_agent.py` | LangChain for document loading + vector storage; custom code for agent logic |
| `framework_extraction.py` | Step-by-step extraction from full LangChain to zero-dependency custom code |
| `framework_advisor.py` | Interactive questionnaire → personalised framework recommendation |

### Tests

| File | Coverage |
|:-----|:---------|
| `test_multi_agent.py` | CrewAI crew, AutoGen/conversational team, comparison metrics, over-engineering detector |
| `test_langchain.py` | LangGraph ReAct agent, LangChain RAG, LangSmith tracer, multi-agent workflow |
| `test_frameworks.py` | framework_comparison, hybrid_rag_agent, framework_advisor, framework_extraction |

## Each Framework's Sweet Spot

| Framework | Use when… | Avoid when… |
|:----------|:----------|:------------|
| **CrewAI** | Structured projects with defined roles and clear deliverables | Real-time, simple single-agent tasks, strict latency requirements |
| **AutoGen** | Open-ended problems requiring conversation and emergent solutions | Predictable workflows, low-latency requirements, tight cost budgets |
| **LangChain** | You need 700+ integrations or a quick RAG prototype | You need full control of prompt format or token budget |
| **LangGraph** | You have branching logic, loops, or human-in-the-loop | Your agent is a simple linear chain with 1-3 tool calls |
| **LangSmith** | You need tracing and eval across any agent type | You have no debugging or evaluation requirements |
| **From scratch** | You understand the concepts and value maintainability | You need a connector to Pinecone, Weaviate, or 100+ sources |

## CrewAI vs. AutoGen

| | CrewAI | AutoGen |
|:---|:---|:---|
| **Mental model** | Organization (roles, tasks, hierarchy) | Conversation (dialogue, turn-taking, emergence) |
| **Workflow** | Predefined task dependencies | Emergent from group chat |
| **Best for** | Structured projects with clear deliverables | Open-ended problems requiring discussion |
| **TypeScript support** | None (May 2026) | None (May 2026) |
| **Go support** | None | None |

## Quick Start

```bash
pip install openai numpy pytest
# Multi-agent frameworks (optional — scripts degrade gracefully without them):
pip install crewai pyautogen
# LangChain/LangGraph (optional):
pip install langchain langchain-openai langchain-community langgraph faiss-cpu
```

**Run the same task through all three approaches:**
```bash
python multi_agent_comparison.py
```

**CrewAI research crew:**
```bash
python crewai_research_crew.py
```

**AutoGen (or fallback) product design team:**
```bash
python autogen_design_team.py
```

**Check a design for over-engineering:**
```bash
python over_engineering_detector.py
```

**LangGraph ReAct agent with step-by-step streaming:**
```bash
python langgraph_react_agent.py
```

**Framework Advisor — personalised recommendation:**
```bash
python framework_advisor.py --preset expert        # Non-interactive demo
python framework_advisor.py                        # Interactive
```

## Run the Tests

```bash
pytest test_multi_agent.py test_frameworks.py test_langchain.py -v
pytest test_multi_agent.py test_frameworks.py -v -m "not integration"  # Skip LLM calls
```

## The Extraction Path

`langchain_rag_pipeline.py` includes `extract_to_custom()` — a four-step guide to incrementally replace LangChain components with custom code:

1. Replace `create_retrieval_chain` with custom prompt assembly
2. Replace the FAISS vector store with `SimpleVectorStore`
3. Replace document loaders with standard file I/O
4. Complete extraction — zero LangChain dependencies

## Key Insights

**Multi-agent is a tradeoff, not an upgrade.** You're trading simplicity, speed, and cost for specialization and collaboration. Only make that trade if you need what you're getting.

**The over-engineering test:** If you can't explain why you need multiple agents in one sentence, you probably don't need multiple agents. Most chatbots, FAQ bots, and CRUD assistants work better as a single agent with good tools.

**Use frameworks for commodities. Own your agent's brain.** LangChain's 700+ connectors are genuinely useful. Its orchestration layer is a liability. Write your own agent loop — you'll debug it 10x faster.

## Cross-Language Ports

- TypeScript: [code/nodejs/06-frameworks/multi_agent_comparison.ts](../../../nodejs/06-frameworks/multi_agent_comparison.ts) — from-scratch, LangChain.js, and conversational (AutoGen pattern)
- TypeScript: [code/nodejs/06-frameworks/langgraph_react_agent.ts](../../../nodejs/06-frameworks/langgraph_react_agent.ts) — LangGraph.js port with strict TypeScript
- TypeScript: [code/nodejs/06-frameworks/hybrid_rag_agent.ts](../../../nodejs/06-frameworks/hybrid_rag_agent.ts) — Vercel AI SDK + LangChain.js
- Go: [code/go/06-frameworks/langgraph_alternative.go](../../../go/06-frameworks/langgraph_alternative.go) — Go-native state machine; proves the graph-based concept transfers without LangGraph
- Go: [code/go/06-frameworks/hybrid_rag_agent.go](../../../go/06-frameworks/hybrid_rag_agent.go) — pure-Go with `go-openai` SDK

## Related Docs

- [When to Use Frameworks](../../../docs/06-frameworks-in-practice/01-when-to-use-frameworks.md)
- [LangChain and LangGraph](../../../docs/06-frameworks-in-practice/02-langchain-langgraph.md)
- [CrewAI and AutoGen](../../../docs/06-frameworks-in-practice/03-crewai-autogen.md)
- [Vercel AI SDK](../../../docs/06-frameworks-in-practice/04-vercel-ai-sdk.md)
