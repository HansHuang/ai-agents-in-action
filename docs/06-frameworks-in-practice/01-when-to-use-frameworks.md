# When to Use Frameworks

## What You'll Learn
- The build vs. buy decision for AI agent frameworks
- What frameworks actually give you: abstraction, ecosystem, and patterns
- What frameworks cost you: complexity, debugging difficulty, and lock-in
- A decision matrix for choosing between LangChain, CrewAI, Vercel AI SDK, and from-scratch
- The hybrid approach: using frameworks for what they're good at, custom code for the rest

## Prerequisites
- [Anatomy of an AI Agent](../02-the-agent-loop/01-anatomy-of-an-agent.md) — you've built agents from scratch
- [Tool Design Patterns](../02-the-agent-loop/02-tool-design-patterns.md) — frameworks wrap these patterns
- [Model Providers](../05-the-tool-ecosystem/01-model-providers.md) — frameworks sit on top of providers

---

## The Framework Question

You've built agents from scratch. You understand the orchestration loop, tool design, memory management, and context engineering. Now you're staring at LangChain's documentation and wondering: *"Should I use this?"*

The answer isn't yes or no. It's: **"Use frameworks for what they're good at. Build from scratch for everything else."**

This chapter helps you make that call.

---

## What Frameworks Actually Give You

### 1. Pre-Built Integrations

LangChain has 700+ integrations. Vector databases, document loaders, LLM providers, tools — someone has already written the connector.

```python
# From scratch: you write the Chroma integration
import chromadb
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("docs")
collection.add(ids=[...], embeddings=[...], documents=[...])

# With LangChain: one-liner (after pip install)
from langchain_community.vectorstores import Chroma
vectordb = Chroma.from_documents(documents, embeddings)
```

This is genuinely valuable. Writing integrations is boilerplate. Using pre-built ones saves days.

### 2. Battle-Tested Patterns

Frameworks encode patterns that work. LangChain's `create_retrieval_chain` has been used in thousands of production systems. You're not discovering edge cases — they've been discovered and fixed.

### 3. Rapid Prototyping

Going from idea to working demo is fast. LangChain's chain composition, CrewAI's agent definitions, Vercel AI SDK's streaming — you can build a working prototype in hours.

### 4. Ecosystem and Community

StackOverflow answers, GitHub issues, blog posts, and tutorials exist for the popular frameworks. When something breaks, someone else has probably already fixed it.

---

## What Frameworks Cost You

### 1. Abstraction Layers That Obscure

```python
# LangChain: what's actually happening here?
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

chain = create_retrieval_chain(retriever, combine_docs_chain)
result = chain.invoke({"input": "What's our return policy?"})

# The same thing, from scratch:
query_embedding = embedder.embed("What's our return policy?")
docs = vector_db.search(query_embedding, k=5)
context = format_docs(docs)
prompt = f"Answer using these documents:\n{context}\n\nQuestion: {query}"
response = llm.chat([{"role": "user", "content": prompt}])
```

The LangChain version is shorter. But when `chain.invoke()` returns `{"answer": "I don't know"}` instead of a grounded answer, you have seven abstraction layers to debug through — and the framework will not tell you that your retriever returned zero documents.

### 2. Dependency Hell

LangChain's dependency tree is large. A full install of `langchain`, `langchain-community`, `langchain-openai`, and `langgraph` pulls in 40–60 packages total. Version conflicts across these sub-packages are common — especially when one of them pins a transitive dependency to a version that another requires to be newer.

### 3. Breaking Changes

The LangChain API has historically been unstable. Methods were deprecated, classes renamed, and entire modules reorganized between minor releases. Production code that worked in January often failed in June.

This is genuinely better now. LangChain v0.3 (2024) introduced a formal deprecation policy, and LangChain Expression Language (LCEL) is the stable surface. But the old `chain` classes are still present, still used in tutorials, and still break. If you adopt LangChain, use LCEL and pin your versions.

### 4. Learning a Framework, Not a Skill

The concepts you learned in this repo — orchestration loops, tool contracts, context management — are universal. They work with any framework, any model, any language.

Framework-specific knowledge — LangChain's `RunnableSequence`, CrewAI's `Task` definitions — is only useful within that framework. If the framework dies or you switch jobs, that knowledge is dead.

### 5. When Things Break, Debugging Is Hard

A from-scratch agent has a clear error trace: your code → LLM API → your response handler. A framework agent has: your code → framework wrapper → internal pipeline → LLM API → response parser → output formatter → your handler. The error message you get is rarely about your actual bug.

---

## The Decision Matrix

### Use LangChain When:

| Scenario | Why |
|:---|:---|
| You need rapid prototyping (hours, not days) | Pre-built chains save enormous time |
| You need many integrations (15+ vector DBs, 20+ LLMs) | Writing all those connectors is wasteful |
| Your team already knows LangChain | Don't fight the team's expertise |
| You're building a standard RAG pipeline | LangChain's retrieval chain is proven |
| You need LangSmith for observability | Deep integration with tracing |

### Use LangGraph When:

| Scenario | Why |
|:---|:---|
| You need complex agent workflows (branching, loops, state) | StateGraph is purpose-built for this |
| You want visual workflow debugging | LangGraph Studio shows the graph execution |
| Your agent has conditional branching logic | Graph-based design maps well to agent decisions |
| You need persistence between agent steps | Built-in checkpointing |

**Don't use LangGraph** for simple linear pipelines (retrieve → augment → generate with no branching). A StateGraph adds boilerplate without benefit there; use a plain function or LangChain's LCEL instead.

### Use CrewAI When:

| Scenario | Why |
|:---|:---|
| You want multi-agent collaboration out of the box | Define agents, assign tasks, it orchestrates |
| Your team thinks in terms of roles and responsibilities | CrewAI's mental model is role-based |
| You're exploring multi-agent patterns quickly | Fastest path to a working multi-agent demo |
| You don't need fine-grained control over agent communication | CrewAI handles the handoffs |

### Use Vercel AI SDK When:

| Scenario | Why |
|:---|:---|
| You're building a full-stack TypeScript app | Native streaming, React hooks, edge runtime |
| Streaming UX is critical | Best-in-class streaming support |
| You want to switch providers easily | Unified API across OpenAI, Anthropic, Google, etc. |
| You're deploying on Vercel/edge | Optimized for serverless and edge runtimes |
| Your team is frontend-heavy | Familiar React patterns for AI integration |

### Build from Scratch When:

| Scenario | Why |
|:---|:---|
| You need complete control over every decision | No framework decisions you didn't make |
| Your use case is unique or non-standard | Frameworks optimize for common patterns |
| You're building a production system that will live for years | No dependency on framework stability |
| Your team is small and wants minimal dependencies | Less code to audit, fewer updates to track |
| You're learning how agents work | Building from scratch is the best teacher |
| Debugging transparency is critical | You can trace every line of execution |

---

## The Hybrid Approach (Recommended)

The best teams don't choose one approach. They mix them:

```
┌─────────────────────────────────────────────────────────┐
│                  PRODUCTION AGENT                        │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ YOUR CODE (the important parts)                    │ │
│  │ - Agent orchestration loop                         │ │
│  │ - Tool design and execution                        │ │
│  │ - Context assembly and management                  │ │
│  │ - State tracking and session management            │ │
│  │ - Error handling and fallback logic                │ │
│  │ - Evaluation and testing                           │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │ FRAMEWORK (the commodity parts)                    │ │
│  │ - Vector database connectors (via LangChain)       │ │
│  │ - Document loaders (via LangChain)                 │ │
│  │ - LLM provider abstraction (via Vercel AI SDK)     │ │
│  │ - Streaming infrastructure (via Vercel AI SDK)     │ │
│  │ - Observability (via LangSmith or Arize)           │ │
│  └────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

The rule: **your agent's brain is custom. Its inputs and outputs can use frameworks.**

### Example: Hybrid RAG Agent

```python
# Framework: use LangChain for document loading and vector store
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Your code: the agent loop and context assembly
class HybridRAGAgent:
    def __init__(self, docs_dir: str):
        # Framework handles the boring part
        self.loader = DirectoryLoader(docs_dir, glob="**/*.md")
        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(embedding_function=self.embeddings)
        
        # Your code handles the important part
        self.llm = YourLLMProvider()
        self.context_assembler = YourContextAssembler()
        self.memory_manager = YourMemoryManager()
    
    def ingest(self):
        """Framework does the heavy lifting."""
        docs = self.loader.load()
        self.vector_store = Chroma.from_documents(docs, self.embeddings)
    
    def query(self, question: str) -> str:
        """Your code controls the logic."""
        # Retrieve (could use framework or custom)
        docs = self.vector_store.similarity_search(question, k=5)
        
        # Assemble context (your logic — you control quality)
        context = self.context_assembler.assemble(docs, question)
        
        # Manage memory (your logic — you control budget)
        messages = self.memory_manager.get_messages(question, context)
        
        # Call LLM (your abstraction — you control provider)
        return self.llm.chat(messages)
```

The framework does the boring, standard stuff. You control the parts that differentiate your agent.

---

## Framework Comparison Table

| | LangChain | LangGraph | CrewAI | Vercel AI SDK | From Scratch |
|:---|:---|:---|:---|:---|:---|
| **Best for** | RAG, integrations | Complex workflows | Multi-agent | Full-stack TS | Full control |
| **Learning curve** | Steep | Steep | Moderate | Gentle | You already know it |
| **API stability** | Improving | Good | Moderate | Good | Depends on you |
| **Debugging** | Hard | Moderate | Hard | Moderate | Easy |
| **Streaming** | Good | Good | Limited | Excellent | Manual |
| **Provider support** | 20+ | 20+ | 5+ | 10+ | Unlimited |
| **Ecosystem** | Huge | Growing | Small | Growing | None needed |
| **Production readiness** | Yes (with care) | Yes | Early | Yes | Yes (with care) |
| **Vendor lock-in** | High | High | Moderate | Low | None |

---

## The Migration Path

Most teams follow this progression:

```
Phase 1: Build from scratch
  → Learn the concepts deeply
  → Ship something simple

Phase 2: Adopt a framework for commodity parts
  → Use LangChain for vector DB connectors
  → Use Vercel AI SDK for streaming
  → Keep agent logic custom

Phase 3: Evaluate specialized frameworks
  → Consider LangGraph for complex workflows
  → Consider CrewAI for multi-agent experiments
  → Only adopt if the benefit is clear

Phase 4: Extract from frameworks as needed
  → Replace framework components that cause problems
  → Keep framework components that work well
  → Never let the framework own your core logic
```

You're in Phase 1 right now. This repo is your Phase 1 foundation. When you hit a problem that a framework genuinely solves better than custom code, that's when you adopt it — not before.

---

## Common Pitfalls

- **"I'll just use LangChain for everything"**: LangChain is a toolkit, not an application framework. It doesn't make architectural decisions for you. If you don't understand the underlying patterns, LangChain won't save you — it'll just make your bugs harder to find.
- **"Frameworks are always slower than custom code"**: For *writing* standard operations (document loading, vector search), frameworks are faster — you write less. For *running* them, frameworks add overhead: Pydantic validation, lazy evaluation chains, and multiple Python call frames can add 10–50 ms per invocation. Whether that matters depends on your latency budget.
- **"I built everything from scratch, so I don't need frameworks"**: You'll eventually need a vector DB connector for a database you haven't used before. That's when a framework's ecosystem saves you a week of work. Don't be dogmatic.
- **"I'll start with a framework and learn the concepts later"**: This is backwards. Frameworks obscure the concepts. If you don't know what a retrieval chain does under the hood, you can't debug it when it fails. Learn concepts first, frameworks second.
- **"I need to pick one framework and commit"**: You can use LangChain for document loading, Vercel AI SDK for streaming, and your own code for the agent loop. Frameworks are tools, not marriages.

## What's Next

You understand when frameworks help and when they hurt. Next: a deep dive into LangChain and LangGraph — the dominant framework, what it does well, and where it stumbles.
→ [LangChain and LangGraph](02-langchain-langgraph.md)