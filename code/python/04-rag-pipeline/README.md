# 04 — RAG Pipeline: Retrieval-Augmented Generation from Scratch

> **RAG gives the LLM access to your documents. The agent decides when to use it.**

Build a complete RAG system — every line from scratch — then integrate it as a
tool in an agent loop.

## Architecture

```
Documents
    │
    ▼  INGEST
┌───────────────┐   chunk → embed → store
│ Vector Store  │
└───────┬───────┘
        │  RETRIEVE
        │  embed query → cosine search → threshold filter
        ▼
┌───────────────┐   AUGMENT
│  RAG Prompt   │   inject retrieved chunks into system prompt
└───────┬───────┘
        │  GENERATE
        ▼
    LLM Answer  ←  grounded in your documents
```

## Files

## Files

### RAG Core
| File | Description |
|---|---|
| `rag_pipeline.py` | Complete RAG pipeline — Ingest → Retrieve → Augment → Generate |
| `rag_evaluator.py` | Retrieval metrics (hit rate, MRR, precision@k) + LLM-as-judge generation scoring |
| `advanced_retriever.py` | HyDE, multi-query, decompose-and-retrieve, contextual retrieval |
| `rag_agent.py` | RAG exposed as a tool in an agent loop; agent decides when to search |
| `knowledge_base_manager.py` | Incremental add / update / remove / directory-sync |

### Supporting Modules
| File | Description |
|---|---|
| `embedding_generator.py` | OpenAI embedding generation; batch API, retry |
| `document_chunker.py` | Fixed, semantic, hierarchical, sentence chunking |
| `simple_vector_store.py` | In-memory cosine-similarity store |
| `embedding_visualizer.py` | Similarity matrices and clustering |
| `memory_manager.py` | Token-aware conversation memory |
| `conversation_summarizer.py` | LLM-based message compression |
| `branch_manager.py` | Parallel conversation contexts |
| `token_tracker.py` | API call audit trail |

### Tests
| File | Description |
|---|---|
| `test_rag_pipeline.py` | 20 tests covering ingest, retrieval, generation, E2E |
| `test_embeddings.py` | Embedding and vector store tests |
| `test_memory_manager.py` | Memory manager tests |

## Quick Start

```python
from embedding_generator import EmbeddingGenerator
from simple_vector_store import SimpleVectorStore
from rag_pipeline import RAGPipeline

embedder = EmbeddingGenerator(model="text-embedding-3-small")
vector_store = SimpleVectorStore()
pipeline = RAGPipeline(vector_store, embedder)

# Ingest documents
pipeline.ingest_text("Returns: 30 days with receipt.", metadata={"source": "policy.md"})
pipeline.ingest_text("Shipping: $4.99 standard, $14.99 express.", metadata={"source": "shipping.md"})

# Query
response = pipeline.query("What is the return window?")
print(response.answer)   # → grounded answer
print(response.sources)  # → ["policy.md"]
print(response.similarity_scores)
```

## RAG + Agent Integration

RAG is a tool, not a separate system. The agent decides when to retrieve:

```python
from rag_agent import RAGAgent

agent = RAGAgent(rag_pipeline=pipeline)
result = agent.run("What is our vacation policy?")
# Agent calls search_knowledge_base("vacation policy") automatically
print(result.decision_trail)  # shows routing decision
```

## Tests

```bash
# Offline unit tests (no API key)
pytest test_rag_pipeline.py -m "not integration" -v

# Full suite with live API calls
OPENAI_API_KEY=sk-... pytest test_rag_pipeline.py -v
```

## Docs

[docs/03-memory-and-retrieval/03-rag-from-scratch.md](../../../docs/03-memory-and-retrieval/03-rag-from-scratch.md)

## Cross-Reference

- TypeScript port: [code/nodejs/04-rag-pipeline/rag_pipeline.ts](../../nodejs/04-rag-pipeline/rag_pipeline.ts)
- Go port: [code/go/04-rag-pipeline/rag_pipeline.go](../../go/04-rag-pipeline/rag_pipeline.go)
