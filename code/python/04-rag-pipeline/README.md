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

## What This Folder Demonstrates

From in-memory prototype to production vector database — every step:

1. **Prototype** — `simple_vector_store.py` / `SimpleVectorStore`: brute-force cosine similarity, zero dependencies, works for ≤10 K documents.
2. **Abstract** — `vector_database.py`: swap any backend with a single config change.
3. **Retrieve better** — `hybrid_search.py`: combine vector similarity with BM25 keyword ranking so "Error 503" beats "server error" when the exact code matters.
4. **Stay fresh** — `embedding_sync.py`: full and incremental sync managers keep your index in step with the source document store.

> **Key insight:** Abstract your vector database early. Switching from Chroma (dev) to Pinecone (prod) should be a one-line config change — and with `VectorDBFactory` it is.

## Files

### Vector Databases
| File | Description |
|---|---|
| `vector_database.py` | Multi-backend abstraction: Chroma, Qdrant, Pinecone, pgvector, SimpleVectorStore |
| `hybrid_search.py` | Vector + BM25 combined search; weighted-sum and RRF fusion |
| `embedding_sync.py` | Full and incremental sync manager; health verification |

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

## Supported Backends

| Backend | Best For | Install |
|---|---|---|
| `SimpleVectorStore` | Tests, prototypes, ≤10 K docs | built-in |
| `ChromaDB` | Local dev, no server needed | `pip install chromadb` |
| `QdrantDB` | Performance, rich metadata filtering | `pip install qdrant-client` |
| `PineconeDB` | Managed cloud, zero ops | `pip install pinecone-client` |
| `PgvectorDB` | Teams already on PostgreSQL | `pip install psycopg2-binary pgvector` |

Switch backends with one config change:

```python
# Development
db = VectorDBFactory.create("chroma", collection_name="my_docs")

# Production — same interface, different backend
db = VectorDBFactory.create("qdrant", host="localhost", port=6333,
                            collection_name="my_docs", dimension=1536)

# Or from a config dict (YAML/env-driven)
db = VectorDBFactory.create_from_config({
    "type": "pinecone",
    "api_key": os.environ["PINECONE_API_KEY"],
    "index_name": "my_docs",
})
```

## Architecture

```
Documents
    │
    ▼  INGEST
┌───────────────┐   chunk → embed → store
│ Vector Store  │   (VectorDatabase ABC — any backend)
└───────┬───────┘
        │  RETRIEVE
        │  embed query → vector search
        │              → BM25 keyword search (optional)
        │              → fuse with weighted sum or RRF
        ▼
┌───────────────┐   AUGMENT
│  RAG Prompt   │   inject retrieved chunks into system prompt
└───────┬───────┘
        │  GENERATE
        ▼
    LLM Answer  ←  grounded in your documents
```

## Prerequisites

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

```python
from embedding_generator import EmbeddingGenerator
from vector_database import VectorDBFactory, VectorDocument
from rag_pipeline import RAGPipeline

# Pick any backend — same code from here on
db = VectorDBFactory.create("chroma", collection_name="my_docs")
embedder = EmbeddingGenerator(model="text-embedding-3-small")
pipeline = RAGPipeline(db, embedder)

pipeline.ingest_text("Returns: 30 days with receipt.", metadata={"source": "policy.md"})
response = pipeline.query("What is the return window?")
print(response.answer)
```

## Hybrid Search Quick Start

```python
from hybrid_search import HybridSearch

hs = HybridSearch(db, vector_weight=0.7)
hs.build_keyword_index(docs)

results = hs.search("Error 503", query_embedding=embed("Error 503"), k=5)
# Doc containing literal "503" is ranked above semantically similar docs
```

## RAG + Agent Integration

RAG is a tool, not a separate system. The agent decides when to retrieve:

```python
from rag_agent import RAGAgent

agent = RAGAgent(rag_pipeline=pipeline)
result = agent.run("What is our vacation policy?")
print(result.decision_trail)
```

## Tests

```bash
# Fast tests — no external services required
pytest test_vector_database.py -m "not integration" -v

# Integration tests — requires Chroma and Qdrant running locally
pytest test_vector_database.py -v

# Full RAG suite
OPENAI_API_KEY=sk-... pytest test_rag_pipeline.py -v
```

## Docs

- [docs/03-memory-and-retrieval/03-rag-from-scratch.md](../../../docs/03-memory-and-retrieval/03-rag-from-scratch.md)
- [docs/05-the-tool-ecosystem/02-vector-databases.md](../../../docs/05-the-tool-ecosystem/02-vector-databases.md)

## Cross-Reference

- TypeScript port: [code/nodejs/04-rag-pipeline/vector_database.ts](../../nodejs/04-rag-pipeline/vector_database.ts)
- Go port: [code/go/04-rag-pipeline/vector_database.go](../../go/04-rag-pipeline/vector_database.go)
