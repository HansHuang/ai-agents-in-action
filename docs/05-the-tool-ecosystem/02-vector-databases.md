# Vector Databases

## What You'll Learn
- What a vector database is and why you can't just use PostgreSQL for everything
- The vector database landscape: Pinecone, Weaviate, Qdrant, Chroma, pgvector, and Milvus
- Choosing a vector database: managed vs. self-hosted, scale, filtering, cost
- Approximate Nearest Neighbor (ANN) search: how vector DBs find results fast
- Hybrid search: combining vector similarity with keyword matching
- Production patterns: indexing, sharding, and keeping embeddings in sync

## Prerequisites
- [Embeddings and Vectors](../03-memory-and-retrieval/02-embeddings-and-vectors.md) — what embeddings are
- [RAG from Scratch](../03-memory-and-retrieval/03-rag-from-scratch.md) — how vector databases fit into RAG
- [Model Providers](01-model-providers.md) — embedding models generate the vectors you store

---

## Why Not Just Use a JSON File?

In an earlier chapter, you built a `SimpleVectorStore` — an in-memory list that does brute-force cosine similarity search. It works for 100 documents. Here's what happens as you scale:

| Documents | Brute Force Time | Memory | Problem |
|:---|:---|:---|:---|
| 100 | <1ms | ~1MB | None |
| 10,000 | ~50ms | ~100MB | Noticeable latency |
| 100,000 | ~500ms | ~1GB | Memory pressure |
| 1,000,000 | ~5 seconds | ~10GB | Unusable — and this is a small corpus |
| 10,000,000 | ~50 seconds | ~100GB | Impossible on any single machine |

These numbers assume 1 536-dimension float32 vectors (OpenAI `text-embedding-3-small`). Larger models (3 072 dimensions) double the memory and latency.

A vector database solves this with **Approximate Nearest Neighbor (ANN)** search. Instead of comparing your query against every single vector, it uses an index to find approximate matches in milliseconds — regardless of scale.

---

## How ANN Search Works (The 30-Second Version)

Vector databases build indexes that group similar vectors together:

1. **During ingestion**, vectors are organized into clusters or graphs
2. **During search**, the database navigates to the right cluster first, then searches within it
3. **Result:** 99% accuracy in <10ms, instead of 100% accuracy in 5 seconds

The most common index types:

| Index | How It Works | Best For |
|:---|:---|:---|
| **HNSW** (Hierarchical Navigable Small World) | Builds a multi-layer graph. Search navigates from top (coarse) to bottom (fine). | General purpose, high recall — default in Qdrant, Weaviate, pgvector |
| **IVF** (Inverted File) | Clusters vectors into groups (Voronoi cells). Search checks only the nearest clusters. | Large datasets, disk-based storage (Milvus IVF_FLAT) |
| **PQ** (Product Quantization) | Compresses vectors into byte codes — 32× smaller, ~5% accuracy loss. | Memory-constrained environments; often combined with IVF |
| **LSH** (Locality-Sensitive Hashing) | Hashes similar vectors to the same bucket with high probability. | Streaming, approximate real-time; rarely used in modern databases |
| **DiskANN** | Graph index optimised for SSD reads. | Billion-scale datasets that don't fit in RAM (Azure AI Search) |

You rarely need to choose an index type manually. Most managed vector databases handle this automatically. The only time you'll configure it is when tuning the recall/latency trade-off at scale (e.g., raising HNSW's `ef_construction` for higher recall at the cost of build time).

---

## The Vector Database Landscape

### Managed (Zero Ops)

| Database | Best For | Pricing | Key Feature |
|:---|:---|:---|:---|
| **Pinecone** | Teams that don't want to manage infrastructure | Serverless (usage-based) or pod-based (~$70/month min) | Easiest setup; serverless tier scales to zero |
| **Zilliz Cloud** (managed Milvus) | Enterprise scale, billion+ vectors | Usage-based | Managed Milvus with all features |
| **Weaviate Cloud** | Hybrid search (vector + keyword) | Usage-based, free tier | GraphQL-native, built-in modules |
| **Qdrant Cloud** | Performance-sensitive apps | Usage-based, free tier | Rust-based, very fast, rich filtering |

### Self-Hosted (You Manage)

| Database | Best For | Deployment | Key Feature |
|:---|:---|:---|:---|
| **Chroma** | Prototyping, small-to-medium datasets | Embedded or server | Simplest API, Python-native |
| **Qdrant** | Performance, rich filtering | Docker, binary | Payload filtering, quantization |
| **Weaviate** | Hybrid search, modular architecture | Docker, K8s | Built-in vectorizer modules |
| **Milvus** | Billion-scale, distributed | K8s, Docker Compose | Most scalable open-source option |
| **pgvector** | Teams already on PostgreSQL | Postgres extension | No new infrastructure, SQL queries |
| **Elasticsearch** | Teams already on Elastic | Plugin | Full-text + vector in one engine |

---

## Choosing a Vector Database: The Decision Matrix

```
Start here:
│
├── Are you prototyping with <10K documents?
│   └── YES → Chroma (embedded, zero config)
│
├── Are you already on PostgreSQL?
│   └── YES → pgvector (no new infrastructure)
│
├── Do you have an ops team?
│   ├── NO → Pinecone or Weaviate Cloud (managed)
│   └── YES → Qdrant or Milvus (self-hosted)
│
├── Do you need hybrid search (vector + keyword)?
│   └── YES → Weaviate or Elasticsearch
│
├── Are you planning to store 100M+ vectors?
│   └── YES → Milvus (purpose-built for scale)
│
└── Do you need the absolute lowest latency?
    └── YES → Qdrant (Rust, optimized for speed)
```

---

## Production-Ready Vector DB Abstraction

Just like you abstracted LLM providers, abstract your vector database:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class VectorDocument:
    id: str
    text: str
    embedding: list[float]
    metadata: dict = None

@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: dict = None

class VectorDatabase(ABC):
    """Abstract interface for vector databases."""
    
    @abstractmethod
    def insert(self, documents: list[VectorDocument]) -> int:
        """Insert documents. Returns count inserted."""
        ...
    
    @abstractmethod
    def search(self, query_embedding: list[float], k: int = 5,
               filter_metadata: dict = None) -> list[SearchResult]:
        """Search for similar documents."""
        ...
    
    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """Delete documents by ID. Returns count deleted."""
        ...
    
    @abstractmethod
    def count(self) -> int:
        """Total document count."""
        ...
    
    @abstractmethod
    def clear(self) -> None:
        """Delete all documents."""
        ...

class ChromaDB(VectorDatabase):
    def __init__(self, collection_name: str = "documents",
                 persist_directory: str = "./chroma_db"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
    
    def insert(self, documents: list[VectorDocument]) -> int:
        self.collection.add(
            ids=[doc.id for doc in documents],
            embeddings=[doc.embedding for doc in documents],
            documents=[doc.text for doc in documents],
            metadatas=[doc.metadata or {} for doc in documents]
        )
        return len(documents)
    
    def search(self, query_embedding: list[float], k: int = 5,
               filter_metadata: dict = None) -> list[SearchResult]:
        where_filter = filter_metadata if filter_metadata else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where_filter
        )
        
        return [
            SearchResult(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                score=1 - results["distances"][0][i],  # Chroma returns distance
                metadata=results["metadatas"][0][i]
            )
            for i in range(len(results["ids"][0]))
        ]
    
    def delete(self, ids: list[str]) -> int:
        self.collection.delete(ids=ids)
        return len(ids)
    
    def count(self) -> int:
        return self.collection.count()
    
    def clear(self) -> None:
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(
            name=self.collection.name
        )

class QdrantDB(VectorDatabase):
    def __init__(self, collection_name: str = "documents",
                 host: str = "localhost", port: int = 6333):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        self.client = QdrantClient(host=host, port=port)
        
        # Create collection if it doesn't exist
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding size
                    distance=Distance.COSINE
                )
            )
        
        self.collection_name = collection_name
    
    def insert(self, documents: list[VectorDocument]) -> int:
        from qdrant_client.models import PointStruct
        
        points = [
            PointStruct(
                id=doc.id,
                vector=doc.embedding,
                payload={
                    "text": doc.text,
                    **(doc.metadata or {})
                }
            )
            for doc in documents
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        return len(documents)
    
    def search(self, query_embedding: list[float], k: int = 5,
               filter_metadata: dict = None) -> list[SearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        search_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(
                    key=key,
                    match=MatchValue(value=value)
                )
                for key, value in filter_metadata.items()
            ]
            search_filter = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=search_filter
        )
        
        return [
            SearchResult(
                id=str(result.id),
                text=result.payload["text"],
                score=result.score,
                metadata={k: v for k, v in result.payload.items() 
                         if k != "text"}
            )
            for result in results
        ]
    
    def delete(self, ids: list[str]) -> int:
        from qdrant_client.models import PointIdsList
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids)
        )
        return len(ids)
    
    def count(self) -> int:
        return self.client.count(
            collection_name=self.collection_name
        ).count
    
    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)

class PineconeDB(VectorDatabase):
    def __init__(self, api_key: str, index_name: str = "documents",
                 dimension: int = 1536):
        from pinecone import Pinecone
        
        self.client = Pinecone(api_key=api_key)
        
        # Create index if it doesn't exist
        if index_name not in self.client.list_indexes().names():
            self.client.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine"
            )
        
        self.index = self.client.Index(index_name)
    
    def insert(self, documents: list[VectorDocument]) -> int:
        vectors = [
            {
                "id": doc.id,
                "values": doc.embedding,
                "metadata": {
                    "text": doc.text,
                    **(doc.metadata or {})
                }
            }
            for doc in documents
        ]
        
        self.index.upsert(vectors=vectors)
        return len(documents)
    
    def search(self, query_embedding: list[float], k: int = 5,
               filter_metadata: dict = None) -> list[SearchResult]:
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            filter=filter_metadata,
            include_metadata=True
        )
        
        return [
            SearchResult(
                id=match["id"],
                text=match["metadata"]["text"],
                score=match["score"],
                metadata={k: v for k, v in match["metadata"].items() 
                         if k != "text"}
            )
            for match in results["matches"]
        ]
    
    def delete(self, ids: list[str]) -> int:
        self.index.delete(ids=ids)
        return len(ids)
    
    def count(self) -> int:
        return self.index.describe_index_stats()["total_vector_count"]
    
    def clear(self) -> None:
        self.index.delete(delete_all=True)

class VectorDBFactory:
    """Create vector databases from configuration."""
    
    DATABASES = {
        "chroma": ChromaDB,
        "qdrant": QdrantDB,
        "pinecone": PineconeDB,
        "simple": SimpleVectorStore,  # In-memory for testing
    }
    
    @classmethod
    def create(cls, db_type: str, **kwargs) -> VectorDatabase:
        db_class = cls.DATABASES.get(db_type)
        if not db_class:
            raise ValueError(f"Unknown database: {db_type}. "
                           f"Available: {list(cls.DATABASES.keys())}")
        return db_class(**kwargs)
```

> **Code Reference:** [Python](../../code/python/04-rag-pipeline/) · [Node.js](../../code/nodejs/04-rag-pipeline/) · [Go](../../code/go/04-rag-pipeline/)  
> The RAG pipeline folder includes `vector_database.py` with the full `VectorDatabase` abstraction, `ChromaDB`, `QdrantDB`, `PineconeDB`, `PgvectorDB`, and `SimpleVectorStore` implementations, plus `VectorDBFactory` for config-driven construction.

---

## Hybrid Search: Vectors + Keywords

Pure vector search fails when exact keywords matter. "Error code 503" and "server error" are semantically similar — their embeddings land close together in vector space — but only one contains the exact error code the user is searching for. This matters for product SKUs, error codes, names, version numbers, and any term where meaning doesn't survive paraphrase.

Hybrid search combines both:

```python
def hybrid_search(query: str, query_embedding: list[float],
                  vector_db: VectorDatabase,
                  keyword_index,  # Your keyword search index
                  k: int = 5,
                  vector_weight: float = 0.7) -> list[SearchResult]:
    """
    Combine vector similarity and keyword relevance.
    
    vector_weight=0.7 means 70% vector, 30% keyword.
    """
    # Get results from both
    vector_results = vector_db.search(query_embedding, k=k*2)
    keyword_results = keyword_index.search(query, k=k*2)
    
    # Normalize scores to 0-1
    vector_scores = normalize_scores([r.score for r in vector_results])
    keyword_scores = normalize_scores([r.score for r in keyword_results])
    
    # Combine with weighting
    combined = {}
    for result, v_score in zip(vector_results, vector_scores):
        combined[result.id] = {
            "result": result,
            "score": v_score * vector_weight
        }
    
    for result, k_score in zip(keyword_results, keyword_scores):
        if result.id in combined:
            combined[result.id]["score"] += k_score * (1 - vector_weight)
        else:
            combined[result.id] = {
                "result": result,
                "score": k_score * (1 - vector_weight)
            }
    
    # Sort by combined score
    sorted_results = sorted(
        combined.values(),
        key=lambda x: x["score"],
        reverse=True
    )
    
    return [item["result"] for item in sorted_results[:k]]
```

Some vector databases (Weaviate, Elasticsearch, Qdrant) have built-in hybrid search with sparse+dense retrieval. If yours doesn't, implement it as a post-processing step.

**Reciprocal Rank Fusion (RRF)** is an alternative to weighted-sum that is often more robust. Instead of normalising raw scores (which can be on very different scales), RRF scores each document by its rank position across both lists:

```
RRF_score(doc) = Σ  1 / (k + rank_i)
```

With k = 60, a document ranked #1 in vector search contributes 1/61 ≈ 0.016; one ranked #30 contributes 1/90 ≈ 0.011. This gracefully handles the case where one ranker returns many good results and the other returns only a few.

---

## Keeping Embeddings in Sync

Documents change. Embeddings must change with them. A stale embedding returns irrelevant results.

### The Sync Strategy

```python
class EmbeddingSyncManager:
    """
    Keep vector database in sync with source documents.
    """
    
    def __init__(self, vector_db: VectorDatabase, 
                 embedder, source_repo):
        self.vector_db = vector_db
        self.embedder = embedder
        self.source_repo = source_repo  # Your document storage
    
    def full_resync(self) -> dict:
        """
        Re-embed all documents and replace the entire index.
        Use for initial load or major schema changes.
        """
        self.vector_db.clear()
        documents = self.source_repo.get_all()
        embedded = [self._embed_document(doc) for doc in documents]
        count = self.vector_db.insert(embedded)
        return {"inserted": count, "strategy": "full_resync"}
    
    def incremental_sync(self, since_timestamp: float = None) -> dict:
        """
        Only update documents that have changed.
        """
        changes = self.source_repo.get_changes(since=since_timestamp)
        
        # Delete removed documents
        if changes["deleted"]:
            self.vector_db.delete(changes["deleted"])
        
        # Update modified documents
        if changes["updated"]:
            updated = [self._embed_document(doc) 
                      for doc in changes["updated"]]
            # Delete old versions, insert new
            self.vector_db.delete([doc.id for doc in changes["updated"]])
            self.vector_db.insert(updated)
        
        # Insert new documents
        if changes["created"]:
            created = [self._embed_document(doc) 
                      for doc in changes["created"]]
            self.vector_db.insert(created)
        
        return {
            "created": len(changes.get("created", [])),
            "updated": len(changes.get("updated", [])),
            "deleted": len(changes.get("deleted", [])),
            "strategy": "incremental_sync"
        }
    
    def _embed_document(self, doc) -> VectorDocument:
        return VectorDocument(
            id=doc.id,
            text=doc.text,
            embedding=self.embedder.embed(doc.text),
            metadata=doc.metadata
        )
```

---

## Production Patterns

### Pattern 1: Namespace Isolation

Use namespaces or collections to separate different types of data:

```python
# Separate collections for different data types
support_docs_db = VectorDBFactory.create("qdrant", collection_name="support_docs")
product_catalog_db = VectorDBFactory.create("qdrant", collection_name="products")
user_manuals_db = VectorDBFactory.create("qdrant", collection_name="manuals")

# Search only the relevant namespace
def search_knowledge_base(query_embedding, query_type):
    if query_type == "support":
        return support_docs_db.search(query_embedding)
    elif query_type == "product":
        return product_catalog_db.search(query_embedding)
    else:
        # Fall back to searching all
        results = []
        for db in [support_docs_db, product_catalog_db, user_manuals_db]:
            results.extend(db.search(query_embedding, k=3))
        return sorted(results, key=lambda r: r.score, reverse=True)[:5]
```

### Pattern 2: Tiered Storage

Hot data in fast storage, cold data in cheap storage:

```python
class TieredVectorDB(VectorDatabase):
    """Hot data in Qdrant (fast), cold data in S3 + pgvector (cheap)."""
    
    def __init__(self, hot_db: VectorDatabase, cold_db: VectorDatabase,
                 hot_threshold_days: int = 30):
        self.hot_db = hot_db
        self.cold_db = cold_db
        self.hot_threshold_days = hot_threshold_days
    
    def search(self, query_embedding, k=5, filter_metadata=None):
        # Search hot first
        hot_results = self.hot_db.search(query_embedding, k=k, 
                                         filter_metadata=filter_metadata)
        
        # If hot results are strong enough, return them
        if hot_results and hot_results[0].score > 0.8:
            return hot_results
        
        # Otherwise, search cold too and merge
        cold_results = self.cold_db.search(query_embedding, k=k,
                                           filter_metadata=filter_metadata)
        
        return self._merge_results(hot_results, cold_results, k)
```

### Pattern 3: Connection Pooling

Vector DB connections are expensive. Pool them:

```python
class VectorDBPool:
    """Simple connection pool for vector databases."""
    
    def __init__(self, factory, max_connections: int = 10, **db_kwargs):
        self.factory = factory
        self.max_connections = max_connections
        self.db_kwargs = db_kwargs
        self.pool = []
        self.in_use = set()
    
    def acquire(self) -> VectorDatabase:
        if self.pool:
            db = self.pool.pop()
        else:
            db = self.factory(**self.db_kwargs)
        self.in_use.add(id(db))
        return db
    
    def release(self, db: VectorDatabase):
        self.in_use.discard(id(db))
        if len(self.pool) < self.max_connections:
            self.pool.append(db)
```

---

## Common Pitfalls

- **"I use the same vector database for dev and prod"**: Dev experiments can corrupt prod data. Use separate instances or at minimum separate collections.
- **"I never re-index after changing embedding models"**: Switching from `text-embedding-ada-002` to `text-embedding-3-small`? All your old embeddings are now in a different vector space — mixing them silently degrades search quality without any error. You must re-embed everything.
- **"I store embeddings but don't version them"**: When you change your chunking strategy or embedding model, you need to know which documents have old embeddings. Store `embedding_model` and `chunking_version` in metadata.
- **"I use cosine distance but my vectors aren't normalized"**: Some databases (Qdrant, pgvector) normalise internally; others don't. When in doubt, normalise before storing. A non-normalised vector will still return results — they'll just be subtly wrong.
- **"I search with no metadata filters"**: A customer asking about their order should only search documents they're authorized to see. Always apply access control filters.
- **"I treat vector DB as the source of truth"**: The vector database is a derived index — it can always be rebuilt from source documents. Deleting your Pinecone index is not data loss. Deleting your source documents is. Keep the two concerns separate.
- **"I search without access control"**: A customer asking about their order should only search documents they are authorised to see. Always apply tenant or permission filters. Most databases support metadata filtering at query time; use it.

## What's Next

You can now choose and integrate any vector database into your RAG pipeline. Next: observability — tracing every agent decision, measuring token usage, and debugging production agents.
→ [Agent Observability](03-agent-observability.md)