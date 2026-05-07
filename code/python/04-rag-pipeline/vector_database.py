"""Vector database abstraction layer with multiple production-ready backends.

Provides a unified interface for Chroma, Qdrant, Pinecone, pgvector, and an
in-memory SimpleVectorStore.  All backends implement the same VectorDatabase
ABC so you can swap one for another with a single config change.

Usage::

    from vector_database import VectorDBFactory, VectorDocument

    # Development
    db = VectorDBFactory.create("chroma", collection_name="my_docs")

    # Production — one-line switch
    db = VectorDBFactory.create("qdrant", host="localhost", port=6333,
                                collection_name="my_docs", dimension=1536)

    # Insert
    docs = [VectorDocument(id=str(i), text=f"Doc {i}",
                           embedding=[0.1]*1536, metadata={"cat": "A"})
            for i in range(10)]
    db.batch_insert(docs)

    # Search
    results = db.search(query_embedding=[0.1]*1536, k=5)

See: docs/05-the-tool-ecosystem/02-vector-databases.md
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VectorDocument:
    """A document ready to be stored in a vector database."""

    id: str
    text: str
    embedding: list[float]
    metadata: Optional[dict] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A single result returned from a vector search."""

    id: str
    text: str
    score: float
    metadata: Optional[dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class VectorDatabase(ABC):
    """Abstract interface for vector databases.

    All backends must implement the five core operations.  ``batch_insert``
    has a default implementation that calls ``insert`` in chunks; override
    it if your database provides a more efficient native batch API.
    """

    @abstractmethod
    def insert(self, documents: list[VectorDocument]) -> int:
        """Insert (or upsert) documents.  Returns the count inserted."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        """Return the *k* nearest documents to *query_embedding*."""
        ...

    @abstractmethod
    def delete(self, ids: list[str]) -> int:
        """Delete documents by ID.  Returns count deleted."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Total number of documents currently stored."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Delete all documents."""
        ...

    def batch_insert(
        self, documents: list[VectorDocument], batch_size: int = 100
    ) -> int:
        """Insert documents in chunks.

        Override this method if your database has a more efficient native
        batch API (e.g. Qdrant's ``upsert`` with a list of points).

        Args:
            documents:  Documents to insert.
            batch_size: Number of documents per call to :meth:`insert`.

        Returns:
            Total number of documents inserted.
        """
        total = 0
        for i in range(0, len(documents), batch_size):
            chunk = documents[i : i + batch_size]
            total += self.insert(chunk)
        return total


# ---------------------------------------------------------------------------
# Backend: SimpleVectorStore (in-memory, brute-force cosine similarity)
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


class SimpleVectorStore(VectorDatabase):
    """In-memory vector store — no dependencies required.

    Uses brute-force O(n) cosine similarity.  Suitable for tests,
    prototyping, and datasets up to ~10,000 documents.
    """

    def __init__(self) -> None:
        self._docs: list[dict] = []

    # -- VectorDatabase interface --

    def insert(self, documents: list[VectorDocument]) -> int:
        for doc in documents:
            # Upsert behaviour: replace existing doc with same id
            self._docs = [d for d in self._docs if d["id"] != doc.id]
            self._docs.append(
                {
                    "id": doc.id,
                    "text": doc.text,
                    "embedding": doc.embedding,
                    "metadata": doc.metadata or {},
                }
            )
        return len(documents)

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        candidates = self._docs
        if filter_metadata:
            candidates = [
                d
                for d in candidates
                if all(d["metadata"].get(key) == val for key, val in filter_metadata.items())
            ]
        if not candidates:
            return []

        q = np.array(query_embedding, dtype=np.float64)
        scored = [
            SearchResult(
                id=d["id"],
                text=d["text"],
                score=_cosine_similarity(q, np.array(d["embedding"], dtype=np.float64)),
                metadata=d["metadata"],
            )
            for d in candidates
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:k]

    def delete(self, ids: list[str]) -> int:
        id_set = set(ids)
        before = len(self._docs)
        self._docs = [d for d in self._docs if d["id"] not in id_set]
        return before - len(self._docs)

    def count(self) -> int:
        return len(self._docs)

    def clear(self) -> None:
        self._docs.clear()


# ---------------------------------------------------------------------------
# Backend: ChromaDB
# ---------------------------------------------------------------------------


class ChromaDB(VectorDatabase):
    """Chroma vector database backend.

    Uses ``chromadb.PersistentClient`` so data survives between runs.
    Chroma returns L2 *distances* (lower = more similar); we convert to
    cosine similarity with ``1 - distance``.

    Args:
        collection_name:  Name of the Chroma collection.
        persist_directory: Directory for Chroma's on-disk storage.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: str = "./chroma_db",
    ) -> None:
        import chromadb  # type: ignore[import]

        self._collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # -- VectorDatabase interface --

    def insert(self, documents: list[VectorDocument]) -> int:
        if not documents:
            return 0
        self.collection.upsert(
            ids=[doc.id for doc in documents],
            embeddings=[doc.embedding for doc in documents],
            documents=[doc.text for doc in documents],
            metadatas=[doc.metadata or {} for doc in documents],
        )
        return len(documents)

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        where = filter_metadata if filter_metadata else None
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            where=where,
        )
        if not results["ids"] or not results["ids"][0]:
            return []
        return [
            SearchResult(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                # Chroma with cosine space returns distances in [0,2]; normalise to [0,1]
                score=1.0 - (results["distances"][0][i] / 2.0),
                metadata=results["metadatas"][0][i],
            )
            for i in range(len(results["ids"][0]))
        ]

    def delete(self, ids: list[str]) -> int:
        self.collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return self.collection.count()

    def clear(self) -> None:
        self.client.delete_collection(self._collection_name)
        self.collection = self.client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )


# ---------------------------------------------------------------------------
# Backend: QdrantDB
# ---------------------------------------------------------------------------


class QdrantDB(VectorDatabase):
    """Qdrant vector database backend.

    Connects to a running Qdrant instance and auto-creates the collection
    when it does not already exist.

    Args:
        collection_name: Name of the Qdrant collection.
        host:            Qdrant server hostname.
        port:            Qdrant server gRPC/HTTP port.
        dimension:       Embedding vector dimension (must match your model).
    """

    def __init__(
        self,
        collection_name: str = "documents",
        host: str = "localhost",
        port: int = 6333,
        dimension: int = 1536,
    ) -> None:
        from qdrant_client import QdrantClient  # type: ignore[import]
        from qdrant_client.models import Distance, VectorParams  # type: ignore[import]

        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)

        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
            )

    # -- VectorDatabase interface --

    def insert(self, documents: list[VectorDocument]) -> int:
        from qdrant_client.models import PointStruct  # type: ignore[import]

        if not documents:
            return 0
        points = [
            PointStruct(
                id=doc.id,
                vector=doc.embedding,
                payload={"text": doc.text, **(doc.metadata or {})},
            )
            for doc in documents
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)
        return len(documents)

    def batch_insert(
        self, documents: list[VectorDocument], batch_size: int = 100
    ) -> int:
        """Uses Qdrant's native batch upsert for efficiency."""
        from qdrant_client.models import PointStruct  # type: ignore[import]

        total = 0
        for i in range(0, len(documents), batch_size):
            chunk = documents[i : i + batch_size]
            points = [
                PointStruct(
                    id=doc.id,
                    vector=doc.embedding,
                    payload={"text": doc.text, **(doc.metadata or {})},
                )
                for doc in chunk
            ]
            self.client.upsert(collection_name=self.collection_name, points=points)
            total += len(chunk)
        return total

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore[import]

        search_filter = None
        if filter_metadata:
            conditions = [
                FieldCondition(key=key, match=MatchValue(value=value))
                for key, value in filter_metadata.items()
            ]
            search_filter = Filter(must=conditions)

        hits = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=k,
            query_filter=search_filter,
        )
        return [
            SearchResult(
                id=str(hit.id),
                text=hit.payload["text"],
                score=hit.score,
                metadata={k: v for k, v in hit.payload.items() if k != "text"},
            )
            for hit in hits
        ]

    def delete(self, ids: list[str]) -> int:
        from qdrant_client.models import PointIdsList  # type: ignore[import]

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=ids),
        )
        return len(ids)

    def count(self) -> int:
        return self.client.count(collection_name=self.collection_name).count

    def clear(self) -> None:
        self.client.delete_collection(self.collection_name)


# ---------------------------------------------------------------------------
# Backend: PineconeDB
# ---------------------------------------------------------------------------


class PineconeDB(VectorDatabase):
    """Pinecone vector database backend.

    Supports both serverless and pod-based indexes.  When *serverless* is
    ``True``, the index is created in a serverless spec; otherwise a
    pod-based starter spec is used.

    Args:
        api_key:     Pinecone API key.
        index_name:  Name of the Pinecone index.
        dimension:   Embedding vector dimension.
        serverless:  Use serverless spec if ``True``.
        cloud:       Cloud provider for serverless (``"aws"`` or ``"gcp"``).
        region:      Region for serverless (e.g. ``"us-east-1"``).
    """

    def __init__(
        self,
        api_key: str,
        index_name: str = "documents",
        dimension: int = 1536,
        serverless: bool = True,
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        from pinecone import Pinecone, ServerlessSpec  # type: ignore[import]

        self.pc = Pinecone(api_key=api_key)
        existing = [idx.name for idx in self.pc.list_indexes()]

        if index_name not in existing:
            if serverless:
                spec = ServerlessSpec(cloud=cloud, region=region)
            else:
                from pinecone import PodSpec  # type: ignore[import]

                spec = PodSpec(environment="gcp-starter")

            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=spec,
            )

        self.index = self.pc.Index(index_name)

    # -- VectorDatabase interface --

    def insert(self, documents: list[VectorDocument]) -> int:
        if not documents:
            return 0
        vectors = [
            {
                "id": doc.id,
                "values": doc.embedding,
                "metadata": {"text": doc.text, **(doc.metadata or {})},
            }
            for doc in documents
        ]
        self.index.upsert(vectors=vectors)
        return len(documents)

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        results = self.index.query(
            vector=query_embedding,
            top_k=k,
            filter=filter_metadata,
            include_metadata=True,
        )
        return [
            SearchResult(
                id=match["id"],
                text=match["metadata"].get("text", ""),
                score=match["score"],
                metadata={
                    key: val
                    for key, val in match["metadata"].items()
                    if key != "text"
                },
            )
            for match in results.get("matches", [])
        ]

    def delete(self, ids: list[str]) -> int:
        self.index.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0)

    def clear(self) -> None:
        self.index.delete(delete_all=True)


# ---------------------------------------------------------------------------
# Backend: PgvectorDB
# ---------------------------------------------------------------------------


class PgvectorDB(VectorDatabase):
    """PostgreSQL + pgvector backend.

    Requires the ``pgvector`` extension to be installed in your Postgres
    instance and the ``psycopg2`` Python package.

    The table is created automatically on first use.  Metadata is stored as
    JSONB so any key-value filter can be expressed as a SQL ``->>/@@``
    expression.

    Args:
        connection_string: libpq connection string, e.g.
            ``"postgresql://user:pass@localhost:5432/mydb"``.
        table_name:        Name of the vectors table.
        dimension:         Embedding vector dimension.
    """

    _CREATE_EXTENSION = "CREATE EXTENSION IF NOT EXISTS vector"
    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS {table} (
            id          TEXT PRIMARY KEY,
            text        TEXT NOT NULL,
            embedding   vector({dim}),
            metadata    JSONB DEFAULT '{{}}'::jsonb
        )
    """
    _CREATE_INDEX = """
        CREATE INDEX IF NOT EXISTS {table}_embedding_idx
        ON {table} USING hnsw (embedding vector_cosine_ops)
    """

    def __init__(
        self,
        connection_string: str,
        table_name: str = "vector_documents",
        dimension: int = 1536,
    ) -> None:
        import psycopg2  # type: ignore[import]
        from psycopg2.extras import RealDictCursor  # type: ignore[import]

        self._conn_str = connection_string
        self.table = table_name
        self.dimension = dimension
        self._conn = psycopg2.connect(connection_string)
        self._conn.autocommit = True
        self._cursor_factory = RealDictCursor
        self._bootstrap()

    def _bootstrap(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(self._CREATE_EXTENSION)
            cur.execute(
                self._CREATE_TABLE.format(table=self.table, dim=self.dimension)
            )
            cur.execute(self._CREATE_INDEX.format(table=self.table))

    # -- VectorDatabase interface --

    def insert(self, documents: list[VectorDocument]) -> int:
        if not documents:
            return 0
        import json
        import psycopg2.extras  # type: ignore[import]

        sql = f"""
            INSERT INTO {self.table} (id, text, embedding, metadata)
            VALUES (%s, %s, %s::vector, %s::jsonb)
            ON CONFLICT (id) DO UPDATE
                SET text = EXCLUDED.text,
                    embedding = EXCLUDED.embedding,
                    metadata = EXCLUDED.metadata
        """
        with self._conn.cursor() as cur:
            psycopg2.extras.execute_batch(
                cur,
                sql,
                [
                    (
                        doc.id,
                        doc.text,
                        str(doc.embedding),
                        json.dumps(doc.metadata or {}),
                    )
                    for doc in documents
                ],
            )
        return len(documents)

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[SearchResult]:
        import json

        where_clause = ""
        params: list = [str(query_embedding), k]

        if filter_metadata:
            conditions = []
            for key, value in filter_metadata.items():
                conditions.append(f"metadata->>%s = %s")
                params.extend([key, str(value)])
            where_clause = "WHERE " + " AND ".join(conditions)

        # Insert vector param at position 0 and k at the end
        sql = f"""
            SELECT id, text,
                   1 - (embedding <=> %s::vector) AS score,
                   metadata
            FROM {self.table}
            {where_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """
        params = [str(query_embedding)]
        if filter_metadata:
            for key, value in filter_metadata.items():
                params.extend([key, str(value)])
        params.extend([str(query_embedding), k])

        with self._conn.cursor(cursor_factory=self._cursor_factory) as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

        return [
            SearchResult(
                id=row["id"],
                text=row["text"],
                score=float(row["score"]),
                metadata=row["metadata"] if isinstance(row["metadata"], dict) else {},
            )
            for row in rows
        ]

    def delete(self, ids: list[str]) -> int:
        if not ids:
            return 0
        sql = f"DELETE FROM {self.table} WHERE id = ANY(%s)"
        with self._conn.cursor() as cur:
            cur.execute(sql, (ids,))
            return cur.rowcount

    def count(self) -> int:
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table}")
            return cur.fetchone()[0]

    def clear(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {self.table}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class VectorDBFactory:
    """Create vector database instances from a type name or config dict.

    Examples::

        # By name
        db = VectorDBFactory.create("chroma", collection_name="my_docs")

        # By config dict (useful for YAML/JSON config files)
        db = VectorDBFactory.create_from_config({
            "type": "qdrant",
            "host": "localhost",
            "port": 6333,
            "collection_name": "my_docs",
            "dimension": 1536,
        })
    """

    DATABASES: dict[str, type[VectorDatabase]] = {
        "chroma": ChromaDB,
        "qdrant": QdrantDB,
        "pinecone": PineconeDB,
        "pgvector": PgvectorDB,
        "simple": SimpleVectorStore,
    }

    @classmethod
    def create(cls, db_type: str, **kwargs: object) -> VectorDatabase:
        """Instantiate a backend by name.

        Args:
            db_type: One of ``"chroma"``, ``"qdrant"``, ``"pinecone"``,
                     ``"pgvector"``, or ``"simple"``.
            **kwargs: Forwarded to the backend's ``__init__``.

        Raises:
            ValueError: If *db_type* is not recognised.
        """
        db_class = cls.DATABASES.get(db_type)
        if db_class is None:
            raise ValueError(
                f"Unknown database type: {db_type!r}. "
                f"Available: {list(cls.DATABASES)}"
            )
        return db_class(**kwargs)  # type: ignore[arg-type]

    @classmethod
    def create_from_config(cls, config: dict) -> VectorDatabase:
        """Instantiate a backend from a config dict.

        The ``"type"`` key selects the backend; all other keys are forwarded
        as keyword arguments.

        Args:
            config: Dict with a ``"type"`` key and backend-specific options.

        Example::

            config = {
                "type": "qdrant",
                "host": "localhost",
                "port": 6333,
                "collection_name": "my_docs",
            }
            db = VectorDBFactory.create_from_config(config)
        """
        config = dict(config)  # defensive copy
        db_type = config.pop("type")
        return cls.create(db_type, **config)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _make_random_embedding(dim: int = 64) -> list[float]:
    """Return a random normalised embedding for demo purposes."""
    v = np.random.randn(dim).astype(np.float64)
    v /= np.linalg.norm(v)
    return v.tolist()


def _run_demo() -> None:
    """Insert 100 documents into Chroma and Simple, run matching queries,
    compare results, and benchmark insert / search speed."""
    import os
    import shutil
    import tempfile

    DIM = 64
    N_DOCS = 100
    QUERY_IDX = 0  # We'll query for document 0

    # Generate shared documents and a known query embedding
    rng = np.random.default_rng(42)
    embeddings = [
        (rng.standard_normal(DIM) / np.linalg.norm(rng.standard_normal(DIM))).tolist()
        for _ in range(N_DOCS)
    ]
    # Make embedding reproducible
    embeddings = []
    for i in range(N_DOCS):
        v = rng.standard_normal(DIM)
        embeddings.append((v / np.linalg.norm(v)).tolist())

    docs = [
        VectorDocument(
            id=str(i),
            text=f"Document number {i} about topic {i % 5}",
            embedding=embeddings[i],
            metadata={"category": f"cat_{i % 3}", "index": i},
        )
        for i in range(N_DOCS)
    ]
    query_embedding = embeddings[QUERY_IDX]

    chroma_dir = tempfile.mkdtemp(prefix="chroma_demo_")
    try:
        backends: dict[str, VectorDatabase] = {
            "Simple": SimpleVectorStore(),
            "Chroma": ChromaDB(
                collection_name="demo", persist_directory=chroma_dir
            ),
        }

        print("=" * 60)
        print("Vector Database Abstraction Demo")
        print("=" * 60)

        for name, db in backends.items():
            # --- Insert benchmark ---
            t0 = time.perf_counter()
            inserted = db.batch_insert(docs)
            insert_ms = (time.perf_counter() - t0) * 1000

            # --- Search benchmark ---
            t0 = time.perf_counter()
            results = db.search(query_embedding, k=5)
            search_ms = (time.perf_counter() - t0) * 1000

            print(f"\n[{name}]")
            print(f"  Inserted {inserted} docs in {insert_ms:.1f} ms")
            print(f"  Top-5 search in {search_ms:.2f} ms")
            print(f"  Count: {db.count()}")
            print(f"  Top result: id={results[0].id!r}, score={results[0].score:.4f}")

            # --- Metadata filter ---
            filtered = db.search(query_embedding, k=5, filter_metadata={"category": "cat_0"})
            print(f"  Filtered (category=cat_0): {[r.id for r in filtered]}")

        # Cross-backend comparison
        print("\n--- Cross-backend result comparison (top 3 IDs) ---")
        result_sets = {}
        for name, db in backends.items():
            top3 = {r.id for r in db.search(query_embedding, k=3)}
            result_sets[name] = top3
            print(f"  {name}: {sorted(top3)}")

        names = list(result_sets)
        if len(names) >= 2:
            overlap = result_sets[names[0]] & result_sets[names[1]]
            print(f"  Overlap ({names[0]} ∩ {names[1]}): {sorted(overlap)}")

    finally:
        shutil.rmtree(chroma_dir, ignore_errors=True)


if __name__ == "__main__":
    _run_demo()
