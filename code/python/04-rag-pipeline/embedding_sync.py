"""Embedding sync manager: keep a vector index in sync with source documents.

The vector database is a *derived index*, not the source of truth.  Documents
change, get deleted, and new ones are added.  When the source document store
drifts from the vector index, search quality degrades silently.

This module provides:

* :class:`EmbeddingSyncManager` — orchestrates full and incremental syncs.
* :class:`InMemoryDocumentStore` — a lightweight document store for demos and
  tests; swap it for your real document storage.
* :class:`SyncReport` / :class:`SyncHealth` — structured results.

Usage::

    from embedding_sync import EmbeddingSyncManager, InMemoryDocumentStore
    from vector_database import SimpleVectorStore

    store = InMemoryDocumentStore()
    db = SimpleVectorStore()
    embedder = lambda text: [0.1] * 64  # replace with real embedder

    manager = EmbeddingSyncManager(db, embedder, store)
    report = manager.full_sync()
    print(report)

    health = manager.verify_sync()
    print(health)

See: docs/05-the-tool-ecosystem/02-vector-databases.md
"""

from __future__ import annotations

import hashlib
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional

from vector_database import VectorDatabase, VectorDocument


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SourceDocument:
    """A document in the source document store."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    updated_at: float = field(default_factory=time.time)

    def content_hash(self) -> str:
        """SHA-256 of the document text — used to detect changes."""
        return hashlib.sha256(self.text.encode()).hexdigest()


@dataclass
class SyncReport:
    """Result of a full or incremental sync operation."""

    strategy: str          # "full" or "incremental"
    created: int = 0
    updated: int = 0
    deleted: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __str__(self) -> str:
        status = "OK" if not self.errors else f"{len(self.errors)} errors"
        return (
            f"SyncReport[{self.strategy}] "
            f"created={self.created} updated={self.updated} deleted={self.deleted} "
            f"duration={self.duration_seconds:.2f}s status={status} "
            f"at={self.timestamp.isoformat()}"
        )


@dataclass
class SyncHealth:
    """Result of a sync health verification."""

    is_healthy: bool
    total_documents: int      # Documents in the source store
    total_vectors: int        # Vectors in the vector DB
    mismatch_count: int       # Documents in store but not in vector DB
    orphan_count: int         # Vectors in DB with no corresponding document
    recommendations: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "HEALTHY" if self.is_healthy else "DEGRADED"
        return (
            f"SyncHealth[{status}] "
            f"docs={self.total_documents} vectors={self.total_vectors} "
            f"missing={self.mismatch_count} orphans={self.orphan_count} "
            f"recommendations={self.recommendations}"
        )


# ---------------------------------------------------------------------------
# Simple in-memory document store (for demos and tests)
# ---------------------------------------------------------------------------


class InMemoryDocumentStore:
    """Minimal in-memory document store.

    Tracks creation/update timestamps so incremental sync can query for
    documents changed after a given timestamp.
    """

    def __init__(self) -> None:
        self._docs: dict[str, SourceDocument] = {}

    # -- Mutation --

    def add(self, doc: SourceDocument) -> None:
        """Add or replace a document."""
        self._docs[doc.id] = doc

    def update(self, doc_id: str, new_text: str, metadata: Optional[dict] = None) -> None:
        """Update the text (and optionally metadata) of an existing document."""
        existing = self._docs.get(doc_id)
        if existing is None:
            raise KeyError(f"Document {doc_id!r} not found")
        self._docs[doc_id] = SourceDocument(
            id=doc_id,
            text=new_text,
            metadata=metadata if metadata is not None else existing.metadata,
            updated_at=time.time(),
        )

    def remove(self, doc_id: str) -> None:
        """Remove a document by ID."""
        self._docs.pop(doc_id, None)

    def add_many(self, docs: list[SourceDocument]) -> None:
        for doc in docs:
            self.add(doc)

    # -- Query --

    def get_all(self) -> list[SourceDocument]:
        return list(self._docs.values())

    def get_ids(self) -> set[str]:
        return set(self._docs)

    def get_changes(
        self, since: Optional[float] = None
    ) -> dict[str, list]:
        """Return documents created or updated after *since* (epoch seconds).

        Returns a dict with keys ``"created_or_updated"`` and ``"all_ids"``
        (all current document IDs, used to find deleted vectors).
        """
        if since is None:
            return {"created_or_updated": self.get_all(), "all_ids": self.get_ids()}
        changed = [d for d in self._docs.values() if d.updated_at > since]
        return {"created_or_updated": changed, "all_ids": self.get_ids()}

    def count(self) -> int:
        return len(self._docs)


# ---------------------------------------------------------------------------
# EmbeddingSyncManager
# ---------------------------------------------------------------------------

# Embedder type: any callable that takes a string and returns a list of floats
Embedder = Callable[[str], list[float]]


class EmbeddingSyncManager:
    """Synchronise a vector database index with a source document store.

    The manager supports two sync strategies:

    * **Full sync** — clear the vector index and re-embed every document from
      scratch.  Use this for initial population, after changing the embedding
      model, or after a major schema change.
    * **Incremental sync** — only process documents added, updated, or deleted
      since the last sync timestamp.  Suitable for regular background jobs.

    A built-in scheduler can run either strategy on a recurring timer.

    Args:
        vector_db:      The vector database to keep in sync.
        embedder:       Callable that converts text to an embedding vector.
        document_store: The authoritative source of documents.
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        embedder: Embedder,
        document_store: InMemoryDocumentStore,
    ) -> None:
        self.vector_db = vector_db
        self.embedder = embedder
        self.documents = document_store

        self._last_sync_ts: Optional[float] = None
        self._scheduler_thread: Optional[threading.Thread] = None
        self._scheduler_stop = threading.Event()

    # ------------------------------------------------------------------
    # Full sync
    # ------------------------------------------------------------------

    def full_sync(self) -> SyncReport:
        """Re-embed all documents and rebuild the entire vector index.

        Existing vectors are deleted before insertion.

        Returns:
            A :class:`SyncReport` describing what happened.
        """
        t0 = time.perf_counter()
        errors: list[str] = []
        created = 0

        self.vector_db.clear()

        for doc in self.documents.get_all():
            try:
                vdoc = self._to_vector_document(doc)
                self.vector_db.insert([vdoc])
                created += 1
            except Exception as exc:
                errors.append(f"[{doc.id}] {exc}")

        self._last_sync_ts = time.time()
        return SyncReport(
            strategy="full",
            created=created,
            updated=0,
            deleted=0,
            errors=errors,
            duration_seconds=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # Incremental sync
    # ------------------------------------------------------------------

    def incremental_sync(self, since: Optional[float] = None) -> SyncReport:
        """Only sync documents that have changed since *since*.

        Args:
            since: Epoch timestamp.  Defaults to the timestamp of the last
                   sync (or the beginning of time if no sync has run yet).

        Returns:
            A :class:`SyncReport` describing what happened.
        """
        since = since if since is not None else self._last_sync_ts
        t0 = time.perf_counter()
        errors: list[str] = []
        created = updated = deleted = 0

        changes = self.documents.get_changes(since=since)
        current_ids = changes["all_ids"]

        # --- Handle deletions: vectors with no matching source document ---
        # We detect deletions by querying the vector DB for IDs that no longer
        # appear in the document store.  SimpleVectorStore exposes _docs;
        # for production backends you would maintain a separate ID set.
        vector_ids = self._get_vector_ids()
        deleted_ids = vector_ids - current_ids
        if deleted_ids:
            try:
                self.vector_db.delete(list(deleted_ids))
                deleted = len(deleted_ids)
            except Exception as exc:
                errors.append(f"delete batch: {exc}")

        # --- Handle additions and updates ---
        for doc in changes["created_or_updated"]:
            try:
                vdoc = self._to_vector_document(doc)
                # Upsert: delete old embedding then insert fresh one
                if doc.id in vector_ids:
                    self.vector_db.delete([doc.id])
                    self.vector_db.insert([vdoc])
                    updated += 1
                else:
                    self.vector_db.insert([vdoc])
                    created += 1
            except Exception as exc:
                errors.append(f"[{doc.id}] {exc}")

        self._last_sync_ts = time.time()
        return SyncReport(
            strategy="incremental",
            created=created,
            updated=updated,
            deleted=deleted,
            errors=errors,
            duration_seconds=time.perf_counter() - t0,
        )

    # ------------------------------------------------------------------
    # Verify
    # ------------------------------------------------------------------

    def verify_sync(self, sample_size: int = 100) -> SyncHealth:
        """Check whether the vector index is consistent with the document store.

        Randomly samples up to *sample_size* documents, checks that each has
        a corresponding vector, and checks for orphaned vectors.

        Args:
            sample_size: Maximum number of documents to sample.

        Returns:
            A :class:`SyncHealth` report.
        """
        all_docs = self.documents.get_all()
        total_docs = len(all_docs)
        total_vectors = self.vector_db.count()

        sample = random.sample(all_docs, min(sample_size, total_docs))
        vector_ids = self._get_vector_ids()

        mismatch_count = 0
        for doc in sample:
            if doc.id not in vector_ids:
                mismatch_count += 1

        # Orphans: in vector DB but not in store
        store_ids = self.documents.get_ids()
        orphan_count = len(vector_ids - store_ids)

        recommendations: list[str] = []
        if mismatch_count > 0:
            recommendations.append(
                f"{mismatch_count} sampled documents missing from vector DB — "
                "run incremental_sync() or full_sync()."
            )
        if orphan_count > 0:
            recommendations.append(
                f"{orphan_count} orphaned vectors found — "
                "run incremental_sync() to clean up."
            )
        if total_docs != total_vectors:
            recommendations.append(
                f"Document count ({total_docs}) ≠ vector count ({total_vectors}) — "
                "index may be stale."
            )

        is_healthy = mismatch_count == 0 and orphan_count == 0 and total_docs == total_vectors
        return SyncHealth(
            is_healthy=is_healthy,
            total_documents=total_docs,
            total_vectors=total_vectors,
            mismatch_count=mismatch_count,
            orphan_count=orphan_count,
            recommendations=recommendations,
        )

    # ------------------------------------------------------------------
    # Scheduler
    # ------------------------------------------------------------------

    def schedule_sync(
        self,
        strategy: str = "incremental",
        interval_minutes: int = 15,
    ) -> None:
        """Start a background thread that runs syncs on a timer.

        Strategies:

        * ``"incremental"`` — runs every *interval_minutes*.
        * ``"full"`` — runs every *interval_minutes*.
        * ``"verify"`` — runs :meth:`verify_sync` every *interval_minutes*
          and prints the health report (does not modify the index).

        Call :meth:`stop_scheduler` to cancel the background thread.

        Args:
            strategy:         ``"incremental"``, ``"full"``, or ``"verify"``.
            interval_minutes: Seconds between runs (reuses the name for clarity).
        """
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            raise RuntimeError("A scheduler is already running.  Call stop_scheduler() first.")

        self._scheduler_stop.clear()

        def _loop() -> None:
            while not self._scheduler_stop.wait(timeout=interval_minutes * 60):
                if strategy == "full":
                    report = self.full_sync()
                    print(f"[Scheduler] {report}")
                elif strategy == "verify":
                    health = self.verify_sync()
                    print(f"[Scheduler] {health}")
                else:
                    report = self.incremental_sync()
                    print(f"[Scheduler] {report}")

        self._scheduler_thread = threading.Thread(target=_loop, daemon=True)
        self._scheduler_thread.start()

    def stop_scheduler(self) -> None:
        """Stop the background sync scheduler."""
        self._scheduler_stop.set()
        if self._scheduler_thread:
            self._scheduler_thread.join(timeout=5)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_vector_document(self, doc: SourceDocument) -> VectorDocument:
        embedding = self.embedder(doc.text)
        return VectorDocument(
            id=doc.id,
            text=doc.text,
            embedding=embedding,
            metadata={
                **doc.metadata,
                "_content_hash": doc.content_hash(),
                "_updated_at": doc.updated_at,
            },
        )

    def _get_vector_ids(self) -> set[str]:
        """Return the set of IDs currently stored in the vector database.

        For SimpleVectorStore we access ``_docs`` directly.  For production
        backends you would maintain a side-channel ID set or use the
        database's listing API.
        """
        # SimpleVectorStore exposes _docs
        if hasattr(self.vector_db, "_docs"):
            return {d["id"] for d in self.vector_db._docs}
        # Generic fallback: we can't cheaply enumerate all IDs
        # Production implementations should override this or use a metadata store.
        return set()


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _run_demo() -> None:
    """Create 50 documents, run full sync, apply changes, then incremental sync
    and verify health.  Also demonstrates a simulated embedding model change."""
    import numpy as np
    from vector_database import SimpleVectorStore

    rng = np.random.default_rng(7)
    DIM = 64

    def make_embedder(seed: int) -> Embedder:
        """Return a deterministic embedder seeded to *seed*."""
        r = np.random.default_rng(seed)

        def _embed(text: str) -> list[float]:
            # Deterministic: hash text to a seed offset, then draw from rng
            offset = int(hashlib.md5(text.encode()).hexdigest(), 16) % 1_000_000
            local = np.random.default_rng(seed + offset)
            v = local.standard_normal(DIM)
            return (v / np.linalg.norm(v)).tolist()

        return _embed

    embedder_v1 = make_embedder(42)

    # --- Build document store ---
    store = InMemoryDocumentStore()
    for i in range(50):
        store.add(
            SourceDocument(
                id=f"doc_{i:03d}",
                text=f"Document {i}: information about topic {i % 10}.",
                metadata={"category": f"cat_{i % 5}"},
            )
        )

    db = SimpleVectorStore()
    manager = EmbeddingSyncManager(db, embedder_v1, store)

    print("=" * 60)
    print("Embedding Sync Demo")
    print("=" * 60)

    # --- Full sync ---
    report = manager.full_sync()
    print(f"\n[1] Full sync:        {report}")
    health = manager.verify_sync(sample_size=50)
    print(f"    Health check:     {health}")

    # --- Modify documents ---
    time.sleep(0.01)  # ensure updated_at is strictly later
    for i in range(5):
        store.update(f"doc_{i:03d}", f"UPDATED — Document {i} revised content.")
    for i in range(45, 48):
        store.remove(f"doc_{i:03d}")
    for i in range(50, 52):
        store.add(
            SourceDocument(
                id=f"doc_{i:03d}",
                text=f"New document {i} added after initial sync.",
            )
        )

    # --- Incremental sync ---
    report2 = manager.incremental_sync()
    print(f"\n[2] Incremental sync: {report2}")
    health2 = manager.verify_sync(sample_size=50)
    print(f"    Health check:     {health2}")

    # --- Simulate embedding model change ---
    print("\n[3] Simulating embedding model change (v1 → v2) ...")
    embedder_v2 = make_embedder(99)  # different seed = different vector space
    manager.embedder = embedder_v2
    report3 = manager.full_sync()
    print(f"    Full re-sync:     {report3}")
    health3 = manager.verify_sync(sample_size=50)
    print(f"    Health check:     {health3}")


if __name__ == "__main__":
    _run_demo()
