"""Incremental knowledge base management for evolving document collections.

Handles document additions, updates, and deletions without rebuilding the
entire index.  Uses content hashing to detect changes and avoid redundant
re-embedding.

See: docs/03-memory-and-retrieval/03-rag-from-scratch.md
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from rag_pipeline import RAGPipeline
from simple_vector_store import SimpleVectorStore

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".rst", ".text"}


# ---------------------------------------------------------------------------
# DocumentRecord
# ---------------------------------------------------------------------------


@dataclass
class DocumentRecord:
    """Metadata about a single document tracked by the knowledge base.

    Attributes:
        source_id:    Unique identifier (typically the file name).
        chunk_ids:    IDs of all vector-store chunks for this document.
        chunk_count:  Number of chunks currently stored.
        content_hash: SHA-256 hex digest of the document text at last ingest.
        last_updated: Unix timestamp of the last ingest or update.
    """

    source_id: str
    chunk_ids: list[str]
    chunk_count: int
    content_hash: str
    last_updated: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# KnowledgeBaseManager
# ---------------------------------------------------------------------------


class KnowledgeBaseManager:
    """Manage a knowledge base that changes over time.

    Wraps a :class:`RAGPipeline` and :class:`SimpleVectorStore` with an
    index layer that enables incremental document lifecycle operations:
    add, update, remove, and directory sync.

    Args:
        rag_pipeline: An initialised :class:`RAGPipeline`.
        vector_store: The same :class:`SimpleVectorStore` used by the pipeline.
    """

    def __init__(
        self,
        rag_pipeline: RAGPipeline,
        vector_store: SimpleVectorStore,
    ) -> None:
        self.pipeline = rag_pipeline
        self.vector_store = vector_store
        self.document_index: dict[str, DocumentRecord] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _chunk_ids_for_source(self, source_id: str) -> list[str]:
        """Return current vector-store IDs whose metadata source matches."""
        return [
            doc["id"]
            for doc in self.vector_store._documents
            if doc.get("metadata", {}).get("source") == source_id
        ]

    def _ingest_and_record(self, source_id: str, text: str) -> int:
        """Ingest text and register chunks in the document index."""
        n = self.pipeline.ingest_text(text, metadata={"source": source_id})
        chunk_ids = self._chunk_ids_for_source(source_id)
        self.document_index[source_id] = DocumentRecord(
            source_id=source_id,
            chunk_ids=chunk_ids,
            chunk_count=n,
            content_hash=self._hash(text),
            last_updated=time.time(),
        )
        return n

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(self, source_id: str, text: str) -> int:
        """Add a new document to the knowledge base.

        If a document with *source_id* already exists it is **not** replaced —
        use :meth:`update_document` instead.

        Args:
            source_id: Unique identifier for the document.
            text:      Raw document text.

        Returns:
            Number of chunks created.

        Raises:
            ValueError: If *source_id* is already tracked.
        """
        if source_id in self.document_index:
            raise ValueError(
                f"Document {source_id!r} already exists. Use update_document() to replace it."
            )
        return self._ingest_and_record(source_id, text)

    def update_document(self, source_id: str, new_text: str) -> dict:
        """Update an existing document with new content.

        Removes all old chunks, ingests the new content, and updates the
        document index.

        Args:
            source_id: The document to update.
            new_text:  Replacement text.

        Returns:
            ``{"removed": int, "added": int, "net_change": int}``

        Raises:
            KeyError: If *source_id* is not in the index.
        """
        if source_id not in self.document_index:
            raise KeyError(f"Document {source_id!r} not found. Use add_document() first.")

        old_hash = self.document_index[source_id].content_hash
        new_hash = self._hash(new_text)
        if old_hash == new_hash:
            existing = self.document_index[source_id]
            return {"removed": 0, "added": 0, "net_change": 0, "unchanged": True}

        # Remove old chunks
        old_record = self.document_index[source_id]
        removed = self.pipeline.remove_document(source_id)

        # Re-ingest
        added = self._ingest_and_record(source_id, new_text)

        return {
            "removed": removed,
            "added": added,
            "net_change": added - removed,
            "unchanged": False,
        }

    def remove_document(self, source_id: str) -> int:
        """Remove all chunks for a document and deregister it.

        Args:
            source_id: The document to remove.

        Returns:
            Number of chunks removed.

        Raises:
            KeyError: If *source_id* is not in the index.
        """
        if source_id not in self.document_index:
            raise KeyError(f"Document {source_id!r} not found in index.")

        removed = self.pipeline.remove_document(source_id)
        del self.document_index[source_id]
        return removed

    def sync_directory(self, directory: str) -> dict:
        """Sync the knowledge base with the current state of *directory*.

        - Files present in the directory but not in the index → added.
        - Files present in both but with changed content → updated.
        - Files in the index but no longer in the directory → removed.
        - Files in both with unchanged content → skipped.

        Args:
            directory: Path to directory containing text files.

        Returns:
            ``{"added": list[str], "updated": list[str], "removed": list[str], "errors": list[str]}``
        """
        dir_path = Path(directory)
        if not dir_path.is_dir():
            raise ValueError(f"Not a directory: {directory!r}")

        added: list[str] = []
        updated: list[str] = []
        removed: list[str] = []
        errors: list[str] = []

        # Collect current files
        disk_files: dict[str, str] = {}  # source_id → text
        for file_path in sorted(dir_path.iterdir()):
            if file_path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue
            try:
                disk_files[file_path.name] = file_path.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{file_path.name}: {exc}")

        # Add / update
        for source_id, text in disk_files.items():
            try:
                if source_id not in self.document_index:
                    self._ingest_and_record(source_id, text)
                    added.append(source_id)
                else:
                    result = self.update_document(source_id, text)
                    if not result.get("unchanged"):
                        updated.append(source_id)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{source_id}: {exc}")

        # Remove documents no longer on disk
        for source_id in list(self.document_index.keys()):
            if source_id not in disk_files:
                try:
                    self.remove_document(source_id)
                    removed.append(source_id)
                except Exception as exc:  # noqa: BLE001
                    errors.append(f"{source_id}: {exc}")

        return {
            "added": added,
            "updated": updated,
            "removed": removed,
            "errors": errors,
        }

    def get_stats(self) -> dict:
        """Return knowledge base statistics."""
        return {
            "total_documents": len(self.document_index),
            "total_chunks": self.vector_store.count(),
            "documents": [
                {
                    "id": record.source_id,
                    "chunks": record.chunk_count,
                    "last_updated": record.last_updated,
                }
                for record in self.document_index.values()
            ],
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _run_demo() -> None:
    import tempfile
    import os

    from embedding_generator import EmbeddingGenerator

    print("=" * 70)
    print("KNOWLEDGE BASE MANAGER DEMO")
    print("=" * 70)

    embedder = EmbeddingGenerator(model="text-embedding-3-small")
    vector_store = SimpleVectorStore()
    pipeline = RAGPipeline(
        vector_store=vector_store,
        embedder=embedder,
        chunk_size=200,
        overlap=30,
        similarity_threshold=0.4,
    )
    manager = KnowledgeBaseManager(rag_pipeline=pipeline, vector_store=vector_store)

    # --- Initial state --------------------------------------------------------
    print("\n--- Step 1: Add 3 documents ---")
    docs = {
        "policy-v1.md": "Return policy: 30 days return window. No restocking fee.",
        "shipping.md": "Standard shipping: 3-5 days at $4.99.",
        "faq.md": "We accept Visa, Mastercard, and PayPal.",
    }
    for source_id, text in docs.items():
        n = manager.add_document(source_id, text)
        print(f"  Added {source_id}: {n} chunks")

    stats = manager.get_stats()
    print(f"\nStats: {stats['total_documents']} docs, {stats['total_chunks']} chunks")

    # --- Update a document ---------------------------------------------------
    print("\n--- Step 2: Update policy-v1.md ---")
    result = manager.update_document(
        "policy-v1.md",
        "Return policy: 60 days return window. Free returns on all orders. No restocking fee.",
    )
    print(f"  Update result: {result}")

    # Query should reflect new content
    response = pipeline.query("How long is the return window?", threshold=0.4)
    print(f"  Query 'return window': {response.answer[:100]}")
    print(f"  Sources: {response.sources}")

    # --- Sync with a directory -----------------------------------------------
    print("\n--- Step 3: Directory sync ---")
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write files: add two new, keep shipping.md (unchanged), remove faq.md
        Path(tmpdir, "policy-v1.md").write_text(
            "Return policy: 60 days return window. Free returns on all orders. No restocking fee."
        )
        Path(tmpdir, "shipping.md").write_text(
            "Standard shipping: 3-5 days at $4.99."
        )
        Path(tmpdir, "products.md").write_text(
            "WidgetPro 3000: $199.99. WidgetLite: $49.99."
        )
        Path(tmpdir, "hr-policy.md").write_text(
            "Employees accrue 15 vacation days per year."
        )
        # faq.md is intentionally absent → should be removed

        sync_result = manager.sync_directory(tmpdir)
        print(f"  Added:   {sync_result['added']}")
        print(f"  Updated: {sync_result['updated']}")
        print(f"  Removed: {sync_result['removed']}")
        print(f"  Errors:  {sync_result['errors']}")

    stats = manager.get_stats()
    print(f"\nFinal stats: {stats['total_documents']} docs, {stats['total_chunks']} chunks")
    for doc in stats["documents"]:
        print(f"  {doc['id']}: {doc['chunks']} chunks")

    # Verify query works after sync
    print("\n--- Step 4: Query after sync ---")
    for question in [
        "What is the return window?",
        "How many vacation days do employees get?",
        "How much does the WidgetPro 3000 cost?",
    ]:
        resp = pipeline.query(question, threshold=0.4)
        print(f"\n  Q: {question}")
        print(f"  A: {resp.answer[:120]}")
        print(f"  Sources: {resp.sources}")


if __name__ == "__main__":
    _run_demo()
