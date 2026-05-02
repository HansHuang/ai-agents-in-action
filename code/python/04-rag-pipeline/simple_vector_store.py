"""In-memory vector store with cosine similarity search.

Suitable for prototyping and datasets up to ~10,000 documents.
For production, switch to a dedicated vector database (Qdrant, Pinecone, etc.).

See: docs/03-memory-and-retrieval/02-embeddings-and-vectors.md
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Optional

import numpy as np


class SimpleVectorStore:
    """In-memory vector store for prototyping and small datasets.

    Documents are stored as plain Python dicts and searched with brute-force
    cosine similarity — O(n) per query. This is fine for up to ~10,000
    documents. At larger scale, switch to a dedicated vector database.

    Each stored document has the shape::

        {
            "id": "<uuid>",
            "text": "<original text>",
            "embedding": [0.01, -0.23, ...],
            "metadata": {"key": "value", ...}   # optional
        }
    """

    def __init__(self) -> None:
        self._documents: list[dict] = []

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(
        self,
        text: str,
        embedding: list[float],
        metadata: Optional[dict] = None,
    ) -> str:
        """Add a single document and return its generated ID.

        Args:
            text: The document's raw text content.
            embedding: Pre-computed embedding vector for *text*.
            metadata: Optional key-value pairs for filtering (e.g.,
                ``{"category": "support", "product": "widget-pro"}``).

        Returns:
            A UUID string that uniquely identifies the stored document.
        """
        doc_id = str(uuid.uuid4())
        self._documents.append(
            {
                "id": doc_id,
                "text": text,
                "embedding": embedding,
                "metadata": metadata or {},
            }
        )
        return doc_id

    def add_batch(self, items: list[dict]) -> list[str]:
        """Add multiple documents at once.

        Args:
            items: List of dicts, each with keys ``"text"``, ``"embedding"``,
                and optionally ``"metadata"``.

        Returns:
            List of generated document IDs in the same order as *items*.
        """
        return [
            self.add(
                text=item["text"],
                embedding=item["embedding"],
                metadata=item.get("metadata"),
            )
            for item in items
        ]

    def delete(self, doc_id: str) -> bool:
        """Remove a document by its ID.

        Args:
            doc_id: The ID returned by :meth:`add` or :meth:`add_batch`.

        Returns:
            ``True`` if a document was found and removed, ``False`` otherwise.
        """
        before = len(self._documents)
        self._documents = [d for d in self._documents if d["id"] != doc_id]
        return len(self._documents) < before

    def clear(self) -> None:
        """Remove all documents from the store."""
        self._documents.clear()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[dict]:
        """Return the *k* most similar documents to *query_embedding*.

        Metadata filtering is applied **before** scoring to keep the
        full-scan cost proportional to the filtered subset, not the total
        document count.

        Args:
            query_embedding: Embedding vector for the user's query.
            k: Maximum number of results to return.
            filter_metadata: If provided, only documents whose metadata
                contains all the specified key-value pairs are considered.

        Returns:
            List of ``{"id": str, "text": str, "score": float,
            "metadata": dict}`` dicts, sorted highest-similarity first.
        """
        candidates = self._apply_filter(filter_metadata)
        if not candidates:
            return []

        q = np.array(query_embedding, dtype=np.float64)
        scored = []
        for doc in candidates:
            score = _cosine_similarity(q, np.array(doc["embedding"], dtype=np.float64))
            scored.append(
                {
                    "id": doc["id"],
                    "text": doc["text"],
                    "score": score,
                    "metadata": doc["metadata"],
                }
            )
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:k]

    def search_with_threshold(
        self,
        query_embedding: list[float],
        threshold: float = 0.7,
        k: int = 5,
    ) -> list[dict]:
        """Search and return only results at or above *threshold* similarity.

        Args:
            query_embedding: Embedding vector for the user's query.
            threshold: Minimum cosine similarity score (inclusive).
            k: Maximum number of results to return.

        Returns:
            Same format as :meth:`search`, filtered to ``score >= threshold``.
        """
        results = self.search(query_embedding, k=k)
        return [r for r in results if r["score"] >= threshold]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def count(self) -> int:
        """Return the number of documents currently stored."""
        return len(self._documents)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Persist the store to a JSON file.

        Intended for small datasets only. For large collections, use a
        dedicated vector database.

        Args:
            filepath: Destination file path. Parent directories must exist.
        """
        with open(filepath, "w", encoding="utf-8") as fh:
            json.dump(self._documents, fh)

    def load(self, filepath: str) -> None:
        """Replace the current store contents by loading from *filepath*.

        The file must have been written by :meth:`save`.

        Args:
            filepath: Path to the JSON file.
        """
        with open(filepath, "r", encoding="utf-8") as fh:
            self._documents = json.load(fh)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_filter(self, filter_metadata: Optional[dict]) -> list[dict]:
        """Return documents matching all key-value pairs in *filter_metadata*.

        A document matches if its ``metadata`` dict contains every key in
        *filter_metadata* with an equal value. Non-matching documents are
        excluded entirely.
        """
        if not filter_metadata:
            return self._documents
        return [
            doc
            for doc in self._documents
            if all(doc["metadata"].get(k) == v for k, v in filter_metadata.items())
        ]


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two numpy vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _make_random_embedding(dim: int = 8) -> list[float]:
    """Return a random unit-normalised vector for demo purposes."""
    v = np.random.randn(dim).astype(np.float64)
    v /= np.linalg.norm(v)
    return v.tolist()


def main() -> None:
    """Demonstrate add, search, metadata filtering, threshold, and persistence."""
    import tempfile

    rng = np.random.default_rng(42)

    store = SimpleVectorStore()

    # --- Add 20 documents across two categories ---
    support_docs = [
        "How do I return a damaged item?",
        "What is the refund timeline?",
        "Can I exchange a product for a different size?",
        "How long does shipping take after an order is placed?",
        "Where do I track my order status?",
        "Is there a restocking fee for returns?",
        "How do I report a missing package?",
        "Can I cancel my order after it ships?",
        "What payment methods are accepted?",
        "How do I apply a coupon code at checkout?",
    ]
    marketing_docs = [
        "Introducing our new summer collection.",
        "Get 20% off your first order with code WELCOME20.",
        "Shop our top-rated products of the year.",
        "Free shipping on orders over $50.",
        "New arrivals every Monday — don't miss out.",
        "Subscribe to our newsletter for exclusive deals.",
        "Gift cards available in any denomination.",
        "Follow us on social media for style inspiration.",
        "Our loyalty program earns you points on every purchase.",
        "Refer a friend and both of you get $10 off.",
    ]

    dim = 16  # Small dimension for demo; real embeddings are 512–3072

    # Simulate embeddings: support docs cluster in one region, marketing in another.
    support_center = rng.standard_normal(dim)
    support_center /= np.linalg.norm(support_center)
    marketing_center = -support_center  # Opposite direction

    def _sim_embedding(center: np.ndarray) -> list[float]:
        noise = rng.standard_normal(dim) * 0.15
        v = center + noise
        v /= np.linalg.norm(v)
        return v.tolist()

    ids_support = []
    for text in support_docs:
        doc_id = store.add(
            text, _sim_embedding(support_center), metadata={"category": "support"}
        )
        ids_support.append(doc_id)

    for text in marketing_docs:
        store.add(
            text, _sim_embedding(marketing_center), metadata={"category": "marketing"}
        )

    print(f"Store contains {store.count()} documents.")

    # --- Basic search ---
    print("\n=== Search: query near 'support' cluster ===")
    query_emb = _sim_embedding(support_center)
    results = store.search(query_emb, k=5)
    for r in results:
        print(f"  score={r['score']:.3f}  [{r['metadata']['category']}]  {r['text']}")

    # --- Metadata-filtered search ---
    print("\n=== Metadata-filtered search (category=support only) ===")
    results_filtered = store.search(query_emb, k=5, filter_metadata={"category": "support"})
    for r in results_filtered:
        assert r["metadata"]["category"] == "support"
        print(f"  score={r['score']:.3f}  {r['text']}")

    # --- Threshold search ---
    print("\n=== Threshold search (>= 0.90) ===")
    results_thresh = store.search_with_threshold(query_emb, threshold=0.90, k=10)
    print(f"  {len(results_thresh)} results above threshold.")
    for r in results_thresh:
        print(f"  score={r['score']:.3f}  {r['text']}")

    # --- Delete ---
    removed = store.delete(ids_support[0])
    print(f"\nDeleted first support doc: {removed}. Store now has {store.count()} docs.")

    # --- Save / load roundtrip ---
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    store.save(tmp_path)
    size_kb = os.path.getsize(tmp_path) / 1024
    print(f"\nSaved store to {tmp_path} ({size_kb:.1f} KB).")

    store2 = SimpleVectorStore()
    store2.load(tmp_path)
    print(f"Loaded store has {store2.count()} documents.")
    results2 = store2.search(query_emb, k=3)
    print("  Top-3 after reload:")
    for r in results2:
        print(f"    score={r['score']:.3f}  {r['text']}")

    os.unlink(tmp_path)


if __name__ == "__main__":
    main()
