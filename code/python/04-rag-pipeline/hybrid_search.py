"""Hybrid search: combine vector similarity with keyword (BM25) ranking.

Pure vector search fails when exact terms matter.  Searching for "Error 503"
may surface semantically similar documents that never mention "503".  Hybrid
search blends the ranked lists from both a vector index and a keyword index so
that exact keyword matches are rewarded without sacrificing semantic coverage.

Two fusion strategies are provided:

* **Weighted sum** — normalise each ranked list to [0, 1] then blend them
  with configurable weights (default 70 % vector / 30 % keyword).
* **Reciprocal Rank Fusion (RRF)** — position-based scoring that is robust to
  score scale differences and typically outperforms weighted sum on diverse
  queries.

Usage::

    from vector_database import SimpleVectorStore, VectorDocument
    from hybrid_search import HybridSearch

    db = SimpleVectorStore()
    # ... insert documents ...
    hs = HybridSearch(db, vector_weight=0.7)
    hs.build_keyword_index(docs)

    results = hs.search("Error 503", query_embedding=..., k=5)
    for r in results:
        print(hs.explain_result("Error 503", r))

Dependencies:
    rank_bm25 >= 0.2.2   (``pip install rank-bm25``)

See: docs/05-the-tool-ecosystem/02-vector-databases.md
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from vector_database import VectorDatabase, VectorDocument


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class HybridSearchResult:
    """A single result from hybrid search, carrying per-source scores."""

    id: str
    text: str
    combined_score: float
    vector_score: float
    keyword_score: float
    vector_rank: int = 0    # 1-based rank in the vector list
    keyword_rank: int = 0   # 1-based rank in the keyword list
    metadata: Optional[dict] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# HybridSearch
# ---------------------------------------------------------------------------


class HybridSearch:
    """Combine vector search and BM25 keyword search.

    Args:
        vector_db:     A :class:`~vector_database.VectorDatabase` already
                       populated with documents.
        vector_weight: Weight for the vector similarity score (0–1).
                       The keyword weight is ``1 - vector_weight``.
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        vector_weight: float = 0.7,
    ) -> None:
        if not 0.0 <= vector_weight <= 1.0:
            raise ValueError("vector_weight must be in [0, 1]")
        self.vector_db = vector_db
        self.vector_weight = vector_weight
        self.keyword_weight = 1.0 - vector_weight

        # Set by build_keyword_index
        self._bm25 = None
        self._indexed_docs: list[VectorDocument] = []
        self._tokenized_corpus: list[list[str]] = []

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase, split on non-alphanumeric, drop empty tokens."""
        return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]

    def build_keyword_index(self, documents: list[VectorDocument]) -> None:
        """Build a BM25 index over *documents*.

        Must be called before :meth:`search` or :meth:`search_with_rrf`.

        Args:
            documents: The same documents that are stored in *vector_db*.
                       Order must be stable; the BM25 index maps positional
                       indices to these documents.
        """
        from rank_bm25 import BM25Okapi  # type: ignore[import]

        self._indexed_docs = list(documents)
        self._tokenized_corpus = [self._tokenize(doc.text) for doc in documents]
        self._bm25 = BM25Okapi(self._tokenized_corpus)

    def _require_index(self) -> None:
        if self._bm25 is None:
            raise RuntimeError("Call build_keyword_index() before searching.")

    # ------------------------------------------------------------------
    # Keyword search helpers
    # ------------------------------------------------------------------

    def _keyword_search(
        self, query: str, k: int
    ) -> list[tuple[VectorDocument, float]]:
        """Return ``(document, bm25_score)`` pairs sorted descending."""
        self._require_index()
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)  # type: ignore[union-attr]
        # Pair with documents and sort
        paired = sorted(
            zip(self._indexed_docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return paired[:k]

    # ------------------------------------------------------------------
    # Score normalisation
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(scores: list[float]) -> list[float]:
        """Min-max normalise *scores* to [0, 1].  Returns zeros if flat."""
        if not scores:
            return []
        lo, hi = min(scores), max(scores)
        if hi == lo:
            return [0.0] * len(scores)
        return [(s - lo) / (hi - lo) for s in scores]

    # ------------------------------------------------------------------
    # Weighted-sum hybrid search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        query_embedding: list[float],
        k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> list[HybridSearchResult]:
        """Hybrid search using a weighted sum of normalised scores.

        Steps:

        1. Retrieve ``k * 2`` candidates from the vector index.
        2. Retrieve ``k * 2`` candidates from the BM25 index.
        3. Normalise both score lists to [0, 1].
        4. Combine: ``score = vector_weight * v_score + keyword_weight * k_score``.
        5. Return top *k* by combined score.

        Args:
            query:            Raw query string (used for BM25 scoring).
            query_embedding:  Embedded query vector (used for vector search).
            k:                Number of results to return.
            filter_metadata:  Optional metadata filter forwarded to the
                              vector database.

        Returns:
            List of :class:`HybridSearchResult` sorted by ``combined_score``.
        """
        self._require_index()
        fetch_k = k * 2

        # --- Vector results ---
        vector_results = self.vector_db.search(
            query_embedding, k=fetch_k, filter_metadata=filter_metadata
        )
        v_scores = self._normalize([r.score for r in vector_results])
        v_map: dict[str, tuple[float, int]] = {}  # id -> (norm_score, rank)
        for rank, (result, norm) in enumerate(zip(vector_results, v_scores), start=1):
            v_map[result.id] = (norm, rank)

        # --- Keyword results ---
        keyword_raw = self._keyword_search(query, k=fetch_k)
        k_scores = self._normalize([s for _, s in keyword_raw])
        k_map: dict[str, tuple[float, int]] = {}
        for rank, ((doc, _), norm) in enumerate(zip(keyword_raw, k_scores), start=1):
            k_map[doc.id] = (norm, rank)

        # --- Merge ---
        all_ids = {r.id for r in vector_results} | {doc.id for doc, _ in keyword_raw}
        merged: dict[str, HybridSearchResult] = {}

        for doc_id in all_ids:
            v_norm, v_rank = v_map.get(doc_id, (0.0, 0))
            k_norm, k_rank = k_map.get(doc_id, (0.0, 0))
            combined = self.vector_weight * v_norm + self.keyword_weight * k_norm

            # Look up text + metadata from vector results first, then index
            text, meta = self._lookup_text_meta(doc_id, vector_results)
            merged[doc_id] = HybridSearchResult(
                id=doc_id,
                text=text,
                combined_score=combined,
                vector_score=v_norm,
                keyword_score=k_norm,
                vector_rank=v_rank,
                keyword_rank=k_rank,
                metadata=meta,
            )

        sorted_results = sorted(
            merged.values(), key=lambda r: r.combined_score, reverse=True
        )
        return sorted_results[:k]

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def search_with_rrf(
        self,
        query: str,
        query_embedding: list[float],
        k: int = 5,
        rrf_k: int = 60,
        filter_metadata: Optional[dict] = None,
    ) -> list[HybridSearchResult]:
        """Hybrid search using Reciprocal Rank Fusion.

        RRF is a parameter-free fusion method that scores each document by its
        position across ranked lists::

            RRF_score(d) = Σ  1 / (rrf_k + rank_i(d))

        This is more robust than weighted-sum normalisation when the raw score
        distributions differ significantly between the two rankers.

        Args:
            query:            Raw query string.
            query_embedding:  Embedded query vector.
            k:                Number of results to return.
            rrf_k:            Ranking constant (default 60 per the original paper).
            filter_metadata:  Optional metadata filter for vector search.

        Returns:
            List of :class:`HybridSearchResult` sorted by ``combined_score``
            (which holds the RRF score here).
        """
        self._require_index()
        fetch_k = k * 2

        # --- Vector ranking ---
        vector_results = self.vector_db.search(
            query_embedding, k=fetch_k, filter_metadata=filter_metadata
        )
        v_rank: dict[str, int] = {r.id: rank for rank, r in enumerate(vector_results, start=1)}
        v_score_raw: dict[str, float] = {r.id: r.score for r in vector_results}

        # --- Keyword ranking ---
        keyword_raw = self._keyword_search(query, k=fetch_k)
        k_rank: dict[str, int] = {doc.id: rank for rank, (doc, _) in enumerate(keyword_raw, start=1)}
        k_score_raw: dict[str, float] = {doc.id: score for doc, score in keyword_raw}

        # Normalise raw scores for reporting (not used in RRF computation)
        v_norms = self._normalize(list(v_score_raw.values()))
        k_norms = self._normalize(list(k_score_raw.values()))
        v_norm_map = dict(zip(v_score_raw.keys(), v_norms))
        k_norm_map = dict(zip(k_score_raw.keys(), k_norms))

        # --- RRF fusion ---
        all_ids = set(v_rank) | set(k_rank)
        results: list[HybridSearchResult] = []

        for doc_id in all_ids:
            rrf = 0.0
            if doc_id in v_rank:
                rrf += 1.0 / (rrf_k + v_rank[doc_id])
            if doc_id in k_rank:
                rrf += 1.0 / (rrf_k + k_rank[doc_id])

            text, meta = self._lookup_text_meta(doc_id, vector_results)
            results.append(
                HybridSearchResult(
                    id=doc_id,
                    text=text,
                    combined_score=rrf,
                    vector_score=v_norm_map.get(doc_id, 0.0),
                    keyword_score=k_norm_map.get(doc_id, 0.0),
                    vector_rank=v_rank.get(doc_id, 0),
                    keyword_rank=k_rank.get(doc_id, 0),
                    metadata=meta,
                )
            )

        results.sort(key=lambda r: r.combined_score, reverse=True)
        return results[:k]

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def explain_result(self, query: str, result: HybridSearchResult) -> str:
        """Return a human-readable explanation for a hybrid search result.

        Example output::

            [id=doc_42] combined=0.8120 | vector rank #1 (score 0.9200, weight 70%)
            + keyword rank #3 (score 0.4500, weight 30%) | query: "Error 503"

        Args:
            query:  The original query string.
            result: A :class:`HybridSearchResult` to explain.

        Returns:
            A single descriptive string.
        """
        v_part = (
            f"vector rank #{result.vector_rank} (score {result.vector_score:.4f}, "
            f"weight {self.vector_weight * 100:.0f}%)"
            if result.vector_rank
            else f"not in vector results (score 0, weight {self.vector_weight * 100:.0f}%)"
        )
        k_part = (
            f"keyword rank #{result.keyword_rank} (score {result.keyword_score:.4f}, "
            f"weight {self.keyword_weight * 100:.0f}%)"
            if result.keyword_rank
            else f"not in keyword results (score 0, weight {self.keyword_weight * 100:.0f}%)"
        )
        return (
            f"[id={result.id!r}] combined={result.combined_score:.4f} | "
            f"{v_part} + {k_part} | query: {query!r}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _lookup_text_meta(
        self, doc_id: str, vector_results: list
    ) -> tuple[str, dict]:
        """Retrieve text and metadata for a document id.

        Prefers vector result objects (fast); falls back to the keyword index.
        """
        for r in vector_results:
            if r.id == doc_id:
                return r.text, r.metadata or {}
        for doc in self._indexed_docs:
            if doc.id == doc_id:
                return doc.text, doc.metadata or {}
        return "", {}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def _run_demo() -> None:
    """Demonstrate that hybrid search ranks the exact-match document higher
    than pure vector search when the query contains a specific error code."""
    import numpy as np
    from vector_database import SimpleVectorStore

    rng = np.random.default_rng(0)
    DIM = 64

    def rand_embed() -> list[float]:
        v = rng.standard_normal(DIM)
        return (v / np.linalg.norm(v)).tolist()

    # Two very similar embeddings for "server error" documents
    base = rand_embed()
    similar = [(b + 0.05 * rng.standard_normal()) for b in base]
    similar_norm = np.array(similar)
    similar_norm = (similar_norm / np.linalg.norm(similar_norm)).tolist()

    docs = [
        VectorDocument(
            id="doc_A",
            text="Server returns error 503 when traffic exceeds limits",
            embedding=base,
        ),
        VectorDocument(
            id="doc_B",
            text="Server errors occur under heavy load",
            embedding=similar_norm,
        ),
    ]
    # Add 18 more unrelated documents
    for i in range(18):
        docs.append(
            VectorDocument(
                id=f"noise_{i}",
                text=f"Unrelated content about topic {i}",
                embedding=rand_embed(),
            )
        )

    db = SimpleVectorStore()
    db.batch_insert(docs)

    hs = HybridSearch(db, vector_weight=0.7)
    hs.build_keyword_index(docs)

    # The query embedding is similar to both doc_A and doc_B
    query_text = "Error 503"
    query_emb = base  # Maximally similar to doc_A

    print("=" * 60)
    print("Hybrid Search Demo — 'Error 503'")
    print("=" * 60)

    # --- Pure vector search ---
    print("\n[Pure vector search — top 3]")
    pure = db.search(query_emb, k=3)
    for i, r in enumerate(pure, 1):
        print(f"  #{i} id={r.id!r:12s} score={r.score:.4f}  text={r.text[:50]!r}")

    # --- Weighted hybrid search ---
    print("\n[Weighted hybrid (70% vector / 30% keyword) — top 3]")
    hybrid = hs.search(query_text, query_emb, k=3)
    for i, r in enumerate(hybrid, 1):
        print(f"  #{i} id={r.id!r:12s} combined={r.combined_score:.4f}  text={r.text[:50]!r}")
        print(f"       {hs.explain_result(query_text, r)}")

    # --- RRF hybrid search ---
    print("\n[RRF hybrid — top 3]")
    rrf = hs.search_with_rrf(query_text, query_emb, k=3)
    for i, r in enumerate(rrf, 1):
        print(f"  #{i} id={r.id!r:12s} combined={r.combined_score:.6f}  text={r.text[:50]!r}")

    # Summary: does hybrid rank doc_A first?
    doc_a_rank_pure = next((i for i, r in enumerate(pure, 1) if r.id == "doc_A"), None)
    doc_a_rank_hybrid = next((i for i, r in enumerate(hybrid, 1) if r.id == "doc_A"), None)
    doc_a_rank_rrf = next((i for i, r in enumerate(rrf, 1) if r.id == "doc_A"), None)
    print(
        f"\ndoc_A ('error 503') rank: "
        f"vector=#{doc_a_rank_pure}  "
        f"hybrid=#{doc_a_rank_hybrid}  "
        f"rrf=#{doc_a_rank_rrf}"
    )


if __name__ == "__main__":
    _run_demo()
