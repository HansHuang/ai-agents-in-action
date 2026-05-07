"""Context Compressor — five-stage filtering pipeline for LLM context.

Maximizes information density while fitting within a token budget.

Pipeline stages:
  1. Relevance filter   — remove low-similarity documents (adaptive threshold)
  2. Quality filter     — remove near-duplicates and low-density chunks
  3. Rerank             — reorder by cross-encoder or embedding similarity
  4. Extractive compress — keep query-relevant sentences verbatim
  5. Budget enforcement  — drop lowest-scoring documents until within budget

See: docs/04-context-engineering/03-context-compression-and-filtering.md
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from context_budget import count_tokens
from density_analyzer import InformationDensityAnalyzer
from extractive_summarizer import ExtractiveSummarizer, KeywordEmbedder, _cosine_similarity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CompressionConfig:
    """Declarative configuration for the compression pipeline.

    Example::

        cfg = CompressionConfig(
            target_tokens=8_000,
            rerank_method="none",
            compress_method="extractive",
        )
    """

    target_tokens:      int   = 12_000
    min_results:        int   = 3
    max_results:        int   = 15
    dedup_threshold:    float = 0.9   # Jaccard similarity above which docs are duplicates
    min_density:        float = 0.1   # density_analyzer overall score minimum
    rerank_method:      str   = "none"          # "cross-encoder" | "llm" | "none"
    compress_method:    str   = "extractive"    # "extractive" | "llm" | "none"
    adaptive_threshold: bool  = True
    fixed_threshold:    float = 0.6             # used when adaptive_threshold=False


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

@dataclass
class _StageRecord:
    name:        str
    doc_count:   int
    token_count: int


@dataclass
class CompressionAudit:
    """Formatted audit of the full compression pipeline.

    Example::

        print(audit.report())
        # Stage               Docs  Tokens  % Remaining
        # ─────────────────────────────────────────────
        # Raw Input             50  125,000         100%
        # Relevance Filter      22   68,000          54%
        # …
    """

    stages:         list[_StageRecord]
    initial_docs:   int
    initial_tokens: int

    def report(self) -> str:
        """Return a formatted table showing docs and tokens at each stage."""
        col = 22
        lines = [
            f"{'Stage':<{col}}  {'Docs':>5}  {'Tokens':>9}  {'% Remaining':>12}",
            "─" * (col + 34),
        ]
        for rec in self.stages:
            pct = (rec.token_count / max(self.initial_tokens, 1)) * 100
            lines.append(
                f"{rec.name:<{col}}  {rec.doc_count:>5}  "
                f"{rec.token_count:>9,}  {pct:>11.0f}%"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Export as a structured dict suitable for logging / observability."""
        return {
            "initial_docs":   self.initial_docs,
            "initial_tokens": self.initial_tokens,
            "stages": [
                {
                    "name":          r.name,
                    "docs":          r.doc_count,
                    "tokens":        r.token_count,
                    "pct_remaining": round(
                        r.token_count / max(self.initial_tokens, 1) * 100, 1
                    ),
                }
                for r in self.stages
            ],
        }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CompressionResult:
    """Output of a :meth:`ContextCompressor.compress` run.

    Attributes:
        documents: Final compressed document list.
        stats:     Per-stage ``{name: {docs, tokens}}`` counts.
        audit:     Full :class:`CompressionAudit` with formatted report.
    """

    documents: list[dict]
    stats:     dict
    audit:     CompressionAudit

    def __str__(self) -> str:
        return self.audit.report()


# ---------------------------------------------------------------------------
# ContextCompressor
# ---------------------------------------------------------------------------

class ContextCompressor:
    """Five-stage context compression pipeline.

    Filters, reranks, and compresses a document list to maximise information
    density within a token budget.

    Example::

        compressor = ContextCompressor(embedder=KeywordEmbedder())
        result = compressor.compress(
            query="How do I return a damaged item?",
            documents=raw_docs,
            target_tokens=8_000,
        )
        print(result.audit.report())

    Cross-encoder reranking uses ``sentence-transformers`` if installed.
    LLM compression requires an ``openai.OpenAI`` client assigned to
    ``compressor._llm_client``.  Both fall back gracefully to "none" when
    the dependency is absent.
    """

    def __init__(self, embedder=None, cross_encoder=None) -> None:
        self.embedder       = embedder or KeywordEmbedder()
        self.cross_encoder  = cross_encoder
        self._density       = InformationDensityAnalyzer()
        self._summarizer    = ExtractiveSummarizer(self.embedder)
        self._llm_client    = None   # inject openai.OpenAI() to enable LLM stages

    # ------------------------------------------------------------------
    # Stage 1 — Relevance filter
    # ------------------------------------------------------------------

    def filter_by_relevance(self, documents: list[dict],
                             threshold: float | None = None,
                             min_results: int = 3,
                             max_results: int = 15) -> list[dict]:
        """Adaptive or fixed-threshold relevance filtering.

        Each document must have a ``"score"`` key (0–1 cosine similarity).
        If *threshold* is ``None``, the best threshold is derived automatically
        from the score distribution.
        """
        if not documents:
            return documents

        if threshold is None:
            threshold = self._adaptive_threshold(documents, min_results)

        filtered = [d for d in documents if d.get("score", 0.0) >= threshold]

        # Safety: never return fewer than min_results
        if len(filtered) < min_results and documents:
            logger.warning(
                "Relevance filter kept %d < min_results=%d (threshold=%.2f). "
                "Falling back to top-%d by score.",
                len(filtered), min_results, threshold, min_results,
            )
            filtered = sorted(
                documents, key=lambda d: d.get("score", 0.0), reverse=True
            )[:min_results]

        filtered = filtered[:max_results]
        logger.info(
            "Relevance filter: %d → %d (threshold=%.2f)",
            len(documents), len(filtered), threshold,
        )
        return filtered

    # ------------------------------------------------------------------
    # Stage 2 — Quality filter
    # ------------------------------------------------------------------

    def filter_by_quality(self, documents: list[dict],
                           dedup_threshold: float = 0.9,
                           min_density: float = 0.1) -> list[dict]:
        """Remove near-duplicate and low-information-density documents."""
        docs = self._deduplicate(documents, dedup_threshold)
        docs = self._filter_density(docs, min_density)
        return docs

    # ------------------------------------------------------------------
    # Stage 3 — Rerank
    # ------------------------------------------------------------------

    def rerank(self, query: str, documents: list[dict],
               top_k: int = 10, method: str = "cross-encoder") -> list[dict]:
        """Reorder documents by true relevance to *query*.

        Falls back to embedding re-scoring when the requested method is
        unavailable (cross-encoder not injected, LLM client not bound).
        """
        if not documents:
            return documents

        if method == "cross-encoder" and self.cross_encoder is not None:
            return self._rerank_cross_encoder(query, documents, top_k)

        if method == "llm" and self._llm_client is not None:
            return self._rerank_llm(query, documents, top_k)

        # Fallback: re-score with the embedder
        return self._rerank_embedding(query, documents, top_k)

    # ------------------------------------------------------------------
    # Stage 4 — Compress
    # ------------------------------------------------------------------

    def compress_documents(self, query: str, documents: list[dict],
                            max_tokens_per_doc: int = 300,
                            method: str = "extractive") -> list[dict]:
        """Compress each document to *max_tokens_per_doc* tokens.

        ``extractive``: key sentence selection (zero hallucination risk).
        ``llm``: LLM-based summarization (requires ``self._llm_client``).
        ``none``: skip compression entirely.
        """
        if method == "none":
            return documents

        if method == "llm" and self._llm_client is not None:
            return [self._compress_one_llm(doc, query, max_tokens_per_doc)
                    for doc in documents]

        # Default: extractive
        return self._summarizer.summarize_document_batch(
            documents, query, max_tokens_per_doc=max_tokens_per_doc
        )

    # ------------------------------------------------------------------
    # Stage 5 — Budget enforcement
    # ------------------------------------------------------------------

    def enforce_budget(self, documents: list[dict],
                       target_tokens: int) -> list[dict]:
        """Remove the lowest-scoring documents until the total fits within
        *target_tokens*.

        Always retains at least one document so the context is never empty.
        """
        docs = list(documents)
        while True:
            total = sum(count_tokens(d.get("text", "")) for d in docs)
            if total <= target_tokens or len(docs) <= 1:
                break
            # Drop the document with the lowest relevance score
            worst = min(
                range(len(docs)),
                key=lambda i: docs[i].get("rerank_score",
                                          docs[i].get("score", 0.0)),
            )
            dropped = docs.pop(worst)
            logger.debug(
                "Budget enforcement: dropped '%.50s…'",
                dropped.get("text", ""),
            )
        return docs

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def compress(self, query: str, documents: list[dict],
                 target_tokens: int = 12_000,
                 config: CompressionConfig | None = None) -> CompressionResult:
        """Run the full five-stage compression pipeline.

        Args:
            query:         User query driving relevance judgements.
            documents:     Raw document list.  Each dict must have ``"text"``
                           and ``"score"`` keys.
            target_tokens: Token budget for the final context.
            config:        :class:`CompressionConfig` (defaults applied when
                           ``None``).

        Returns:
            :class:`CompressionResult` with compressed documents, per-stage
            stats, and a :class:`CompressionAudit`.
        """
        cfg  = config or CompressionConfig(target_tokens=target_tokens)
        docs = list(documents)
        stages: list[_StageRecord] = []

        def _snap(name: str) -> None:
            stages.append(_StageRecord(
                name        = name,
                doc_count   = len(docs),
                token_count = sum(count_tokens(d.get("text", "")) for d in docs),
            ))

        _snap("Raw Input")

        # Stage 1
        threshold = None if cfg.adaptive_threshold else cfg.fixed_threshold
        docs = self.filter_by_relevance(
            docs,
            threshold   = threshold,
            min_results = cfg.min_results,
            max_results = cfg.max_results,
        )
        _snap("Relevance Filter")

        # Stage 2
        docs = self.filter_by_quality(
            docs,
            dedup_threshold = cfg.dedup_threshold,
            min_density     = cfg.min_density,
        )
        _snap("Quality Filter")

        # Stage 3
        if cfg.rerank_method != "none":
            docs = self.rerank(
                query, docs,
                top_k  = cfg.max_results,
                method = cfg.rerank_method,
            )
        _snap(f"Rerank ({cfg.rerank_method})")

        # Stage 4
        if cfg.compress_method != "none":
            tokens_per_doc = max(50, target_tokens // max(len(docs), 1))
            docs = self.compress_documents(
                query, docs,
                max_tokens_per_doc = tokens_per_doc,
                method             = cfg.compress_method,
            )
        _snap("Extract/Compress")

        # Stage 5
        docs = self.enforce_budget(docs, target_tokens)
        _snap("Budget Enforcement")

        audit = CompressionAudit(
            stages         = stages,
            initial_docs   = stages[0].doc_count,
            initial_tokens = stages[0].token_count,
        )
        stats = {
            r.name: {"docs": r.doc_count, "tokens": r.token_count}
            for r in stages
        }
        return CompressionResult(documents=docs, stats=stats, audit=audit)

    # ------------------------------------------------------------------
    # Audit helper
    # ------------------------------------------------------------------

    def audit(self, stages: dict) -> CompressionAudit:
        """Build a :class:`CompressionAudit` from an existing stage dict."""
        records = [
            _StageRecord(
                name        = name,
                doc_count   = info["docs"],
                token_count = info["tokens"],
            )
            for name, info in stages.items()
        ]
        if records:
            return CompressionAudit(
                stages         = records,
                initial_docs   = records[0].doc_count,
                initial_tokens = records[0].token_count,
            )
        return CompressionAudit(stages=[], initial_docs=0, initial_tokens=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _adaptive_threshold(self, documents: list[dict],
                             min_results: int) -> float:
        """Find the highest threshold that still keeps ≥ *min_results*."""
        scores = sorted(
            (d.get("score", 0.0) for d in documents), reverse=True
        )
        for t in (0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
            if sum(1 for s in scores if s >= t) >= min_results:
                return t
        return 0.3

    def _deduplicate(self, documents: list[dict],
                     threshold: float) -> list[dict]:
        """Remove near-duplicate documents using character 3-gram Jaccard."""

        def _ngrams(text: str, n: int = 3) -> set[str]:
            t = text.lower()
            return {t[i: i + n] for i in range(len(t) - n + 1)}

        def _jaccard(a: str, b: str) -> float:
            sa, sb = _ngrams(a), _ngrams(b)
            if not sa or not sb:
                return 0.0
            return len(sa & sb) / len(sa | sb)

        kept: list[dict] = []
        for doc in documents:
            if not any(
                _jaccard(doc.get("text", ""), k.get("text", "")) >= threshold
                for k in kept
            ):
                kept.append(doc)

        removed = len(documents) - len(kept)
        if removed:
            logger.info("Dedup removed %d/%d documents.", removed, len(documents))
        return kept

    def _filter_density(self, documents: list[dict],
                        min_density: float) -> list[dict]:
        kept = []
        for doc in documents:
            ds = self._density.score(doc.get("text", ""))
            if ds.overall >= min_density:
                kept.append({**doc, "density_score": ds.overall})
            else:
                logger.debug(
                    "Density drop (%.2f): %.60s…", ds.overall, doc.get("text", "")
                )
        logger.info(
            "Density filter: %d → %d (min=%.2f)",
            len(documents), len(kept), min_density,
        )
        return kept

    def _rerank_embedding(self, query: str, documents: list[dict],
                          top_k: int) -> list[dict]:
        q_emb = self.embedder.embed(query)
        for doc in documents:
            doc["rerank_score"] = _cosine_similarity(
                self.embedder.embed(doc.get("text", "")), q_emb
            )
        documents.sort(key=lambda d: d["rerank_score"], reverse=True)
        return documents[:top_k]

    def _rerank_cross_encoder(self, query: str, documents: list[dict],
                               top_k: int) -> list[dict]:
        pairs  = [(query, doc.get("text", "")) for doc in documents]
        scores = self.cross_encoder.predict(pairs)
        for doc, s in zip(documents, scores):
            doc["rerank_score"] = float(s)
        documents.sort(key=lambda d: d["rerank_score"], reverse=True)
        return documents[:top_k]

    def _rerank_llm(self, query: str, documents: list[dict],
                    top_k: int) -> list[dict]:
        import json as _json
        doc_list = "\n\n".join(
            f"[{i}] {doc.get('text', '')[:200]}…"
            for i, doc in enumerate(documents)
        )
        response = self._llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Query: {query}\n\nDocuments:\n{doc_list}\n\n"
                    f"Return the indices of the top {top_k} most relevant documents "
                    f'as JSON: {{"indices": [3, 1, ...]}}'
                ),
            }],
            response_format={"type": "json_object"},
        )
        try:
            indices = _json.loads(response.choices[0].message.content)["indices"]
            ranked  = [documents[i] for i in indices if i < len(documents)]
            for rank, doc in enumerate(ranked):
                doc["rerank_score"] = 1.0 - rank / max(len(ranked), 1)
            return ranked
        except Exception:
            return documents[:top_k]

    def _compress_one_llm(self, doc: dict, query: str,
                          max_tokens: int) -> dict:
        text = doc.get("text", "")
        if count_tokens(text) <= max_tokens:
            return doc
        response = self._llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize in ≤{max_tokens} tokens. "
                    f"Focus on information relevant to: {query}\n\n"
                    "Preserve: numbers, dates, names, actionable facts.\n"
                    "Remove: filler, repeated explanations.\n\n"
                    f"Document:\n{text}"
                ),
            }],
            max_tokens=max_tokens,
        )
        compressed = response.choices[0].message.content
        return {
            **doc,
            "text":              compressed,
            "compressed":        True,
            "original_tokens":   count_tokens(text),
            "compressed_tokens": count_tokens(compressed),
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    import random
    random.seed(42)

    # 30 sample documents with varying relevance scores
    topic_sentences = {
        "high_relevance": [
            "Damaged items must be reported within 48 hours with photos for a full refund.",
            "To return a damaged product, visit returns.acme.com and select 'Damaged Item'.",
            "We replace damaged goods at no cost, including return shipping.",
            "Refunds for damaged items are processed within 3 business days.",
            "Our damage claims team is available 24/7 at damage@acme.com.",
        ],
        "medium_relevance": [
            "Standard returns are accepted within 30 days of purchase.",
            "Refunds are processed in 5–7 business days.",
            "You must include the original receipt and packaging.",
            "International returns include a prepaid label.",
            "Final sale items are not eligible for returns.",
        ],
        "low_relevance": [
            "We offer free shipping on orders over $50.",
            "Our premium members get early access to new products.",
            "Follow us on social media for exclusive discounts.",
            "Our headquarters is located in Austin, Texas.",
            "We partner with sustainable packaging suppliers.",
        ],
        "boilerplate": [
            "Thank you for choosing us. We value your business.",
            "We appreciate your patience and understanding.",
            "Please let us know if you have any further questions.",
            "We hope to hear from you soon.",
            "Have a wonderful day and please reach out anytime.",
        ],
        "off_topic": [
            "Quarterly earnings exceeded analysts' expectations by 12%.",
            "The software update includes significant performance improvements.",
            "New product flavours are available in select markets.",
            "The team scored three goals in the second half.",
            "Weather forecast: sunny with highs of 75°F.",
        ],
    }

    score_map = {
        "high_relevance":  (0.85, 0.95),
        "medium_relevance":(0.60, 0.75),
        "low_relevance":   (0.35, 0.55),
        "boilerplate":     (0.10, 0.30),
        "off_topic":       (0.05, 0.20),
    }

    documents = []
    for category, sentences in topic_sentences.items():
        lo, hi = score_map[category]
        for sentence in sentences:
            documents.append({
                "text":  sentence + "  " + sentence[:40],
                "score": round(random.uniform(lo, hi), 2),
                "metadata": {"category": category},
            })

    # Add a near-duplicate to test dedup
    dup = dict(documents[0])
    dup["text"] = documents[0]["text"] + " (see also above for details)"
    documents.append(dup)

    random.shuffle(documents)

    total_before = sum(count_tokens(d["text"]) for d in documents)
    print("=" * 65)
    print(f"Input: {len(documents)} documents, {total_before:,} tokens")
    print("=" * 65)

    compressor = ContextCompressor()
    query = "How do I return a damaged item?"

    # Adaptive threshold
    r_adaptive = compressor.compress(
        query, documents, target_tokens=2_000,
        config=CompressionConfig(
            target_tokens=2_000, adaptive_threshold=True,
            compress_method="extractive",
        ),
    )
    print("\n--- Adaptive threshold ---")
    print(r_adaptive.audit.report())

    # Fixed threshold for comparison
    r_fixed = compressor.compress(
        query, documents, target_tokens=2_000,
        config=CompressionConfig(
            target_tokens=2_000, adaptive_threshold=False,
            fixed_threshold=0.7, compress_method="extractive",
        ),
    )
    print("\n--- Fixed threshold (0.7) ---")
    print(r_fixed.audit.report())

    print("\n--- Final documents (adaptive) ---")
    for i, d in enumerate(r_adaptive.documents, 1):
        print(f"  [{i}] score={d.get('score', 0):.2f}  {d['text'][:70]}")


if __name__ == "__main__":
    _demo()
