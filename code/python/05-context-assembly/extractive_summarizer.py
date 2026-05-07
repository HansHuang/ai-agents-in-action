"""Extractive Summarizer — pull the most query-relevant sentences verbatim.

Unlike abstractive summarization (which rewrites), extractive selection
preserves original wording, eliminating hallucination risk.  Every sentence
in the output appears verbatim in the input.

A :class:`KeywordEmbedder` is included for offline use and tests.  Swap in a
real embedder (OpenAI, sentence-transformers, etc.) for production.

See: docs/04-context-engineering/03-context-compression-and-filtering.md
"""

from __future__ import annotations

import math
import re
from typing import Protocol

from context_budget import count_tokens


# ---------------------------------------------------------------------------
# Embedder protocol
# ---------------------------------------------------------------------------

class Embedder(Protocol):
    """Minimal interface expected by :class:`ExtractiveSummarizer`."""

    def embed(self, text: str) -> list[float]:
        ...


# ---------------------------------------------------------------------------
# Cosine similarity (no external dependencies)
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot   = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# SentenceSplitter
# ---------------------------------------------------------------------------

# Common abbreviations whose period must not split a sentence
_ABBREV_RE = re.compile(
    r'\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Ave|Blvd|etc|vs|e\.g|i\.e|'
    r'approx|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec|'
    r'U\.S|U\.K|E\.U|Corp|Inc|Ltd|LLC)\.',
    re.IGNORECASE,
)

# Decimal numbers: "version 3.14"
_DECIMAL_RE = re.compile(r'(\d+)\.(\d)')

# Sentence boundary: .!? followed by whitespace + uppercase
_BOUNDARY_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


class SentenceSplitter:
    """Smart sentence splitter using regex, handling abbreviations and numbers.

    Example::

        splitter = SentenceSplitter()
        sentences = splitter.split("Dr. Smith found 3.14 issues. They were fixed.")
        # → ["Dr. Smith found 3.14 issues.", "They were fixed."]
    """

    _MASK = '\x00'

    def split(self, text: str) -> list[str]:
        """Return a list of non-empty sentences from *text*."""
        # Mask abbreviation periods so they don't trigger a boundary
        masked = _ABBREV_RE.sub(
            lambda m: m.group(0).replace('.', self._MASK), text
        )
        # Mask decimal points inside numbers
        masked = _DECIMAL_RE.sub(r'\1' + self._MASK + r'\2', masked)

        parts     = _BOUNDARY_RE.split(masked)
        sentences = [s.replace(self._MASK, '.').strip() for s in parts]
        return [s for s in sentences if s]


# ---------------------------------------------------------------------------
# ExtractiveSummarizer
# ---------------------------------------------------------------------------

class ExtractiveSummarizer:
    """Summarize text by extracting the most query-relevant sentences.

    All output sentences appear verbatim in the input → zero hallucination risk.

    Example::

        summarizer = ExtractiveSummarizer(embedder=KeywordEmbedder())
        short = summarizer.summarize(long_doc, query="damaged item return",
                                     max_sentences=3)
    """

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.splitter = SentenceSplitter()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def summarize(self, text: str, query: str,
                  max_sentences: int = 5,
                  min_sentence_length: int = 20) -> str:
        """Extract up to *max_sentences* query-relevant sentences from *text*.

        Sentences shorter than *min_sentence_length* characters are skipped
        unless they are the only option.

        Always retains the first and last sentence to preserve intro/conclusion
        context when the text has more than *max_sentences* sentences.
        """
        sentences = self.splitter.split(text)
        sentences = [s for s in sentences if len(s) >= min_sentence_length]
        if not sentences:
            return text
        if len(sentences) <= max_sentences:
            return " ".join(sentences)

        q_emb  = self.embedder.embed(query)
        scores = self.score_sentences(sentences, q_emb)

        # Anchor first and last sentences for readability context
        keep: set[int] = {0, len(sentences) - 1}
        remaining = max_sentences - len(keep)

        if remaining > 0:
            ranked = sorted(
                ((i, s) for i, s in enumerate(scores) if i not in keep),
                key=lambda x: x[1],
                reverse=True,
            )
            for idx, _ in ranked[:remaining]:
                keep.add(idx)

        return " ".join(sentences[i] for i in sorted(keep))

    def summarize_document_batch(self, documents: list[dict],
                                  query: str,
                                  max_tokens_per_doc: int = 250) -> list[dict]:
        """Compress each document individually.

        Each document receives a *max_sentences* budget derived from
        *max_tokens_per_doc* (assuming ~30 tokens per sentence on average).
        """
        result = []
        for doc in documents:
            text = doc.get("text", "")
            orig_tokens = count_tokens(text)
            max_sentences = max(1, max_tokens_per_doc // 30)
            compressed = self.summarize(text, query, max_sentences=max_sentences)
            result.append({
                **doc,
                "text": compressed,
                "compressed": True,
                "original_tokens": orig_tokens,
                "compressed_tokens": count_tokens(compressed),
            })
        return result

    def extract_with_context(self, text: str, query: str,
                              context_window: int = 2) -> str:
        """Extract key sentences and include *context_window* neighbours.

        *context_window=2* includes 2 sentences before and after each key
        sentence so the extracted passage reads naturally.
        """
        sentences = self.splitter.split(text)
        if not sentences:
            return text

        q_emb  = self.embedder.embed(query)
        scores = self.score_sentences(sentences, q_emb)

        top_n = max(1, len(sentences) // 3)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        key_indices = {i for i, _ in ranked[:top_n]}

        include: set[int] = set()
        for ki in key_indices:
            for offset in range(-context_window, context_window + 1):
                idx = ki + offset
                if 0 <= idx < len(sentences):
                    include.add(idx)

        return " ".join(sentences[i] for i in sorted(include))

    def score_sentences(self, sentences: list[str],
                        query_embedding: list[float]) -> list[float]:
        """Return cosine similarity of each sentence to *query_embedding*."""
        return [
            _cosine_similarity(self.embedder.embed(s), query_embedding)
            for s in sentences
        ]

    def compress_with_ratio(self, text: str, query: str,
                             compression_ratio: float = 0.3) -> str:
        """Keep *compression_ratio* fraction of the original sentences."""
        sentences = self.splitter.split(text)
        keep_n = max(1, round(len(sentences) * compression_ratio))
        return self.summarize(text, query, max_sentences=keep_n)


# ---------------------------------------------------------------------------
# KeywordEmbedder — deterministic mock for offline use / tests
# ---------------------------------------------------------------------------

class KeywordEmbedder:
    """Deterministic keyword-vector embedder (no API required).

    Maps recurring domain terms to fixed vector dimensions so cosine
    similarity reflects topic overlap.  Suitable for tests and demos.
    Swap for a real embedder in production.
    """

    _DIM = 32
    _VOCAB: dict[str, int] = {
        # weather cluster
        "weather": 0, "temperature": 0, "rain": 0, "sunny": 0, "forecast": 0,
        # stock cluster
        "stock": 1, "price": 1, "market": 1, "shares": 1, "dividend": 1,
        # sport cluster
        "sport": 2, "game": 2, "team": 2, "score": 2, "player": 2,
        # returns/policy cluster
        "return": 3, "refund": 3, "policy": 3, "damaged": 3, "receipt": 3,
        # billing cluster
        "billing": 4, "invoice": 4, "payment": 4, "charge": 4, "cost": 4,
        # shipping cluster
        "shipping": 5, "delivery": 5, "package": 5, "tracking": 5, "order": 5,
        # security cluster
        "security": 6, "password": 6, "authentication": 6, "access": 6,
        # performance cluster
        "performance": 7, "speed": 7, "latency": 7, "benchmark": 7,
    }

    def embed(self, text: str) -> list[float]:
        """Return an L2-normalised keyword-count vector for *text*."""
        vec = [0.0] * self._DIM
        for word in re.findall(r'\b\w+\b', text.lower()):
            if word in self._VOCAB:
                vec[self._VOCAB[word]] += 1.0
        mag = math.sqrt(sum(x * x for x in vec))
        if mag > 0:
            return [x / mag for x in vec]
        # Fallback: uniform vector for texts with no recognised keywords
        val = 1.0 / math.sqrt(self._DIM)
        return [val] * self._DIM


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    summarizer = ExtractiveSummarizer(KeywordEmbedder())

    doc = (
        "Our return policy allows customers to return most items within 30 days of purchase. "
        "Items must be in their original packaging and in unused condition. "
        "To initiate a return, visit our website and log into your account. "
        "You can also call support at 1-800-555-0199 between 9 AM and 5 PM EST. "
        "If your item arrived damaged, please contact us within 48 hours and include photos. "
        "Damaged items qualify for an immediate replacement or full refund without return shipping. "
        "For international orders, a prepaid return label is included in the package. "
        "Processing time for refunds is 5–7 business days after we receive your return. "
        "Final sale items and digital downloads are not eligible for returns. "
        "For items over $200, a manager approval is required before the refund is processed."
    )

    query = "How do I return a damaged item?"
    print("=" * 60)
    print(f"Original: {len(doc.split())} words / {count_tokens(doc)} tokens")
    print("=" * 60)

    # Standard extractive summary
    compressed = summarizer.summarize(doc, query, max_sentences=3)
    print(f"Compressed ({len(compressed.split())} words):\n{compressed}\n")

    # Verify no hallucination
    splitter = SentenceSplitter()
    for sentence in splitter.split(compressed):
        assert sentence in doc, f"Hallucinated: {sentence!r}"
    print("Verified: all sentences appear verbatim in original.\n")

    # Context-window comparison
    with_ctx    = summarizer.extract_with_context(doc, query, context_window=1)
    without_ctx = summarizer.summarize(doc, query, max_sentences=3)
    print(f"With context window (±1):  {len(with_ctx.split())} words")
    print(f"Without context window:    {len(without_ctx.split())} words")


if __name__ == "__main__":
    _demo()
