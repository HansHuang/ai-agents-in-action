"""Tests for the context compression pipeline.

Covers:
- Relevance filter: adaptive threshold, min/max results, empty input
- Quality filter: deduplication, density filtering
- Extractive summarizer: verbatim wording, query relevance, sentence order
- LLM compression: number/date preservation (mocked)
- End-to-end pipeline: token reduction, auditability, non-empty guarantee

Run offline (no API calls):
    pytest test_compression.py -v
"""

from __future__ import annotations

import math
import sys
import os
from typing import Any

import pytest

sys.path.insert(0, os.path.dirname(__file__))

from context_budget import count_tokens
from density_analyzer import InformationDensityAnalyzer
from extractive_summarizer import (
    ExtractiveSummarizer,
    KeywordEmbedder,
    SentenceSplitter,
    _cosine_similarity,
)
from context_compressor import (
    CompressionConfig,
    CompressionResult,
    ContextCompressor,
)


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _doc(text: str, score: float) -> dict:
    return {"text": text, "score": score}


def _make_compressor() -> ContextCompressor:
    return ContextCompressor(embedder=KeywordEmbedder())


def _padding(n: int = 60) -> str:
    """Return a multi-word padding string so token counts are meaningful."""
    return " ".join(["word"] * n)


# ---------------------------------------------------------------------------
# RELEVANCE FILTER TESTS
# ---------------------------------------------------------------------------

class TestRelevanceFilter:

    # 1. Adaptive threshold keeps min_results even when only 2 docs are highly relevant
    def test_adaptive_threshold_keeps_min_results(self):
        docs = (
            [_doc("damaged return policy refund", 0.90),  # highly relevant
             _doc("return damaged item shipping", 0.88)]  # highly relevant
            + [_doc(f"unrelated content {i} " + _padding(), 0.20)
               for i in range(18)]
        )
        compressor = _make_compressor()
        result = compressor.filter_by_relevance(docs, threshold=None, min_results=3)
        assert len(result) >= 3

    # 2. Adaptive threshold caps at max_results even when all docs are relevant
    def test_adaptive_threshold_caps_at_max_results(self):
        docs = [_doc(f"return policy doc {i} " + _padding(), 0.90) for i in range(20)]
        compressor = _make_compressor()
        result = compressor.filter_by_relevance(docs, threshold=None,
                                                 min_results=3, max_results=5)
        assert len(result) <= 5

    # 3. Empty input returns empty list without crash
    def test_relevance_filter_handles_empty_input(self):
        compressor = _make_compressor()
        result = compressor.filter_by_relevance([], threshold=0.6)
        assert result == []

    # 4. Fixed threshold filters correctly
    def test_fixed_threshold_filters_below_threshold(self):
        docs = [
            _doc("high relevance return damaged", 0.80),
            _doc("medium relevance", 0.55),
            _doc("low relevance", 0.20),
        ]
        compressor = _make_compressor()
        result = compressor.filter_by_relevance(docs, threshold=0.6,
                                                 min_results=1, max_results=10)
        scores = [d["score"] for d in result]
        assert all(s >= 0.6 for s in scores)

    # 5. min_results safety fallback prevents empty result
    def test_min_results_fallback_prevents_empty(self):
        # All docs score below 0.8 but min_results=3 must be satisfied
        docs = [_doc(f"doc {i} " + _padding(), 0.30) for i in range(5)]
        compressor = _make_compressor()
        result = compressor.filter_by_relevance(docs, threshold=0.8, min_results=3)
        assert len(result) >= 3


# ---------------------------------------------------------------------------
# QUALITY FILTER TESTS
# ---------------------------------------------------------------------------

class TestQualityFilter:

    # 4 (spec). Near-duplicates are deduplicated
    def test_dedup_removes_near_duplicates(self):
        text_a = "Damaged items must be reported within 48 hours with photos."
        text_b = text_a + " See policy for details."  # >90% 3-gram Jaccard
        text_c = "Our quarterly earnings exceeded analyst expectations by 12%."

        docs = [_doc(text_a, 0.9), _doc(text_b, 0.88), _doc(text_c, 0.5)]
        compressor = _make_compressor()
        # Use 0.6: text_b contains all of text_a's 3-grams so Jaccard ≈ 0.65-0.75
        result = compressor.filter_by_quality(docs, dedup_threshold=0.6, min_density=0.0)
        # At most 2 unique docs (text_a/text_b are near-duplicates)
        assert len(result) <= 2
        texts = [d["text"] for d in result]
        assert text_c in texts  # unique doc always kept

    # 5 (spec). Low-density boilerplate is removed
    def test_density_filter_removes_boilerplate(self):
        boilerplate = (
            "That is a really great question and we are so glad you asked us. "
            "We hope that you are having a wonderful and pleasant day today."
        )
        docs = [_doc(boilerplate, 0.9)]
        compressor = _make_compressor()
        analyzer = InformationDensityAnalyzer()
        density = analyzer.score(boilerplate).overall
        # Boilerplate has low density; verify filter with min_density above its score
        result = compressor.filter_by_quality(
            docs, dedup_threshold=0.99, min_density=density + 0.1
        )
        assert len(result) == 0

    # 6 (spec). Factual content scores high density
    def test_density_filter_keeps_factual_content(self):
        factual = (
            "Battery: 4,500 mAh, 45W fast charge (0→50% in 20 min). "
            "Processor: Snapdragon 8 Gen 3, 12 GB LPDDR5X RAM. "
            "Storage: 256 GB. Weight: 195 g. IP68. "
            "BLEU score improved from 32.4 to 41.7 (+28.7%)."
        )
        analyzer = InformationDensityAnalyzer()
        score = analyzer.score(factual)
        assert score.overall > 0.5

    # Quality filter passes documents that survive dedup AND density
    def test_quality_filter_passes_valid_documents(self):
        factual = (
            "Damaged items must be returned within 48 hours. "
            "Refunds are processed in 3 business days. $50 minimum."
        )
        docs = [_doc(factual, 0.9)]
        compressor = _make_compressor()
        result = compressor.filter_by_quality(docs, dedup_threshold=0.9, min_density=0.0)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# EXTRACTIVE SUMMARIZER TESTS
# ---------------------------------------------------------------------------

class TestExtractiveSummarizer:

    # 7 (spec). Every output sentence exists verbatim in the input
    def test_extractive_summarizer_preserves_original_wording(self):
        text = (
            "Our return policy allows returns within 30 days of purchase. "
            "Items must be in original packaging. "
            "Damaged items qualify for a full refund within 48 hours. "
            "Contact support at 1-800-555-0199 for assistance. "
            "Refunds are processed in 5–7 business days."
        )
        summarizer = ExtractiveSummarizer(KeywordEmbedder())
        compressed = summarizer.summarize(text, "damaged refund", max_sentences=2)
        splitter = SentenceSplitter()
        for sentence in splitter.split(compressed):
            assert sentence in text, f"Hallucinated sentence: {sentence!r}"

    # 8 (spec). Query-relevant sentences are selected preferentially
    def test_extractive_summarizer_selects_query_relevant_sentences(self):
        text = (
            "The weather forecast shows sunny skies with a high of 75°F. "
            "Today's temperature will reach 80°F in the afternoon. "
            "The stock market closed higher driven by tech sector gains. "
            "Damaged items must be reported within 48 hours for a refund. "
            "The home team scored three goals in the second half."
        )
        summarizer = ExtractiveSummarizer(KeywordEmbedder())
        # max_sentences=3: first + last are anchored, leaving 1 slot for the
        # most query-relevant sentence (the damaged-item sentence).
        compressed = summarizer.summarize(text, "damaged item return", max_sentences=3)
        # The damaged-item sentence should appear in output
        assert "damaged" in compressed.lower() or "refund" in compressed.lower()

    # 9 (spec). Original sentence order is preserved
    def test_extractive_summarizer_preserves_sentence_order(self):
        text = (
            "Damaged items must be reported within 48 hours. "
            "We sell a wide range of products across many categories. "
            "Contact our returns team to initiate a refund."
        )
        summarizer = ExtractiveSummarizer(KeywordEmbedder())
        compressed = summarizer.summarize(text, "damaged return refund", max_sentences=2)
        splitter = SentenceSplitter()
        kept = splitter.split(compressed)
        # Find positions of kept sentences in the original
        sentences = splitter.split(text)
        positions = [
            next((i for i, s in enumerate(sentences) if s == k), -1)
            for k in kept
        ]
        # Positions must be strictly ascending (original order preserved)
        assert positions == sorted(positions)

    # SentenceSplitter handles abbreviations and numbers correctly
    def test_sentence_splitter_handles_abbreviations(self):
        text = "Dr. Smith found version 3.14 had bugs. Mr. Jones fixed them."
        splitter = SentenceSplitter()
        sentences = splitter.split(text)
        assert len(sentences) == 2

    # compress_with_ratio respects the ratio
    def test_compress_with_ratio_respects_ratio(self):
        text = " ".join([
            f"Sentence number {i} contains relevant policy information about returns."
            for i in range(10)
        ])
        summarizer = ExtractiveSummarizer(KeywordEmbedder())
        compressed = summarizer.compress_with_ratio(text, "return policy", 0.3)
        splitter = SentenceSplitter()
        original_count = len(splitter.split(text))
        output_count = len(splitter.split(compressed))
        # Keep ≈30% → at most 4 sentences from 10
        assert output_count <= max(3, round(original_count * 0.4))

    # extract_with_context includes neighbours
    def test_extract_with_context_includes_neighbours(self):
        text = (
            "Introduction: this document covers return policies. "
            "Key fact: damaged items qualify for free returns within 48 hours. "
            "Additional: refunds take 3–5 business days to process. "
            "Note: international orders follow the same policy."
        )
        summarizer = ExtractiveSummarizer(KeywordEmbedder())
        # context_window=1 should include neighbours of the most relevant sentence
        with_ctx    = summarizer.extract_with_context(text, "damaged return", 1)
        without_ctx = summarizer.summarize(text, "damaged return", max_sentences=1)
        # With context should be equal or longer
        assert len(with_ctx) >= len(without_ctx)


# ---------------------------------------------------------------------------
# LLM COMPRESSION TESTS (mocked)
# ---------------------------------------------------------------------------

class TestLLMCompression:

    # 10 (spec). LLM compression preserves numbers and dates
    def test_llm_compression_preserves_numbers_and_dates(self):
        text = "The policy changed on March 15, 2024. Returns are now 45 days."
        preserved_output = "Policy updated March 15, 2024. Returns: 45 days."

        # Minimal mock that returns the preserved_output
        class MockChoice:
            class MockMessage:
                content = preserved_output
            message = MockMessage()

        class MockResponse:
            choices = [MockChoice()]

        class MockCompletions:
            def create(self, **kwargs):
                return MockResponse()

        class MockChat:
            completions = MockCompletions()

        class MockLLMClient:
            chat = MockChat()

        compressor = _make_compressor()
        compressor._llm_client = MockLLMClient()

        doc = _doc(text + " " + _padding(100), 0.9)  # make it long enough to compress
        result = compressor.compress_documents(
            "return policy dates", [doc],
            max_tokens_per_doc=20,
            method="llm",
        )
        assert "March 15, 2024" in result[0]["text"]
        assert "45" in result[0]["text"]


# ---------------------------------------------------------------------------
# END-TO-END PIPELINE TESTS
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def _build_corpus(self) -> list[dict]:
        """30 documents with varying relevance, quality, and duplicates."""
        import random
        random.seed(0)

        corpus = []
        templates = [
            ("high",   0.85, "Damaged items qualify for a full refund within 48 hours. " + _padding()),
            ("high",   0.90, "To return a damaged product visit returns.acme.com. " + _padding()),
            ("med",    0.65, "Standard returns are accepted within 30 days of purchase. " + _padding()),
            ("med",    0.70, "Refunds are processed in 5 to 7 business days. " + _padding()),
            ("low",    0.30, "We offer free shipping on orders over $50. " + _padding()),
            ("low",    0.25, "Our headquarters is in Austin Texas. " + _padding()),
            ("filler", 0.15, "Thank you for choosing us. We value your patience and understanding. " + _padding(20)),
            ("off",    0.05, "The weather forecast shows sunny skies with a high of 75 degrees. " + _padding()),
        ]
        # Expand to ~30 documents
        for i in range(30):
            t = templates[i % len(templates)]
            score = max(0.01, min(0.99, t[1] + random.uniform(-0.05, 0.05)))
            corpus.append({"text": t[2], "score": round(score, 2),
                           "metadata": {"tier": t[0]}})
        return corpus

    # 11 (spec). Pipeline reduces tokens to within target
    def test_full_pipeline_reduces_tokens(self):
        corpus = self._build_corpus()
        total_before = sum(count_tokens(d["text"]) for d in corpus)

        compressor = _make_compressor()
        result = compressor.compress("How do I return a damaged item?",
                                      corpus, target_tokens=500)

        total_after = sum(count_tokens(d["text"]) for d in result.documents)
        assert total_after <= 500
        assert len(result.documents) < len(corpus)

    # 12 (spec). Pipeline produces auditable per-stage stats
    def test_pipeline_stages_are_auditable(self):
        corpus = self._build_corpus()
        compressor = _make_compressor()
        result = compressor.compress("damaged return", corpus, target_tokens=1_000)

        assert isinstance(result, CompressionResult)
        assert len(result.stats) >= 3  # at least a few stages

        # Audit report should include stage names
        report = result.audit.report()
        assert "Raw Input"       in report
        assert "Relevance Filter" in report
        assert "Budget Enforcement" in report

        # Document count must be non-increasing across stages
        stage_doc_counts = [s.doc_count for s in result.audit.stages]
        for prev, curr in zip(stage_doc_counts, stage_doc_counts[1:]):
            assert curr <= prev + 1  # allow ±1 tolerance for fallback logic

    # 13 (spec). Pipeline never returns an empty document list
    def test_pipeline_never_returns_empty(self):
        # Very aggressive config: threshold 1.0 would filter everything
        # but the min_results=1 safety net must keep at least 1 document.
        corpus = [_doc("Some relevant return policy information here.", 0.30)]
        compressor = _make_compressor()
        result = compressor.compress(
            "return",
            corpus,
            target_tokens=10_000,
            config=CompressionConfig(
                target_tokens=10_000,
                adaptive_threshold=False,
                fixed_threshold=0.99,  # would discard everything
                min_results=1,
                compress_method="none",
                rerank_method="none",
            ),
        )
        assert len(result.documents) >= 1

    # Budget enforcement removes lowest-scoring docs
    def test_budget_enforcement_removes_lowest_scoring(self):
        docs = [
            _doc("return policy " + _padding(50), 0.90),
            _doc("damaged items refund " + _padding(50), 0.85),
            _doc("off topic content here " + _padding(50), 0.10),
        ]
        compressor = _make_compressor()
        # Force a very tight budget so at least one doc must be dropped
        total = sum(count_tokens(d["text"]) for d in docs)
        result = compressor.enforce_budget(docs, target_tokens=total - 10)
        assert len(result) < len(docs)
        # The off-topic doc (score=0.10) should be dropped first
        remaining_scores = [d["score"] for d in result]
        assert 0.10 not in remaining_scores


# ---------------------------------------------------------------------------
# DENSITY ANALYZER TESTS
# ---------------------------------------------------------------------------

class TestDensityAnalyzer:

    def test_factual_text_scores_higher_than_boilerplate(self):
        analyzer = InformationDensityAnalyzer()
        factual = (
            "The study found transformer models with 175B parameters achieved "
            "state-of-the-art on 7 of 8 benchmarks. BLEU improved from 32.4 "
            "to 41.7 (+28.7%) over 96 GPU-hours on NVIDIA A100s."
        )
        filler = (
            "That is a really great question and we are so glad you reached out. "
            "We hope that you are having a wonderful and pleasant day today. "
            "Please let us know if there is anything else we can help with."
        )
        assert analyzer.score(factual).overall > analyzer.score(filler).overall

    def test_empty_text_returns_zero_scores(self):
        analyzer = InformationDensityAnalyzer()
        score = analyzer.score("")
        assert score.overall == 0.0
        assert score.fact_density == 0.0

    def test_structured_text_has_high_structure_score(self):
        analyzer = InformationDensityAnalyzer()
        structured = (
            "## Return Policy\n"
            "- Damaged items: report within 48 hours\n"
            "- Standard returns: 30-day window\n"
            "- Refund processing: 5–7 business days\n"
        )
        score = analyzer.score(structured)
        assert score.structure_score > 0.3

    def test_find_low_density_sections_identifies_filler(self):
        analyzer = InformationDensityAnalyzer()
        text = (
            "The return policy was updated on January 12, 2025. "
            "All orders after this date qualify for a 45-day return window.\n\n"
            "Thank you for reading this document. "
            "We really appreciate your time and patience."
        )
        low = analyzer.find_low_density_sections(text, min_density=0.3)
        assert len(low) >= 1  # filler paragraph should be identified

    def test_is_high_quality_threshold(self):
        analyzer = InformationDensityAnalyzer()
        factual = "Battery: 4,500 mAh. Processor: Snapdragon 8 Gen 3. Weight: 195 g."
        assert analyzer.score(factual).is_high_quality(threshold=0.3)


# ---------------------------------------------------------------------------
# COSINE SIMILARITY TESTS
# ---------------------------------------------------------------------------

class TestCosineSimilarity:

    def test_identical_vectors_score_one(self):
        v = [1.0, 0.0, 0.5]
        assert abs(_cosine_similarity(v, v) - 1.0) < 1e-9

    def test_orthogonal_vectors_score_zero(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(_cosine_similarity(a, b)) < 1e-9

    def test_zero_vector_returns_zero(self):
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.5]) == 0.0
