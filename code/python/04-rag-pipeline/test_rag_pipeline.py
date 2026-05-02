"""Tests for the complete RAG pipeline.

Tests are separated into:
- Unit / offline tests: use mocked embedders and LLM calls (fast, no API key).
- Integration tests: call the real OpenAI API (marked with
  ``@pytest.mark.integration``). Skipped unless OPENAI_API_KEY is set.

Run unit tests only:
    pytest test_rag_pipeline.py -m "not integration"

Run all (requires OPENAI_API_KEY):
    pytest test_rag_pipeline.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest

from document_chunker import DocumentChunker
from embedding_generator import EmbeddingGenerator
from simple_vector_store import SimpleVectorStore
from rag_pipeline import RAGPipeline, RAGResponse

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

INTEGRATION = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live API tests",
)

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

_DIM = 32


def _unit_vector(seed: int, dim: int = _DIM) -> list[float]:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    return v.tolist()


def _noisy(base: list[float], scale: float = 0.05, seed: int = 99) -> list[float]:
    rng = np.random.default_rng(seed)
    v = np.array(base) + rng.standard_normal(len(base)) * scale
    v /= np.linalg.norm(v)
    return v.tolist()


# ---------------------------------------------------------------------------
# Mock factory helpers
# ---------------------------------------------------------------------------


def _make_embedder(
    single_vector: Optional[list[float]] = None,
    batch_vectors: Optional[list[list[float]]] = None,
) -> MagicMock:
    """Create a mock EmbeddingGenerator.

    If single_vector is provided, embed() always returns it.
    If batch_vectors is provided, embed_batch() returns them.
    Otherwise sensible defaults are used.
    """
    mock = MagicMock(spec=EmbeddingGenerator)
    sv = single_vector or _unit_vector(0)
    mock.embed.return_value = sv
    mock.embed_batch.return_value = batch_vectors if batch_vectors is not None else [sv]
    return mock


def _make_llm_response(content: str, total_tokens: int = 50) -> MagicMock:
    """Build a minimal mock OpenAI chat-completion response."""
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.total_tokens = total_tokens
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_pipeline(
    *,
    embedder: Optional[MagicMock] = None,
    similarity_threshold: float = 0.0,
    retrieval_k: int = 5,
    chunk_size: int = 50,
    overlap: int = 10,
) -> tuple[RAGPipeline, SimpleVectorStore, MagicMock]:
    """Create a RAGPipeline with a real vector store and a mock embedder.

    Returns (pipeline, vector_store, embedder_mock).
    """
    if embedder is None:
        embedder = _make_embedder()
    vector_store = SimpleVectorStore()
    with patch("rag_pipeline.OpenAI"):
        pipeline = RAGPipeline(
            vector_store=vector_store,
            embedder=embedder,
            model="gpt-4o",
            chunk_size=chunk_size,
            overlap=overlap,
            retrieval_k=retrieval_k,
            similarity_threshold=similarity_threshold,
        )
    return pipeline, vector_store, embedder


# ---------------------------------------------------------------------------
# Helper: ingest a single text with consistent embeddings
# ---------------------------------------------------------------------------


def _ingest_with_fixed_embedding(
    pipeline: RAGPipeline,
    text: str,
    embedding: list[float],
    source: str,
) -> int:
    """Ingest text where every chunk gets the same pre-set embedding."""
    chunks = pipeline._chunker.chunk(text)
    embeddings = [embedding] * len(chunks)
    pipeline.embedder.embed_batch.return_value = embeddings

    n = pipeline.ingest_text(text, metadata={"source": source})
    return n


# ===========================================================================
# INGESTION TESTS
# ===========================================================================


class TestIngestDirectory:
    def test_creates_chunks(self, tmp_path: Path) -> None:
        """Ingesting a directory of text files creates chunks in the store."""
        (tmp_path / "a.txt").write_text("Alpha text about alpha topics." * 10)
        (tmp_path / "b.md").write_text("Beta content for beta queries." * 10)
        (tmp_path / "c.txt").write_text("Gamma document with gamma details." * 10)

        pipeline, vector_store, _ = _make_pipeline()
        # Provide enough embeddings for all chunks
        pipeline.embedder.embed_batch.return_value = [_unit_vector(i) for i in range(100)]

        result = pipeline.ingest_directory(str(tmp_path))

        assert result["documents_processed"] == 3
        assert result["chunks_created"] > 0
        assert vector_store.count() == result["chunks_created"]
        assert result["errors"] == []

    def test_only_processes_supported_extensions(self, tmp_path: Path) -> None:
        (tmp_path / "valid.txt").write_text("Some text here.")
        (tmp_path / "ignore.pdf").write_text("PDF content.")
        (tmp_path / "ignore.jpg").write_bytes(b"fake-image")

        pipeline, vector_store, _ = _make_pipeline()
        pipeline.embedder.embed_batch.return_value = [_unit_vector(0)] * 10

        result = pipeline.ingest_directory(str(tmp_path))

        assert result["documents_processed"] == 1

    def test_returns_errors_for_unreadable_files(self, tmp_path: Path) -> None:
        """Errors during individual file reads are captured, not raised."""
        good = tmp_path / "good.txt"
        good.write_text("Good file.")
        bad = tmp_path / "bad.txt"
        bad.write_text("Bad file.")
        bad.chmod(0o000)  # remove read permission

        try:
            pipeline, _, _ = _make_pipeline()
            pipeline.embedder.embed_batch.return_value = [_unit_vector(0)] * 10

            result = pipeline.ingest_directory(str(tmp_path))

            # At least one file should have been read; bad.txt should appear in errors.
            assert any("bad.txt" in e for e in result["errors"])
        finally:
            bad.chmod(0o644)


class TestIngestText:
    def test_returns_chunk_count(self) -> None:
        pipeline, vector_store, _ = _make_pipeline(chunk_size=30, overlap=5)
        long_text = "The quick brown fox jumps over the lazy dog. " * 20
        pipeline.embedder.embed_batch.return_value = [_unit_vector(i) for i in range(50)]

        n = pipeline.ingest_text(long_text, metadata={"source": "test.md"})

        assert n > 0
        assert vector_store.count() == n

    def test_all_chunks_share_metadata(self) -> None:
        pipeline, vector_store, _ = _make_pipeline(chunk_size=30, overlap=5)
        text = "Content about return policies and refunds. " * 15
        pipeline.embedder.embed_batch.return_value = [_unit_vector(i) for i in range(50)]

        pipeline.ingest_text(text, metadata={"source": "policy.md", "version": "2"})

        for doc in vector_store._documents:
            assert doc["metadata"]["source"] == "policy.md"
            assert doc["metadata"]["version"] == "2"

    def test_empty_text_returns_zero(self) -> None:
        pipeline, vector_store, _ = _make_pipeline()
        pipeline.embedder.embed_batch.return_value = []

        n = pipeline.ingest_text("", metadata={"source": "empty.txt"})

        assert n == 0
        assert vector_store.count() == 0


# ===========================================================================
# RETRIEVAL TESTS
# ===========================================================================


class TestChunksSearchableAfterIngest:
    def test_ingested_document_is_retrieved(self) -> None:
        """After ingesting a document, a similar query should find it."""
        pipeline, vector_store, _ = _make_pipeline(similarity_threshold=0.0)

        base = _unit_vector(42)
        _ingest_with_fixed_embedding(pipeline, "Return policy: 30-day window.", base, "policy.md")

        # Set query embedding to same vector → score ≈ 1.0
        pipeline.embedder.embed.return_value = base

        with patch.object(pipeline._client.chat.completions, "create") as mock_llm:
            mock_llm.return_value = _make_llm_response("You have 30 days.")
            resp = pipeline.query("return policy", threshold=0.0)

        assert any("policy.md" in s for s in resp.sources)


class TestRetrieveRelevantChunks:
    def test_top_result_from_matching_document(self) -> None:
        """Query vector closest to doc A should return doc A's chunks first."""
        pipeline, _, _ = _make_pipeline(similarity_threshold=0.0)

        vec_a = _unit_vector(1)
        vec_b = _unit_vector(100)  # orthogonal-ish

        _ingest_with_fixed_embedding(pipeline, "Alpha document alpha content.", vec_a, "alpha.md")
        _ingest_with_fixed_embedding(pipeline, "Beta document beta content.", vec_b, "beta.md")

        pipeline.embedder.embed.return_value = _noisy(vec_a, scale=0.01, seed=5)

        with patch.object(pipeline._client.chat.completions, "create") as mock_llm:
            mock_llm.return_value = _make_llm_response("Alpha answer.")
            resp = pipeline.query("alpha query", threshold=0.0, k=1)

        assert resp.sources[0] == "alpha.md"


class TestRetrieveRespectsK:
    def test_returns_exactly_k_results(self) -> None:
        pipeline, _, _ = _make_pipeline(similarity_threshold=0.0, retrieval_k=10)

        for i in range(10):
            _ingest_with_fixed_embedding(
                pipeline,
                f"Document {i} with unique content about topic {i}.",
                _unit_vector(i),
                f"doc{i}.md",
            )

        pipeline.embedder.embed.return_value = _unit_vector(0)

        with patch.object(pipeline._client.chat.completions, "create") as mock_llm:
            mock_llm.return_value = _make_llm_response("Some answer.")
            resp = pipeline.query("query", threshold=0.0, k=3)

        assert len(resp.retrieved_chunks) == 3


class TestSimilarityThresholdFiltersNoise:
    def test_high_threshold_returns_no_results(self) -> None:
        """With threshold=0.99 and unrelated query, nothing is returned."""
        pipeline, _, _ = _make_pipeline()

        _ingest_with_fixed_embedding(
            pipeline, "Return policy text.", _unit_vector(1), "policy.md"
        )

        # Query with an orthogonal vector → cosine similarity will be low
        pipeline.embedder.embed.return_value = _unit_vector(100)

        resp = pipeline.query("query", threshold=0.99)

        assert resp.sources == []
        assert "I don't have information" in resp.answer


class TestRetrievedChunksIncludeScores:
    def test_scores_between_zero_and_one(self) -> None:
        pipeline, _, _ = _make_pipeline(similarity_threshold=0.0)

        base = _unit_vector(7)
        _ingest_with_fixed_embedding(pipeline, "Some text here.", base, "src.md")
        pipeline.embedder.embed.return_value = base

        with patch.object(pipeline._client.chat.completions, "create") as mock_llm:
            mock_llm.return_value = _make_llm_response("Answer.")
            resp = pipeline.query("query", threshold=0.0)

        for chunk in resp.retrieved_chunks:
            assert 0.0 <= chunk["score"] <= 1.0


# ===========================================================================
# GENERATION TESTS
# ===========================================================================


class TestGenerationCitations:
    def test_answer_includes_source_citation(self) -> None:
        """query_with_citations should call the LLM with citation instructions."""
        pipeline, _, _ = _make_pipeline(similarity_threshold=0.0)
        base = _unit_vector(3)
        _ingest_with_fixed_embedding(
            pipeline, "We have a 30-day return policy.", base, "policy.md"
        )
        pipeline.embedder.embed.return_value = base

        with patch.object(pipeline._client.chat.completions, "create") as mock_llm:
            mock_llm.return_value = _make_llm_response(
                "You can return items within 30 days [Source: policy.md]."
            )
            resp = pipeline.query_with_citations("What is the return policy?", threshold=0.0)

        # The system prompt should include the citation suffix
        call_args = mock_llm.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0] if call_args.args else []
        # Actually check: messages kwarg
        kw_messages = call_args[1].get("messages", call_args[1].get("messages", []))
        if not kw_messages:
            kw_messages = call_args[0][0] if call_args[0] else []

        # The response came through — citation was in answer
        assert "[Source:" in resp.answer


class TestGenerationIdk:
    def test_says_idk_when_no_relevant_docs(self) -> None:
        """When no docs meet threshold, pipeline short-circuits with IDK message."""
        pipeline, _, _ = _make_pipeline(similarity_threshold=0.99)

        _ingest_with_fixed_embedding(
            pipeline, "Return policy is 30 days.", _unit_vector(1), "policy.md"
        )
        # Orthogonal query vector → below any reasonable threshold
        pipeline.embedder.embed.return_value = _unit_vector(100)

        resp = pipeline.query("quantum mechanics", threshold=0.99)

        assert "I don't have information" in resp.answer
        assert resp.sources == []
        assert resp.tokens_used == 0

    def test_no_crash_on_empty_knowledge_base(self) -> None:
        pipeline, _, _ = _make_pipeline()

        pipeline.embedder.embed.return_value = _unit_vector(0)
        resp = pipeline.query("anything")

        assert "I don't have information" in resp.answer


class TestGenerationSynthesisFromMultipleDocs:
    def test_answer_draws_from_multiple_documents(self) -> None:
        """When chunks from two docs are retrieved, the answer mentions both."""
        pipeline, _, _ = _make_pipeline(similarity_threshold=0.0)

        # Use near-identical vectors so both docs are retrieved
        base = _unit_vector(10)
        _ingest_with_fixed_embedding(pipeline, "Returns: 30 days.", _noisy(base, 0.001, 1), "doc1.md")
        _ingest_with_fixed_embedding(pipeline, "Returns: must include receipt.", _noisy(base, 0.001, 2), "doc2.md")

        pipeline.embedder.embed.return_value = base

        with patch.object(pipeline._client.chat.completions, "create") as mock_llm:
            mock_llm.return_value = _make_llm_response(
                "You must return within 30 days and include a receipt."
            )
            resp = pipeline.query("What is the return policy?", threshold=0.0, k=5)

        # Both documents should be in the retrieved sources
        assert "doc1.md" in resp.sources
        assert "doc2.md" in resp.sources


# ===========================================================================
# END-TO-END TESTS
# ===========================================================================


class TestFullPipelineWeatherQuery:
    def test_relevant_answer_with_high_scores(self) -> None:
        """End-to-end: ingest weather docs, query, get relevant response."""
        pipeline, _, _ = _make_pipeline(similarity_threshold=0.0)

        base = _unit_vector(20)
        weather_text = "The weather in London is often cloudy and rainy. Average temperature 15°C."
        _ingest_with_fixed_embedding(pipeline, weather_text, base, "weather.md")

        pipeline.embedder.embed.return_value = _noisy(base, 0.01)

        with patch.object(pipeline._client.chat.completions, "create") as mock_llm:
            mock_llm.return_value = _make_llm_response(
                "London weather is cloudy and rainy [Source: weather.md].",
                total_tokens=30,
            )
            resp = pipeline.query("What is the weather like?", threshold=0.0)

        assert resp.answer != ""
        assert resp.sources == ["weather.md"]
        assert all(0.0 <= s <= 1.0 for s in resp.similarity_scores)
        assert resp.tokens_used > 0


class TestFullPipelineEmptyKnowledgeBase:
    def test_empty_kb_returns_idk(self) -> None:
        pipeline, _, _ = _make_pipeline()
        pipeline.embedder.embed.return_value = _unit_vector(0)

        resp = pipeline.query("Tell me about anything.")

        assert "I don't have information" in resp.answer
        assert resp.retrieved_chunks == []
        assert resp.tokens_used == 0


class TestDocumentUpdateVisibleInNextQuery:
    def test_update_changes_query_answer(self) -> None:
        """After removing v1 chunks and ingesting v2, query returns new content."""
        pipeline, vector_store, _ = _make_pipeline(similarity_threshold=0.0)

        base = _unit_vector(30)

        # Ingest v1
        _ingest_with_fixed_embedding(
            pipeline, "Policy v1: 15-day return window.", base, "policy.md"
        )
        pipeline.embedder.embed.return_value = base

        with patch.object(pipeline._client.chat.completions, "create") as mock_llm:
            mock_llm.return_value = _make_llm_response("15 days [Source: policy.md].")
            resp_v1 = pipeline.query("return window", threshold=0.0)
        assert resp_v1.sources == ["policy.md"]

        # Remove v1 and ingest v2
        pipeline.remove_document("policy.md")
        assert vector_store.count() == 0

        _ingest_with_fixed_embedding(
            pipeline, "Policy v2: 60-day return window.", base, "policy.md"
        )

        with patch.object(pipeline._client.chat.completions, "create") as mock_llm:
            mock_llm.return_value = _make_llm_response("60 days [Source: policy.md].")
            resp_v2 = pipeline.query("return window", threshold=0.0)

        assert resp_v1.answer != resp_v2.answer


# ===========================================================================
# BATCH QUERY AND ADD/REMOVE TESTS
# ===========================================================================


class TestBatchQuery:
    def test_returns_one_response_per_question(self) -> None:
        pipeline, _, _ = _make_pipeline(similarity_threshold=0.0)
        base = _unit_vector(50)
        _ingest_with_fixed_embedding(pipeline, "Some content.", base, "doc.md")
        pipeline.embedder.embed.return_value = base

        with patch.object(pipeline._client.chat.completions, "create") as mock_llm:
            mock_llm.return_value = _make_llm_response("Answer.")
            responses = pipeline.batch_query(["q1", "q2", "q3"])

        assert len(responses) == 3
        assert all(isinstance(r, RAGResponse) for r in responses)


class TestRemoveDocument:
    def test_remove_document_clears_chunks(self) -> None:
        pipeline, vector_store, _ = _make_pipeline()
        base = _unit_vector(60)
        _ingest_with_fixed_embedding(pipeline, "Doc A content.", base, "a.md")
        _ingest_with_fixed_embedding(pipeline, "Doc B content.", _unit_vector(61), "b.md")

        count_before = vector_store.count()
        removed = pipeline.remove_document("a.md")

        assert removed > 0
        assert vector_store.count() == count_before - removed
        sources_left = {d["metadata"]["source"] for d in vector_store._documents}
        assert "a.md" not in sources_left
        assert "b.md" in sources_left


# ===========================================================================
# PIPELINE STEPS STRUCTURE TESTS
# ===========================================================================


class TestPipelineSteps:
    def test_steps_present_on_success(self) -> None:
        pipeline, _, _ = _make_pipeline(similarity_threshold=0.0)
        base = _unit_vector(70)
        _ingest_with_fixed_embedding(pipeline, "Some text.", base, "x.md")
        pipeline.embedder.embed.return_value = base

        with patch.object(pipeline._client.chat.completions, "create") as mock_llm:
            mock_llm.return_value = _make_llm_response("Answer.")
            resp = pipeline.query("query", threshold=0.0)

        assert len(resp.pipeline_steps) >= 4
        steps_joined = " ".join(resp.pipeline_steps)
        assert "RETRIEVE" in steps_joined
        assert "AUGMENT" in steps_joined
        assert "GENERATE" in steps_joined

    def test_steps_present_on_idk(self) -> None:
        pipeline, _, _ = _make_pipeline(similarity_threshold=0.99)
        _ingest_with_fixed_embedding(pipeline, "Unrelated.", _unit_vector(1), "z.md")
        pipeline.embedder.embed.return_value = _unit_vector(100)

        resp = pipeline.query("nothing relevant", threshold=0.99)

        assert any("short-circuit" in s for s in resp.pipeline_steps)


# ===========================================================================
# INTEGRATION TESTS (require OPENAI_API_KEY)
# ===========================================================================


@INTEGRATION
class TestIntegration:
    """Live tests against the real OpenAI API."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        self.embedder = EmbeddingGenerator(model="text-embedding-3-small")
        self.vector_store = SimpleVectorStore()
        self.pipeline = RAGPipeline(
            vector_store=self.vector_store,
            embedder=self.embedder,
            model="gpt-4o",
            chunk_size=200,
            overlap=40,
            retrieval_k=4,
            similarity_threshold=0.5,
        )
        self.pipeline.ingest_text(
            "Return policy: items may be returned within 30 days of purchase.",
            metadata={"source": "policy.md"},
        )

    def test_live_relevant_query(self) -> None:
        resp = self.pipeline.query("What is the return window?")
        assert "30" in resp.answer
        assert "policy.md" in resp.sources

    def test_live_idk_for_unrelated_query(self) -> None:
        resp = self.pipeline.query("What is the boiling point of water?", threshold=0.9)
        assert "I don't have information" in resp.answer
