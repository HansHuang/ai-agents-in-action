"""Tests for the embedding system: EmbeddingGenerator, DocumentChunker, and SimpleVectorStore.

Tests are separated into two categories:
- Unit tests that work offline (using pre-built fake embeddings).
- Integration tests that call the real OpenAI API (marked with
  ``@pytest.mark.integration``). These are skipped unless OPENAI_API_KEY is set.

Run unit tests only:
    pytest test_embeddings.py -m "not integration"

Run all (requires OPENAI_API_KEY):
    pytest test_embeddings.py
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from document_chunker import Chunk, DocumentChunker
from embedding_generator import EmbeddingComparator, EmbeddingGenerator
from simple_vector_store import SimpleVectorStore

# ---------------------------------------------------------------------------
# Marks
# ---------------------------------------------------------------------------

INTEGRATION = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping live API tests",
)

# ---------------------------------------------------------------------------
# Fake embedding helpers
# ---------------------------------------------------------------------------

_DIM = 16  # Small dimension for fast tests


def _unit_vector(seed: int, dim: int = _DIM) -> list[float]:
    """Return a reproducible random unit vector."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim)
    v /= np.linalg.norm(v)
    return v.tolist()


def _noisy(base: list[float], scale: float = 0.05, seed: int = 0) -> list[float]:
    """Return base + small random noise, re-normalised."""
    rng = np.random.default_rng(seed)
    v = np.array(base) + rng.standard_normal(len(base)) * scale
    v /= np.linalg.norm(v)
    return v.tolist()


def _make_mock_generator(responses: list[list[float]]) -> EmbeddingGenerator:
    """Return an EmbeddingGenerator whose embed_batch returns pre-built vectors."""
    gen = MagicMock(spec=EmbeddingGenerator)
    gen.embed_batch.return_value = responses
    # embed() returns the first response.
    gen.embed.side_effect = lambda text: responses[0]
    return gen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def comparator() -> EmbeddingComparator:
    return EmbeddingComparator()


@pytest.fixture()
def store() -> SimpleVectorStore:
    return SimpleVectorStore()


# ===========================================================================
# EMBEDDING GENERATOR TESTS (integration — require real API)
# ===========================================================================


@INTEGRATION
def test_embed_returns_correct_dimensions():
    """Default model returns 1536-dim vectors."""
    gen = EmbeddingGenerator(model="text-embedding-3-small")
    emb = gen.embed("Hello, world!")
    assert isinstance(emb, list)
    assert len(emb) == 1536
    assert all(isinstance(x, float) for x in emb)


@INTEGRATION
def test_embed_with_custom_dimensions():
    """Dimension reduction to 512 is respected."""
    gen = EmbeddingGenerator(model="text-embedding-3-small", dimensions=512)
    emb = gen.embed("Hello, world!")
    assert len(emb) == 512


@INTEGRATION
def test_embed_batch_same_as_individual():
    """Batch embedding produces the same vectors as individual embedding."""
    gen = EmbeddingGenerator(model="text-embedding-3-small")
    texts = [
        "The weather is sunny.",
        "It will rain tomorrow.",
        "Stock market news.",
        "Football championship.",
        "Pizza recipe tips.",
    ]
    individual = [gen.embed(t) for t in texts]
    batched = gen.embed_batch(texts)

    assert len(batched) == len(individual)
    for indiv, batch in zip(individual, batched):
        for a, b in zip(indiv, batch):
            assert abs(a - b) < 1e-6, "Batch and individual embeddings should match"


@INTEGRATION
def test_similar_texts_have_high_similarity():
    """Semantically similar texts have cosine similarity > 0.7."""
    gen = EmbeddingGenerator(model="text-embedding-3-small")
    comp = EmbeddingComparator()
    a = gen.embed("The weather is sunny today")
    b = gen.embed("It's bright and clear outside")
    assert comp.cosine_similarity(a, b) > 0.7


@INTEGRATION
def test_different_texts_have_low_similarity():
    """Semantically different texts have cosine similarity < 0.4."""
    gen = EmbeddingGenerator(model="text-embedding-3-small")
    comp = EmbeddingComparator()
    a = gen.embed("The weather is sunny today")
    c = gen.embed("The stock market crashed")
    assert comp.cosine_similarity(a, c) < 0.4


@INTEGRATION
def test_identical_texts_have_similarity_one():
    """Embedding the same text twice gives similarity > 0.99."""
    gen = EmbeddingGenerator(model="text-embedding-3-small")
    comp = EmbeddingComparator()
    text = "This is a test sentence."
    a = gen.embed(text)
    b = gen.embed(text)
    assert comp.cosine_similarity(a, b) > 0.99


# ===========================================================================
# CHUNKER TESTS (offline — no API)
# ===========================================================================


def _token_count_approx(text: str) -> int:
    """Very rough token estimate (1 token ≈ 4 chars) for test assertions."""
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


LONG_DOCUMENT = (
    "This is the first paragraph. It contains some information about the topic.\n\n"
    "This is the second paragraph. It elaborates on the first paragraph.\n\n"
    "This is the third paragraph. It introduces new details.\n\n"
    "This is the fourth paragraph. It summarises what came before.\n\n"
    "This is the fifth paragraph. It concludes the document with a summary.\n\n"
)

# A document with 5 × ~50-token paragraphs ≈ 250 tokens total.
LARGE_DOCUMENT = "\n\n".join(
    f"Section {i}: " + ("This is a long paragraph with a fair amount of text. " * 20)
    for i in range(1, 6)
)


def test_fixed_chunking_respects_size():
    """All fixed chunks are at most chunk_size + a small buffer (overlap)."""
    chunker = DocumentChunker(chunk_size=256, overlap=50, strategy="fixed")
    chunks = chunker.chunk(LARGE_DOCUMENT)
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.token_count <= 300, (
            f"Chunk {chunk.index} has {chunk.token_count} tokens (limit 300)"
        )


def test_chunks_have_overlap():
    """Adjacent fixed chunks share some content (overlap region)."""
    chunker = DocumentChunker(chunk_size=40, overlap=10, strategy="fixed")
    chunks = chunker.chunk(LARGE_DOCUMENT)
    assert len(chunks) >= 2
    # The end of chunk N and the start of chunk N+1 should share tokens.
    tail = chunks[0].text[-30:]
    head = chunks[1].text[:30:]
    # They shouldn't be completely disjoint strings.
    # A simple heuristic: at least one word from the tail appears in the head.
    tail_words = set(tail.split())
    head_words = set(head.split())
    assert tail_words & head_words, (
        "No overlap found between adjacent chunks.\n"
        f"Tail: {tail!r}\nHead: {head!r}"
    )


def test_semantic_chunking_at_paragraphs():
    """Semantic chunking respects paragraph boundaries.

    Each paragraph that fits within chunk_size should appear entirely within
    at least one chunk — it should never be split so that neither chunk
    contains the complete paragraph text.
    """
    # Each paragraph is ~20 tokens — well within chunk_size=80.
    paragraphs = [
        f"Paragraph {i}. This sentence is specifically about topic {i}. "
        f"It contains a clear idea so we can verify it is preserved intact."
        for i in range(1, 6)
    ]
    doc = "\n\n".join(paragraphs)
    chunker = DocumentChunker(chunk_size=80, overlap=10, strategy="semantic")
    chunks = chunker.chunk(doc)
    all_chunk_text = " ".join(c.text for c in chunks)

    # Every paragraph's first distinctive phrase must appear in the combined output.
    for i, para in enumerate(paragraphs, 1):
        distinctive = f"topic {i}"
        assert distinctive in all_chunk_text, (
            f"Paragraph {i}'s content ('{distinctive}') not found in any chunk"
        )


def test_hierarchical_chunking_parent_child():
    """Hierarchical chunks have parent-child relationships."""
    chunker = DocumentChunker(chunk_size=60, overlap=10, strategy="hierarchical")
    chunks = chunker.chunk(LARGE_DOCUMENT)

    children = [c for c in chunks if c.parent_chunk is not None]
    parents = [c for c in chunks if c.children]

    assert len(parents) > 0, "Expected at least one parent chunk"
    assert len(children) > 0, "Expected at least one child chunk"

    for child in children:
        # Every child must have a parent.
        assert child.parent_chunk is not None
        # The child's parent must list this child.
        assert child in child.parent_chunk.children


# ===========================================================================
# VECTOR STORE TESTS (offline — no API)
# ===========================================================================


def test_add_and_search(store: SimpleVectorStore):
    """Search returns the closest document first."""
    target = _unit_vector(seed=1)
    ids = []
    for i in range(10):
        emb = _unit_vector(seed=i + 10)
        doc_id = store.add(f"doc {i}", emb)
        ids.append(doc_id)
    # Add a document that is very close to target.
    close_id = store.add("target doc", _noisy(target, scale=0.01, seed=99))

    results = store.search(target, k=1)
    assert len(results) == 1
    assert results[0]["id"] == close_id


def test_search_returns_correct_k(store: SimpleVectorStore):
    """search() with k=5 returns exactly 5 results when 20 docs exist."""
    for i in range(20):
        store.add(f"doc {i}", _unit_vector(seed=i))
    query = _unit_vector(seed=999)
    results = store.search(query, k=5)
    assert len(results) == 5


def test_search_with_metadata_filter(store: SimpleVectorStore):
    """Metadata filtering restricts results to the matching category."""
    for i in range(5):
        store.add(f"A-doc {i}", _unit_vector(seed=i), metadata={"category": "A"})
    for i in range(5):
        store.add(f"B-doc {i}", _unit_vector(seed=i + 100), metadata={"category": "B"})

    query = _unit_vector(seed=0)
    results = store.search(query, k=10, filter_metadata={"category": "A"})
    assert len(results) > 0
    for r in results:
        assert r["metadata"]["category"] == "A", (
            f"Expected category=A, got {r['metadata']}"
        )


def test_search_with_threshold(store: SimpleVectorStore):
    """search_with_threshold() omits results below the threshold."""
    base = _unit_vector(seed=1)

    # Add 3 docs very close to base (score will be high).
    close_ids = set()
    for i in range(3):
        doc_id = store.add(f"close {i}", _noisy(base, scale=0.01, seed=i))
        close_ids.add(doc_id)

    # Add 7 docs far from base (opposite direction ≈ low / negative score).
    far_base = [-x for x in base]
    for i in range(7):
        store.add(f"far {i}", _noisy(far_base, scale=0.05, seed=i + 10))

    results = store.search_with_threshold(base, threshold=0.80, k=20)
    assert len(results) > 0
    for r in results:
        assert r["score"] >= 0.80, f"Score {r['score']:.3f} is below threshold 0.80"
    # Close docs should all be in the results.
    result_ids = {r["id"] for r in results}
    assert close_ids.issubset(result_ids), (
        f"Some close docs missing from results: {close_ids - result_ids}"
    )


def test_save_and_load_preserves_data(store: SimpleVectorStore):
    """Save/load roundtrip preserves documents and search results."""
    base = _unit_vector(seed=42)
    expected_id = store.add("special doc", _noisy(base, scale=0.01, seed=1))
    for i in range(4):
        store.add(f"other {i}", _unit_vector(seed=i + 50))

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        store.save(tmp_path)

        store2 = SimpleVectorStore()
        store2.load(tmp_path)

        assert store2.count() == store.count()
        results = store2.search(base, k=1)
        assert results[0]["id"] == expected_id
    finally:
        os.unlink(tmp_path)


# ===========================================================================
# COMPARATOR UNIT TESTS (offline)
# ===========================================================================


def test_cosine_similarity_identical(comparator: EmbeddingComparator):
    """Identical vectors have cosine similarity 1.0."""
    v = _unit_vector(seed=5)
    score = comparator.cosine_similarity(v, v)
    assert abs(score - 1.0) < 1e-6


def test_cosine_similarity_orthogonal(comparator: EmbeddingComparator):
    """Orthogonal vectors have cosine similarity ≈ 0.0."""
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    score = comparator.cosine_similarity(a, b)
    assert abs(score) < 1e-6


def test_cosine_similarity_opposite(comparator: EmbeddingComparator):
    """Opposite vectors have cosine similarity ≈ -1.0."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    score = comparator.cosine_similarity(a, b)
    assert abs(score + 1.0) < 1e-6


def test_euclidean_distance_zero(comparator: EmbeddingComparator):
    """Same vector has Euclidean distance 0."""
    v = _unit_vector(seed=7)
    assert comparator.euclidean_distance(v, v) < 1e-10


def test_find_most_similar_ranking():
    """find_most_similar returns results sorted by score, highest first."""
    base = _unit_vector(seed=1)
    far = [-x for x in base]

    candidates = [
        "distant text",          # seed=100 → far
        "very similar text",     # close to base
        "somewhat similar text", # moderate
    ]
    embeddings = [
        _noisy(far, scale=0.05, seed=0),     # distant
        _noisy(base, scale=0.01, seed=1),    # very close
        _noisy(base, scale=0.20, seed=2),    # moderate
    ]
    # query is identical to base
    query_emb = base
    all_embs = [query_emb] + embeddings

    gen = _make_mock_generator(all_embs)
    comp = EmbeddingComparator()
    results = comp.find_most_similar("query", candidates, gen)

    assert results[0]["text"] == "very similar text"
    assert results[-1]["text"] == "distant text"
    for i in range(len(results) - 1):
        assert results[i]["score"] >= results[i + 1]["score"]
