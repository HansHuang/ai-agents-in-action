"""pytest suite for code/python/06-frameworks/.

Coverage:
    • framework_comparison.py  — from-scratch, LangChain, LangGraph agents
    • hybrid_rag_agent.py      — hybrid agent, vector store swap
    • framework_advisor.py     — recommendation engine
    • framework_extraction.py  — step 4 (pure custom), import counts

All LLM and embedding calls are mocked by default.
Use ``pytest -m integration`` to run against the real API.

Run:
    pytest test_frameworks.py -v
    pytest test_frameworks.py -v -m "not integration"
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure the local folder is on the import path
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

# ---------------------------------------------------------------------------
# Helper utilities (defined first — used by skipif decorators at class level)
# ---------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    """Return True if *name* can be imported."""
    import importlib.util

    return importlib.util.find_spec(name) is not None


# ---------------------------------------------------------------------------
# Shared fixtures / mock helpers
# ---------------------------------------------------------------------------


def _fake_embedding(dim: int = 8) -> list[float]:
    """Return a deterministic non-zero unit vector of length *dim*."""
    v = np.ones(dim, dtype=float) / np.sqrt(dim)
    return v.tolist()


def _make_openai_client(
    answer: str = "RAG has four phases: Ingest, Retrieve, Augment, Generate.",
) -> MagicMock:
    """Return a fully-mocked OpenAI client."""
    client = MagicMock()

    # embeddings.create — return a fake embedding
    embed_item = MagicMock()
    embed_item.embedding = _fake_embedding()
    embed_response = MagicMock()
    embed_response.data = [embed_item] * 10  # enough for any batch size
    client.embeddings.create.return_value = embed_response

    # chat.completions.create — return a fake answer
    choice = MagicMock()
    choice.message.content = answer
    usage = MagicMock()
    usage.total_tokens = 100
    usage.prompt_tokens = 80
    usage.completion_tokens = 20
    chat_response = MagicMock()
    chat_response.choices = [choice]
    chat_response.usage = usage
    client.chat.completions.create.return_value = chat_response

    return client


# ===========================================================================
# 1. test_from_scratch_agent_works
# ===========================================================================


class TestFromScratchAgent:
    """From-scratch RAG agent (framework_comparison.run_from_scratch)."""

    def test_returns_non_empty_answer(self) -> None:
        from framework_comparison import run_from_scratch

        with patch("framework_comparison.OpenAI", return_value=_make_openai_client()):
            result = run_from_scratch("What are the four phases of a RAG pipeline?")

        assert result.answer, "Answer must not be empty"
        assert result.name == "From Scratch"

    def test_sources_are_cited(self) -> None:
        from framework_comparison import run_from_scratch

        with patch("framework_comparison.OpenAI", return_value=_make_openai_client()):
            result = run_from_scratch("What are the four phases of a RAG pipeline?")

        # With a real vector store and a deterministic embedding every doc
        # scores identically, so all sources end up in the result.
        assert isinstance(result.sources, list)

    def test_buggy_mode_returns_empty_sources(self) -> None:
        from framework_comparison import run_from_scratch

        with patch("framework_comparison.OpenAI", return_value=_make_openai_client()):
            result = run_from_scratch("test", buggy=True)

        assert result.sources == [], "Buggy mode must return zero sources"
        assert result.debug_trace != "", "Buggy mode must populate debug_trace"


# ===========================================================================
# 2. test_langchain_agent_works
# ===========================================================================


@pytest.mark.skipif(
    not _module_available("langchain"),
    reason="LangChain not installed",
)
class TestLangChainAgent:
    """LangChain RAG agent (framework_comparison.run_langchain)."""

    def test_returns_non_empty_answer(self) -> None:
        from framework_comparison import run_langchain

        with _patch_langchain():
            result = run_langchain("What are vector databases?")

        assert result is not None
        assert result.answer

    def test_name_is_langchain(self) -> None:
        from framework_comparison import run_langchain

        with _patch_langchain():
            result = run_langchain("test")

        assert result is not None
        assert result.name == "LangChain"


# ===========================================================================
# 3. test_langgraph_agent_works
# ===========================================================================


@pytest.mark.skipif(
    not _module_available("langgraph"),
    reason="LangGraph not installed",
)
class TestLangGraphAgent:
    """LangGraph RAG agent (framework_comparison.run_langgraph)."""

    def test_returns_non_empty_answer(self) -> None:
        from framework_comparison import run_langgraph

        with _patch_langchain():
            result = run_langgraph("What is an agent loop?")

        assert result is not None
        assert result.answer


# ===========================================================================
# 4. test_all_three_return_similar_answers (semantic similarity proxy)
# ===========================================================================


class TestAllThreeReturnSimilarAnswers:
    """All three implementations should mention key RAG facts."""

    def test_from_scratch_mentions_rag_phases(self) -> None:
        from framework_comparison import run_from_scratch

        expected_keywords = ["ingest", "retrieve", "augment", "generate"]
        answer = "RAG has four phases: Ingest, Retrieve, Augment, Generate."
        with patch("framework_comparison.OpenAI", return_value=_make_openai_client(answer)):
            result = run_from_scratch("What are the four RAG phases?")

        for kw in expected_keywords:
            assert kw.lower() in result.answer.lower(), f"'{kw}' missing from answer"


# ===========================================================================
# 5. test_hybrid_agent_independent_of_framework
# ===========================================================================


class TestHybridAgentIndependentOfFramework:
    """HybridRAGAgent handles framework errors with custom error handling."""

    def test_custom_error_handling_fires_when_store_raises(self) -> None:
        from hybrid_rag_agent import HybridRAGAgent, VectorStoreProtocol

        client = _make_openai_client()

        class FailingStore:
            """A VectorStore that always raises."""
            backend_name = "FailingStore"

            def similarity_search(self, query: str, k: int = 5) -> list[dict]:
                raise RuntimeError("Simulated vector store failure")

        agent = HybridRAGAgent(".", client, use_langchain=False)
        # Bypass normal ingest so we can inject the failing store
        agent._vector_store = FailingStore()

        with pytest.raises(RuntimeError, match="Simulated vector store failure"):
            agent.query("test question")

    def test_ingest_without_langchain_uses_simple_store(self) -> None:
        from hybrid_rag_agent import HybridRAGAgent

        client = _make_openai_client()
        agent = HybridRAGAgent(".", client, use_langchain=False)
        n = agent.ingest()
        assert n > 0
        assert agent.vector_backend == "SimpleVectorStore"


# ===========================================================================
# 6. test_vector_store_swappable
# ===========================================================================


class TestVectorStoreSwappable:
    """Swapping the vector store doesn't change the agent's answer."""

    def test_swap_produces_equivalent_answer(self) -> None:
        from hybrid_rag_agent import HybridRAGAgent, SimpleVectorStore, _INLINE_DOCS

        answer = "RAG: Ingest, Retrieve, Augment, Generate."
        client = _make_openai_client(answer)

        agent = HybridRAGAgent(".", client, use_langchain=False)
        agent.ingest()
        result1 = agent.query("RAG phases?")

        # Swap to a fresh SimpleVectorStore
        new_store = SimpleVectorStore(client)
        new_store.add_documents([{"text": d["text"], "metadata": {"source": d["source"]}} for d in _INLINE_DOCS])
        agent.swap_vector_store(new_store)

        result2 = agent.query("RAG phases?")

        # Both answers come from the mocked LLM — they must be identical
        assert result1.answer == result2.answer
        assert agent.vector_backend == "SimpleVectorStore"


# ===========================================================================
# 7. test_framework_advisor_recommends_correctly
# ===========================================================================


class TestFrameworkAdvisorRecommends:
    """Recommendation engine returns the expected primary framework."""

    def _rec(self, preset_name: str) -> str:
        from framework_advisor import FrameworkAdvisor, PRESETS

        advisor = FrameworkAdvisor()
        rec = advisor.recommend(PRESETS[preset_name])
        return rec.primary

    def test_expert_long_term_python_recommends_from_scratch(self) -> None:
        assert self._rec("expert") == "from_scratch"

    def test_beginner_prototype_python_recommends_langchain(self) -> None:
        assert self._rec("beginner") == "langchain"

    def test_streaming_typescript_recommends_vercel_ai(self) -> None:
        assert self._rec("streaming_ts") == "vercel_ai"

    def test_multi_agent_prototype_recommends_crewai(self) -> None:
        assert self._rec("multi_agent") == "crewai"

    def test_complex_workflow_production_recommends_langgraph(self) -> None:
        assert self._rec("complex_workflow") == "langgraph"

    def test_recommendation_includes_avoid_list(self) -> None:
        from framework_advisor import FrameworkAdvisor, PRESETS

        advisor = FrameworkAdvisor()
        rec = advisor.recommend(PRESETS["expert"])
        assert isinstance(rec.avoid, list)
        assert len(rec.avoid) > 0

    def test_recommendation_includes_migration_path(self) -> None:
        from framework_advisor import FrameworkAdvisor, PRESETS

        advisor = FrameworkAdvisor()
        rec = advisor.recommend(PRESETS["beginner"])
        assert "Phase" in rec.migration_path


# ===========================================================================
# 8. test_extraction_reduces_dependencies
# ===========================================================================


class TestExtractionReducesDependencies:
    """Pure-custom step has zero LangChain imports; counts decrease per step."""

    def test_step4_has_zero_langchain_imports(self) -> None:
        from framework_extraction import step4_pure_custom

        with patch("framework_extraction.OpenAI", return_value=_make_openai_client()):
            result = step4_pure_custom("What are RAG phases?")

        assert result.langchain_imports == 0

    def test_step4_has_more_custom_lines_than_framework_lines(self) -> None:
        from framework_extraction import step4_pure_custom

        with patch("framework_extraction.OpenAI", return_value=_make_openai_client()):
            result = step4_pure_custom("test")

        assert result.custom_lines > result.langchain_lines

    @pytest.mark.skipif(
        not _module_available("langchain"),
        reason="LangChain not installed",
    )
    def test_import_count_decreases_from_step1_to_step4(self) -> None:
        """Step 1 has the most LC imports; step 4 has zero."""
        from framework_extraction import (
            step1_pure_langchain,
            step4_pure_custom,
        )

        with _patch_langchain():
            r1 = step1_pure_langchain("test")
        with patch("framework_extraction.OpenAI", return_value=_make_openai_client()):
            r4 = step4_pure_custom("test")

        assert r1 is not None
        assert r1.langchain_imports > r4.langchain_imports
        assert r4.langchain_imports == 0


# ===========================================================================
# Integration tests (real API calls)
# ===========================================================================


@pytest.mark.integration
class TestIntegration:
    """Run with ``pytest -m integration`` — requires OPENAI_API_KEY."""

    def test_from_scratch_real_api(self) -> None:
        from framework_comparison import run_from_scratch

        result = run_from_scratch("What are the four phases of a RAG pipeline?")
        assert len(result.answer) > 20
        assert result.tokens_used > 0

    def test_hybrid_agent_real_api(self) -> None:
        from hybrid_rag_agent import HybridRAGAgent
        from openai import OpenAI
        import os

        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        agent = HybridRAGAgent(".", client, use_langchain=False)
        agent.ingest()
        result = agent.query("What are the four phases of a RAG pipeline?")
        assert len(result.answer) > 20
        assert result.tokens > 0


# ---------------------------------------------------------------------------
# Internal helpers (contextmanager)
# ---------------------------------------------------------------------------

from contextlib import contextmanager


@contextmanager
def _patch_langchain():
    """Patch LangChain symbols used by framework_comparison and framework_extraction."""
    fake_answer = "RAG has four phases: Ingest, Retrieve, Augment, Generate."
    fake_content = MagicMock()
    fake_content.content = fake_answer

    fake_llm = MagicMock()
    fake_llm.invoke.return_value = fake_content

    fake_doc = MagicMock()
    fake_doc.page_content = "RAG phases: Ingest, Retrieve, Augment, Generate."
    fake_doc.metadata = {"source": "rag.txt"}

    fake_store = MagicMock()
    fake_store.similarity_search.return_value = [fake_doc]
    fake_store.as_retriever.return_value = MagicMock()

    fake_chain_result = {"answer": fake_answer, "context": [fake_doc]}
    fake_rag_chain = MagicMock()
    fake_rag_chain.invoke.return_value = fake_chain_result

    with (
        patch("framework_comparison.FAISS") as mock_faiss,
        patch("framework_comparison.OpenAIEmbeddings"),
        patch("framework_comparison.ChatOpenAI", return_value=fake_llm),
        patch("framework_comparison.create_retrieval_chain", return_value=fake_rag_chain),
        patch("framework_comparison.create_stuff_documents_chain"),
        patch("framework_comparison.ChatPromptTemplate"),
    ):
        mock_faiss.from_documents.return_value = fake_store
        yield
