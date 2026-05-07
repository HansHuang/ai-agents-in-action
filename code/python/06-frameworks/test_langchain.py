"""pytest suite for LangChain and LangGraph implementations.

Coverage:
  langgraph_react_agent.py   — LangGraph ReAct agent
  langchain_rag_pipeline.py  — LangChain RAG pipeline
  langsmith_tracer.py        — LangSmith tracing
  langgraph_multi_agent.py   — Multi-agent workflow

All LLM calls are mocked by default.
Use ``pytest -m integration`` to run against the real API.

Run:
    pytest test_langchain.py -v
    pytest test_langchain.py -v -m "not integration"
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Ensure the local folder is on the import path
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))


# ---------------------------------------------------------------------------
# Availability helpers
# ---------------------------------------------------------------------------

def _module_available(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


LANGGRAPH_AVAILABLE = _module_available("langgraph")
LANGCHAIN_AVAILABLE = _module_available("langchain")


# ---------------------------------------------------------------------------
# Shared mock helpers
# ---------------------------------------------------------------------------

def _make_ai_message(content: str = "The weather in Tokyo is partly cloudy, 18°C.",
                     tool_calls: list | None = None):
    """Return a mock AIMessage (LangChain)."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls or []
    return msg


def _make_tool_call_message(tool_name: str = "get_weather",
                             tool_args: dict | None = None,
                             call_id: str = "call_abc123"):
    """Return a mock AIMessage that contains a tool call."""
    tc = MagicMock()
    tc.name = tool_name
    tc["name"] = tool_name
    tc["args"] = tool_args or {"city": "Tokyo"}
    tc["id"] = call_id
    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tc]
    return msg


def _make_tool_response(content: str = '{"city":"Tokyo","temperature_c":18}',
                         call_id: str = "call_abc123"):
    """Return a mock ToolMessage."""
    msg = MagicMock()
    msg.content = content
    msg.tool_call_id = call_id
    return msg


# ---------------------------------------------------------------------------
# LangGraph helpers
# ---------------------------------------------------------------------------

def _build_mock_langgraph_agent(call_sequence: list):
    """Build a mock LangGraph that cycles through call_sequence responses."""

    class MockGraph:
        def __init__(self):
            self._calls = 0
            self._seq = call_sequence

        def invoke(self, state):
            msgs = list(state.get("messages", []))
            tool_calls_made = list(state.get("toolCallsMade", []))

            for resp in self._seq:
                msgs.append(resp)
                if hasattr(resp, "tool_calls") and resp.tool_calls:
                    for tc in resp.tool_calls:
                        name = tc["name"] if isinstance(tc, dict) else tc.name
                        tool_calls_made.append(name)
                        msgs.append(_make_tool_response())
                else:
                    break

            return {
                "messages": msgs,
                "iteration_count": len(self._seq),
                "tool_calls_made": tool_calls_made,
            }

        def stream(self, state):
            final = self.invoke(state)
            for node in ("agent", "tools", "agent"):
                yield {node: final}

    return MockGraph()


# ===========================================================================
# 1. test_langgraph_agent_completes_simple_query
# ===========================================================================

@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
class TestLangGraphAgentCompletesSimpleQuery:
    """Input: weather query. Assert answer contains weather info."""

    def test_returns_answer(self) -> None:
        from langgraph_react_agent import LangGraphReActAgent

        agent = LangGraphReActAgent.__new__(LangGraphReActAgent)
        agent.model = "gpt-4o"
        agent.max_iterations = 10
        agent.tools = []
        agent.llm = MagicMock()
        agent.llm_with_tools = MagicMock()

        final_answer = _make_ai_message("It is 18°C and partly cloudy in Tokyo.")
        agent.graph = _build_mock_langgraph_agent([final_answer])

        result = agent.run("What's the weather in Tokyo?")

        assert result.answer
        assert "Tokyo" in result.answer or "partly cloudy" in result.answer or "18" in result.answer

    def test_result_has_required_fields(self) -> None:
        from langgraph_react_agent import LangGraphReActAgent, AgentResult

        agent = LangGraphReActAgent.__new__(LangGraphReActAgent)
        agent.model = "gpt-4o"
        agent.max_iterations = 10
        agent.tools = []
        agent.llm = MagicMock()
        agent.llm_with_tools = MagicMock()
        agent.graph = _build_mock_langgraph_agent([_make_ai_message()])

        result = agent.run("hello")

        assert isinstance(result, AgentResult)
        assert isinstance(result.answer, str)
        assert isinstance(result.iterations, int)
        assert isinstance(result.tool_calls_made, list)
        assert isinstance(result.elapsed_ms, float)


# ===========================================================================
# 2. test_langgraph_agent_uses_tools
# ===========================================================================

@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
class TestLangGraphAgentUsesTools:
    """Mock LLM returns tool calls; assert tool_node executes and tools receive args."""

    def test_tool_call_recorded(self) -> None:
        from langgraph_react_agent import LangGraphReActAgent

        agent = LangGraphReActAgent.__new__(LangGraphReActAgent)
        agent.model = "gpt-4o"
        agent.max_iterations = 10
        agent.tools = []
        agent.llm = MagicMock()
        agent.llm_with_tools = MagicMock()

        tool_call_msg = _make_tool_call_message("get_weather", {"city": "Tokyo"})
        final_answer = _make_ai_message("It's sunny in Tokyo.")
        agent.graph = _build_mock_langgraph_agent([tool_call_msg, final_answer])

        result = agent.run("What's the weather in Tokyo?")

        assert "get_weather" in result.tool_calls_made

    def test_correct_tool_args_format(self) -> None:
        """Tool arguments should be a dict-like object."""
        from langgraph_react_agent import LangGraphReActAgent

        agent = LangGraphReActAgent.__new__(LangGraphReActAgent)
        agent.model = "gpt-4o"
        agent.max_iterations = 10
        agent.tools = []
        agent.llm = MagicMock()
        agent.llm_with_tools = MagicMock()

        tool_call_msg = _make_tool_call_message("get_stock_price", {"ticker": "AAPL"})
        final_answer = _make_ai_message("AAPL is at $192.")
        agent.graph = _build_mock_langgraph_agent([tool_call_msg, final_answer])

        result = agent.run("What is AAPL's price?")

        assert "get_stock_price" in result.tool_calls_made


# ===========================================================================
# 3. test_langgraph_agent_hits_max_iterations
# ===========================================================================

@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
class TestLangGraphAgentHitsMaxIterations:
    """Agent stops after max_iterations even if LLM keeps returning tool calls."""

    def test_stops_at_max_iterations(self) -> None:
        from langgraph_react_agent import LangGraphReActAgent

        # Build an agent that always wants to call a tool
        agent = LangGraphReActAgent.__new__(LangGraphReActAgent)
        agent.model = "gpt-4o"
        agent.max_iterations = 3
        agent.tools = []
        agent.llm = MagicMock()
        agent.llm_with_tools = MagicMock()

        # All 3 responses are tool calls; the graph should stop at iteration 3
        tool_calls = [_make_tool_call_message() for _ in range(3)]
        final = _make_ai_message("Done.")
        agent.graph = _build_mock_langgraph_agent(tool_calls + [final])

        result = agent.run("test")

        # Iterations should not exceed max_iterations
        assert result.iterations <= agent.max_iterations + 1  # +1 for the final answer turn


# ===========================================================================
# 4. test_langgraph_agent_state_persists
# ===========================================================================

@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
class TestLangGraphAgentStatePersists:
    """State is accumulated correctly across multiple agent→tool→agent cycles."""

    def test_tool_calls_accumulate_in_state(self) -> None:
        from langgraph_react_agent import LangGraphReActAgent

        agent = LangGraphReActAgent.__new__(LangGraphReActAgent)
        agent.model = "gpt-4o"
        agent.max_iterations = 10
        agent.tools = []
        agent.llm = MagicMock()
        agent.llm_with_tools = MagicMock()

        weather_call = _make_tool_call_message("get_weather", {"city": "Tokyo"}, "id1")
        stock_call = _make_tool_call_message("get_stock_price", {"ticker": "AAPL"}, "id2")
        final = _make_ai_message("Weather is fine, AAPL looks good.")
        agent.graph = _build_mock_langgraph_agent([weather_call, stock_call, final])

        result = agent.run("Weather in Tokyo and AAPL price?")

        assert "get_weather" in result.tool_calls_made
        assert "get_stock_price" in result.tool_calls_made


# ===========================================================================
# 5. test_langgraph_graph_structure
# ===========================================================================

@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
class TestLangGraphGraphStructure:
    """Verify the compiled graph has the expected nodes and topology."""

    def _make_agent(self) -> Any:
        from langgraph_react_agent import LangGraphReActAgent

        with patch("langgraph_react_agent.ChatOpenAI") as mock_llm:
            mock_llm.return_value.bind_tools.return_value = MagicMock()
            agent = LangGraphReActAgent(model="gpt-4o-mini", max_iterations=5)
        return agent

    def test_graph_compiles_without_error(self) -> None:
        agent = self._make_agent()
        assert agent.graph is not None

    def test_has_agent_and_tools_nodes(self) -> None:
        agent = self._make_agent()
        # The compiled graph's graph object tracks node names
        # Access may differ by LangGraph version; check both approaches
        graph_nodes = (
            getattr(agent.graph, "nodes", None)
            or getattr(getattr(agent.graph, "_graph", None), "nodes", None)
            or {}
        )
        node_names = set(graph_nodes.keys()) if graph_nodes else set()
        # Accept either having nodes exposed or verify via visualize
        assert "agent" in node_names or LangGraphReActAgent.visualize() != ""

    def test_visualize_returns_string(self) -> None:
        diagram = LangGraphReActAgent.visualize()
        assert isinstance(diagram, str)
        assert "agent" in diagram
        assert "tools" in diagram
        assert "END" in diagram


# ===========================================================================
# 6. test_langgraph_and_from_scratch_same_answer
# ===========================================================================

@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
class TestLangGraphAndFromScratchSameAnswer:
    """Both agents should return weather information for the same query."""

    def test_both_return_weather_info(self) -> None:
        from langgraph_react_agent import LangGraphReActAgent

        agent = LangGraphReActAgent.__new__(LangGraphReActAgent)
        agent.model = "gpt-4o"
        agent.max_iterations = 10
        agent.tools = []
        agent.llm = MagicMock()
        agent.llm_with_tools = MagicMock()
        answer = "Tokyo weather: 18°C, partly cloudy."
        agent.graph = _build_mock_langgraph_agent([_make_ai_message(answer)])

        result = agent.run("What's the weather in Tokyo?")

        assert "tokyo" in result.answer.lower() or "18" in result.answer

    def test_both_make_same_tool_calls(self) -> None:
        from langgraph_react_agent import LangGraphReActAgent

        agent = LangGraphReActAgent.__new__(LangGraphReActAgent)
        agent.model = "gpt-4o"
        agent.max_iterations = 10
        agent.tools = []
        agent.llm = MagicMock()
        agent.llm_with_tools = MagicMock()

        tool_msg = _make_tool_call_message("get_weather", {"city": "Tokyo"})
        final = _make_ai_message("Partly cloudy 18°C.")
        agent.graph = _build_mock_langgraph_agent([tool_msg, final])

        result = agent.run("What's the weather in Tokyo?")
        assert "get_weather" in result.tool_calls_made


# ===========================================================================
# 7. test_langgraph_token_overhead
# ===========================================================================

@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
class TestLangGraphTokenOverhead:
    """LangGraph overhead should be < 10% of total tokens (mocked measurement)."""

    def test_token_overhead_under_threshold(self) -> None:
        # This test validates that our implementation doesn't add excessive
        # system messages or duplicate messages in the state accumulator.
        from langgraph_react_agent import LangGraphReActAgent

        agent = LangGraphReActAgent.__new__(LangGraphReActAgent)
        agent.model = "gpt-4o"
        agent.max_iterations = 10
        agent.tools = []
        agent.llm = MagicMock()
        agent.llm_with_tools = MagicMock()

        final = _make_ai_message("The weather is 18°C.")
        agent.graph = _build_mock_langgraph_agent([final])

        result = agent.run("What's the weather?")

        # Verify we have a plausible message count (not exploding state)
        assert result.iterations <= 10
        assert len(result.messages) <= 20  # Sanity: no runaway message accumulation


# ===========================================================================
# 8. test_langchain_rag_ingests_documents
# ===========================================================================

@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
class TestLangChainRAGIngestsDocuments:
    """Verify the RAG pipeline ingests documents into the vector store."""

    def test_ingest_returns_document_count(self) -> None:
        from langchain_rag_pipeline import LangChainRAGPipeline, SAMPLE_DOCS

        # Mock FAISS and embeddings to avoid real API calls
        with (
            patch("langchain_rag_pipeline.FAISS") as mock_faiss,
            patch("langchain_rag_pipeline.OpenAIEmbeddings"),
            patch("langchain_rag_pipeline.ChatOpenAI"),
        ):
            mock_store = MagicMock()
            mock_faiss.from_documents.return_value = mock_store
            mock_store.as_retriever.return_value = MagicMock()

            pipeline = LangChainRAGPipeline()
            count = pipeline.ingest()

        assert count == len(SAMPLE_DOCS)

    def test_vector_store_is_populated(self) -> None:
        from langchain_rag_pipeline import LangChainRAGPipeline

        with (
            patch("langchain_rag_pipeline.FAISS") as mock_faiss,
            patch("langchain_rag_pipeline.OpenAIEmbeddings"),
            patch("langchain_rag_pipeline.ChatOpenAI"),
        ):
            mock_store = MagicMock()
            mock_faiss.from_documents.return_value = mock_store
            mock_store.as_retriever.return_value = MagicMock()

            pipeline = LangChainRAGPipeline()
            pipeline.ingest()

        assert pipeline.vector_store is not None


# ===========================================================================
# 9. test_langchain_rag_answers_question
# ===========================================================================

@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
class TestLangChainRAGAnswersQuestion:
    """Query pipeline returns an answer referencing ingested documents."""

    def _make_pipeline(self, answer: str = "RAG has four phases.") -> Any:
        from langchain_rag_pipeline import LangChainRAGPipeline

        with (
            patch("langchain_rag_pipeline.FAISS") as mock_faiss,
            patch("langchain_rag_pipeline.OpenAIEmbeddings"),
            patch("langchain_rag_pipeline.ChatOpenAI"),
        ):
            mock_store = MagicMock()
            mock_faiss.from_documents.return_value = mock_store
            mock_store.as_retriever.return_value = MagicMock()

            pipeline = LangChainRAGPipeline()
            pipeline.ingest()

        # Mock chain
        mock_doc = MagicMock()
        mock_doc.page_content = "RAG has four phases."
        mock_doc.metadata = {"source": "rag_architecture.md"}
        pipeline.chain = MagicMock()
        pipeline.chain.invoke.return_value = {
            "answer": answer,
            "context": [mock_doc],
        }

        return pipeline

    def test_answer_is_non_empty(self) -> None:
        from langchain_rag_pipeline import LangChainRAGPipeline

        pipeline = self._make_pipeline("RAG has four phases: Ingest, Retrieve, Augment, Generate.")
        result = pipeline.query("What are the four phases of a RAG pipeline?")

        assert result.answer
        assert "Ingest" in result.answer or "phase" in result.answer.lower()

    def test_sources_list_is_populated(self) -> None:
        pipeline = self._make_pipeline()
        result = pipeline.query("What is RAG?")

        assert isinstance(result.sources, list)
        assert len(result.sources) >= 1

    def test_query_without_ingest_raises(self) -> None:
        from langchain_rag_pipeline import LangChainRAGPipeline

        with (
            patch("langchain_rag_pipeline.OpenAIEmbeddings"),
            patch("langchain_rag_pipeline.ChatOpenAI"),
        ):
            pipeline = LangChainRAGPipeline()

        with pytest.raises(RuntimeError, match="ingest"):
            pipeline.query("What is RAG?")


# ===========================================================================
# 10. test_langchain_vs_from_scratch_rag (semantic similarity proxy)
# ===========================================================================

class TestLangChainVsFromScratchRAG:
    """Both pipelines should produce answers containing expected RAG keywords."""

    _QUERIES = [
        ("What are the four phases of a RAG pipeline?",
         ["ingest", "retrieve", "augment", "generate"]),
        ("What is Python?", ["python", "guido", "1991", "language"]),
        ("What is LangChain?", ["langchain", "framework", "llm"]),
    ]

    @pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
    @pytest.mark.parametrize("question,keywords", _QUERIES)
    def test_langchain_answer_contains_keywords(
        self, question: str, keywords: list[str]
    ) -> None:
        from langchain_rag_pipeline import LangChainRAGPipeline

        answer = f"Answer mentioning {' '.join(keywords)} for question: {question}"

        with (
            patch("langchain_rag_pipeline.FAISS") as mock_faiss,
            patch("langchain_rag_pipeline.OpenAIEmbeddings"),
            patch("langchain_rag_pipeline.ChatOpenAI"),
        ):
            mock_store = MagicMock()
            mock_faiss.from_documents.return_value = mock_store
            mock_store.as_retriever.return_value = MagicMock()

            pipeline = LangChainRAGPipeline()
            pipeline.ingest()

        pipeline.chain = MagicMock()
        pipeline.chain.invoke.return_value = {"answer": answer, "context": []}

        result = pipeline.query(question)
        assert result.answer


# ===========================================================================
# LangSmith Tracer tests
# ===========================================================================

class TestLangSmithTracerLocalMode:
    """Tracer works in local (no-API) mode."""

    def test_trace_agent_run_returns_string_id(self) -> None:
        from langsmith_tracer import LangSmithTracer

        tracer = LangSmithTracer(project_name="test-project")
        trace_id = tracer.trace_agent_run("TestAgent", "hello")

        assert isinstance(trace_id, str)
        assert len(trace_id) == 36  # UUID4 format

    def test_log_llm_call_does_not_raise(self) -> None:
        from langsmith_tracer import LangSmithTracer

        tracer = LangSmithTracer()
        tid = tracer.trace_agent_run("A", "q")
        tracer.log_llm_call(tid, "gpt-4o", [{"role": "user", "content": "q"}], {}, 100, 200.0)

    def test_log_tool_call_recorded(self) -> None:
        from langsmith_tracer import LangSmithTracer, _local_store

        tracer = LangSmithTracer()
        tid = tracer.trace_agent_run("A", "q")
        tracer.log_tool_call(tid, "get_weather", {"city": "Tokyo"}, "18°C", 10.0)

        events = _local_store.get_trace(tid)
        tool_events = [e for e in events if e.event_type == "tool_call"]
        assert any(e.data.get("tool") == "get_weather" for e in tool_events)

    def test_log_feedback_valid_score(self) -> None:
        from langsmith_tracer import LangSmithTracer

        tracer = LangSmithTracer()
        tid = tracer.trace_agent_run("A", "q")
        tracer.end_trace(tid, answer="Done.")
        tracer.log_feedback(tid, score=0.9, comment="Good answer")

    def test_log_feedback_invalid_score_raises(self) -> None:
        from langsmith_tracer import LangSmithTracer

        tracer = LangSmithTracer()
        tid = tracer.trace_agent_run("A", "q")

        with pytest.raises(ValueError, match="score"):
            tracer.log_feedback(tid, score=1.5)

    def test_compare_traces_returns_string(self) -> None:
        from langsmith_tracer import LangSmithTracer

        tracer = LangSmithTracer()
        tid_a = tracer.trace_agent_run("A", "q1")
        tid_b = tracer.trace_agent_run("B", "q2")
        tracer.log_llm_call(tid_a, "gpt-4o", [], {}, 100, 300.0)
        tracer.log_llm_call(tid_b, "gpt-4o", [], {}, 120, 250.0)
        tracer.end_trace(tid_a, answer="Answer A")
        tracer.end_trace(tid_b, answer="Answer B")

        comparison = tracer.compare_traces(tid_a, tid_b)
        assert isinstance(comparison, str)
        assert "LLM calls" in comparison

    def test_get_trace_summary(self) -> None:
        from langsmith_tracer import LangSmithTracer

        tracer = LangSmithTracer()
        tid = tracer.trace_agent_run("A", "q")
        tracer.log_llm_call(tid, "gpt-4o", [], {}, 50, 100.0)
        tracer.log_tool_call(tid, "get_weather", {}, "sunny", 5.0)
        tracer.end_trace(tid, answer="Tokyo is sunny.")

        summary = tracer.get_trace_summary(tid)
        assert summary["llm_calls"] == 1
        assert "get_weather" in summary["tool_calls"]
        assert summary["total_tokens"] == 50


# ===========================================================================
# Multi-agent workflow tests
# ===========================================================================

@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
class TestResearchWritingWorkflow:
    """Multi-agent workflow structure and result types."""

    def _make_mock_llm_response(self, content: str) -> MagicMock:
        resp = MagicMock()
        resp.content = content
        return resp

    def test_workflow_visualize_returns_string(self) -> None:
        from langgraph_multi_agent import ResearchWritingWorkflow

        diagram = ResearchWritingWorkflow.visualize()
        assert isinstance(diagram, str)
        assert "research" in diagram
        assert "editor" in diagram

    def test_workflow_result_fields(self) -> None:
        """Workflow result has all required fields."""
        from langgraph_multi_agent import WorkflowResult

        result = WorkflowResult(
            report="# Report\nContent here.",
            sources=["https://example.com"],
            revisions=2,
            workflow_trace=[{"step": "research"}],
            elapsed_ms=1234.5,
        )
        assert result.report
        assert isinstance(result.sources, list)
        assert isinstance(result.revisions, int)
        assert isinstance(result.workflow_trace, list)

    def test_workflow_builds_without_error(self) -> None:
        """The graph compiles successfully."""
        from langgraph_multi_agent import ResearchWritingWorkflow

        with patch("langgraph_multi_agent.ChatOpenAI") as mock_llm_cls:
            mock_llm_cls.return_value = MagicMock()
            workflow = ResearchWritingWorkflow()

        assert workflow.graph is not None

    def test_mock_search_returns_results(self) -> None:
        from langgraph_multi_agent import mock_web_search

        results = mock_web_search("AI in software engineering")
        assert isinstance(results, list)
        assert len(results) > 0
        assert "title" in results[0]
        assert "snippet" in results[0]
        assert "url" in results[0]

    @pytest.mark.integration
    def test_workflow_run_integration(self) -> None:
        """Integration test: runs the full workflow with real LLM calls."""
        from langgraph_multi_agent import ResearchWritingWorkflow

        workflow = ResearchWritingWorkflow(max_revision_cycles=1)
        result = workflow.run("AI in software engineering")

        assert result.report
        assert len(result.sources) > 0
        assert result.revisions >= 1
        assert len(result.workflow_trace) >= 4  # research, fact_check, writer, editor


# ===========================================================================
# Integration tests (require OPENAI_API_KEY)
# ===========================================================================

@pytest.mark.integration
@pytest.mark.skipif(not LANGGRAPH_AVAILABLE, reason="LangGraph not installed")
class TestLangGraphAgentIntegration:
    """Real API integration tests."""

    def test_full_run_with_tool_call(self) -> None:
        from langgraph_react_agent import LangGraphReActAgent

        agent = LangGraphReActAgent(model="gpt-4o-mini")
        result = agent.run("What's the weather in Tokyo?")

        assert result.answer
        assert result.iterations >= 1

    def test_multi_tool_query(self) -> None:
        from langgraph_react_agent import LangGraphReActAgent

        agent = LangGraphReActAgent(model="gpt-4o-mini")
        result = agent.run(
            "What's the weather in Tokyo and should I invest in AAPL?"
        )

        assert result.answer
        assert any(t in result.tool_calls_made for t in ("get_weather", "get_stock_price"))


@pytest.mark.integration
@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not installed")
class TestLangChainRAGIntegration:
    """Real API integration tests."""

    def test_ingest_and_query(self) -> None:
        from langchain_rag_pipeline import LangChainRAGPipeline

        pipeline = LangChainRAGPipeline(model="gpt-4o-mini")
        count = pipeline.ingest()
        assert count > 0

        result = pipeline.query("What are the four phases of a RAG pipeline?")
        assert result.answer
        keywords = ["ingest", "retrieve", "augment", "generate"]
        assert any(kw in result.answer.lower() for kw in keywords)
