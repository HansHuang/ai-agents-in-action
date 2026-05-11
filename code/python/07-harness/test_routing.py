"""
Pytest test suite for the hybrid routing system.

Run with: pytest test_routing.py -v

All LLM calls are mocked. Deterministic patterns run with real regex.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from hybrid_router import (
    DeterministicRouter,
    EscalatingRouter,
    HandlerConfig,
    HandlerRegistry,
    HandlerResponse,
    HybridRouter,
    LLMRouter,
    RouteResult,
    RoutingEvaluator,
    RoutingTestCase,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_route_result(intent: str, confidence: float = 0.88, method: str = "llm") -> RouteResult:
    return RouteResult(intent=intent, confidence=confidence, method=method)


async def _noop_handler(user_input, history, config) -> HandlerResponse:
    return HandlerResponse(content="ok", handler_used="noop", metadata={})


# ---------------------------------------------------------------------------
# DeterministicRouter tests
# ---------------------------------------------------------------------------


class TestDeterministicRouter:
    def setup_method(self):
        self.router = DeterministicRouter()

    def test_detects_greeting(self):
        result = self.router.classify("hi there")
        assert result is not None
        assert result.intent == "greeting"

    def test_detects_goodbye(self):
        result = self.router.classify("bye, talk later")
        assert result is not None
        assert result.intent == "goodbye"

    def test_detects_thanks(self):
        result = self.router.classify("thank you so much")
        assert result is not None
        assert result.intent == "thanks"

    def test_detects_reset(self):
        result = self.router.classify("let's start over")
        assert result is not None
        assert result.intent == "reset"

    def test_detects_weather(self):
        result = self.router.classify("what's the weather in Paris?")
        assert result is not None
        assert result.intent == "weather"

    def test_detects_stock(self):
        result = self.router.classify("AAPL stock price today")
        assert result is not None
        assert result.intent == "stock"

    def test_detects_order(self):
        result = self.router.classify("where is my order #12345?")
        assert result is not None
        assert result.intent == "order_lookup"

    def test_detects_return(self):
        result = self.router.classify("I want to return my purchase")
        assert result is not None
        assert result.intent == "return_request"

    def test_detects_billing(self):
        result = self.router.classify("I have a question about my invoice")
        assert result is not None
        assert result.intent == "billing"

    def test_detects_technical(self):
        result = self.router.classify("the app is not working")
        assert result is not None
        assert result.intent == "technical_support"

    def test_detects_account(self):
        result = self.router.classify("I need to change my password")
        assert result is not None
        assert result.intent == "account"

    def test_no_match_returns_none(self):
        result = self.router.classify("quantum computing applications in medicine")
        assert result is None

    def test_case_insensitive(self):
        result = self.router.classify("WHAT'S THE WEATHER TODAY")
        assert result is not None
        assert result.intent == "weather"

    def test_confidence_single_match(self):
        result = self.router.classify("good morning")
        assert result is not None
        assert result.confidence == 0.85

    def test_confidence_multiple_matches(self):
        # "thanks" and "billing" patterns can both match "thank you for the invoice"
        # but we just verify multiple-match confidence is 0.65 when it happens
        router = DeterministicRouter(
            patterns={
                "a": [r"\bfoo\b"],
                "b": [r"\bfoo\b"],
            }
        )
        result = router.classify("foo bar")
        assert result is not None
        assert result.confidence == 0.65

    def test_multiple_matches_uses_longest(self):
        # a_long has a longer pattern; when both match, a_long should win
        router = DeterministicRouter(
            patterns={
                "short_intent": [r"\bhello\b"],
                "long_intent": [r"\bhello\b and this is a much longer pattern string"],
            }
        )
        # Only "short_intent" can actually match since the longer pattern
        # requires additional text.  The point is that the router selects
        # the intent whose matching pattern string is longer.
        result = router.classify("hello and this is a much longer pattern string here")
        assert result is not None
        assert result.intent == "long_intent"

    def test_custom_patterns(self):
        router = DeterministicRouter(patterns={"custom": [r"\bcustom\b"]})
        result = router.classify("this is a custom message")
        assert result is not None
        assert result.intent == "custom"


# ---------------------------------------------------------------------------
# HybridRouter tests
# ---------------------------------------------------------------------------


class TestHybridRouter:
    @pytest.mark.anyio
    async def test_deterministic_first_no_llm_called(self):
        """Clear deterministic match → LLM should NOT be called."""
        router = HybridRouter()
        llm_called = False

        async def _no_llm(*a, **kw):
            nonlocal llm_called
            llm_called = True
            return make_route_result("simple_chat")

        router.llm.classify = _no_llm

        result = await router.route("hello there")
        assert result.intent == "greeting"
        assert result.method == "deterministic"
        assert not llm_called

    @pytest.mark.anyio
    async def test_llm_fallback_on_no_pattern_match(self):
        """No deterministic match → LLM must be called."""
        router = HybridRouter()
        llm_called = False

        async def _stub_llm(user_input, history=None):
            nonlocal llm_called
            llm_called = True
            return make_route_result("knowledge_question")

        router.llm.classify = _stub_llm

        result = await router.route("what is the meaning of life?")
        assert llm_called
        assert result.method == "llm"

    @pytest.mark.anyio
    async def test_llm_used_when_llm_confidence_higher(self):
        """Deterministic low confidence + LLM high confidence → LLM wins."""
        router = HybridRouter()

        async def _stub_llm(user_input, history=None):
            return make_route_result("agent_task", confidence=0.95)

        router.llm.classify = _stub_llm

        # Inject low-confidence deterministic result
        original_classify = router.deterministic.classify
        router.deterministic.classify = lambda _: make_route_result(
            "greeting", confidence=0.65, method="deterministic"
        )

        result = await router.route("hello there, book me a flight")
        assert result.intent == "agent_task"
        assert result.method == "llm"

        router.deterministic.classify = original_classify

    @pytest.mark.anyio
    async def test_deterministic_wins_over_low_confidence_llm(self):
        """Deterministic result outranks LLM when LLM confidence is lower."""
        router = HybridRouter()

        async def _low_llm(user_input, history=None):
            return make_route_result("out_of_scope", confidence=0.4)

        router.llm.classify = _low_llm

        result = await router.route("thanks a lot!")
        assert result.intent == "thanks"

    @pytest.mark.anyio
    async def test_metrics_recorded(self):
        router = HybridRouter()
        await router.route("hello")
        metrics = router.get_metrics()
        assert metrics["total_routed"] == 1
        assert "intent_distribution" in metrics


# ---------------------------------------------------------------------------
# EscalatingRouter tests
# ---------------------------------------------------------------------------


def _make_registry_with_handlers(handlers: dict) -> HandlerRegistry:
    registry = HandlerRegistry()

    async def _fallback(user_input, history, config):
        return HandlerResponse(content="fallback", handler_used="simple_chat", metadata={})

    registry.register(
        "simple_chat", _fallback, HandlerConfig()
    )

    for intent, fn in handlers.items():
        registry.register(intent, fn, HandlerConfig())

    return registry


class TestEscalatingRouter:
    @pytest.mark.anyio
    async def test_rag_no_results_escalates(self):
        """RAG returning 0 documents triggers escalation to agent_task."""

        async def _rag_handler(user_input, history, config):
            return HandlerResponse(
                content="I couldn't find anything.",
                handler_used="rag",
                metadata={"documents_found": 0},
            )

        async def _agent_handler(user_input, history, config):
            return HandlerResponse(
                content="Agent result",
                handler_used="agent_loop",
                metadata={},
            )

        registry = _make_registry_with_handlers(
            {"knowledge_question": _rag_handler, "agent_task": _agent_handler}
        )

        router = HybridRouter()

        async def _llm_classify(user_input, history=None):
            return make_route_result("knowledge_question", confidence=0.9)

        router.llm.classify = _llm_classify

        escrouter = EscalatingRouter(router, registry)
        resp = await escrouter.handle("tell me about returns")
        assert resp.handler_used == "agent_loop"
        assert resp.metadata.get("escalated_from") == "knowledge_question"

    @pytest.mark.anyio
    async def test_agent_max_iterations_escalates(self):
        """Agent reaching max iterations → escalation to human_escalation."""

        async def _agent_handler(user_input, history, config):
            return HandlerResponse(
                content="I reached the maximum number of steps.",
                handler_used="agent_loop",
                metadata={"iterations": 10},
            )

        async def _human_handler(user_input, history, config):
            return HandlerResponse(
                content="Connecting you to a human.",
                handler_used="escalation",
                metadata={},
            )

        registry = _make_registry_with_handlers(
            {"agent_task": _agent_handler, "human_escalation": _human_handler}
        )

        router = HybridRouter()

        async def _llm_classify(user_input, history=None):
            return make_route_result("agent_task", confidence=0.9)

        router.llm.classify = _llm_classify

        escrouter = EscalatingRouter(router, registry)
        resp = await escrouter.handle("do something complex")
        assert resp.handler_used == "escalation"
        assert resp.metadata.get("escalated_from") == "agent_task"

    @pytest.mark.anyio
    async def test_handler_exception_escalates(self):
        """Exception in primary handler → escalation."""

        async def _broken_handler(user_input, history, config):
            raise RuntimeError("broken")

        async def _fallback_handler(user_input, history, config):
            return HandlerResponse(
                content="Escalated response",
                handler_used="knowledge_question",
                metadata={},
            )

        registry = _make_registry_with_handlers(
            {"simple_chat": _broken_handler, "knowledge_question": _fallback_handler}
        )

        router = HybridRouter()
        # Force deterministic to return simple_chat
        router.deterministic.classify = lambda _: make_route_result(
            "simple_chat", confidence=0.9, method="deterministic"
        )

        escrouter = EscalatingRouter(router, registry)
        resp = await escrouter.handle("hello")
        assert resp.handler_used == "knowledge_question"

    @pytest.mark.anyio
    async def test_escalation_chain_exhausted(self):
        """When all escalation handlers fail, return graceful error message."""

        async def _broken(user_input, history, config):
            raise RuntimeError("all broken")

        registry = _make_registry_with_handlers(
            {
                "agent_task": _broken,
                "human_escalation": _broken,
                "simple_chat": _broken,
            }
        )

        router = HybridRouter()

        async def _llm_classify(user_input, history=None):
            return make_route_result("agent_task", confidence=0.9)

        router.llm.classify = _llm_classify

        escrouter = EscalatingRouter(router, registry)
        resp = await escrouter.handle("do something")
        assert resp.handler_used == "escalation_fallback"
        assert resp.metadata.get("escalation_chain_exhausted") is True

    @pytest.mark.anyio
    async def test_escalation_context_injected(self):
        """Escalated request includes reference to previous attempt."""
        received_inputs = []

        async def _primary(user_input, history, config):
            return HandlerResponse(
                content="I'm not sure about that.",
                handler_used="simple_chat",
                metadata={},
            )

        async def _secondary(user_input, history, config):
            received_inputs.append(user_input)
            return HandlerResponse(content="ok", handler_used="knowledge_question", metadata={})

        registry = _make_registry_with_handlers(
            {"simple_chat": _primary, "knowledge_question": _secondary}
        )

        router = HybridRouter()
        router.deterministic.classify = lambda _: make_route_result(
            "simple_chat", confidence=0.9, method="deterministic"
        )

        escrouter = EscalatingRouter(router, registry)
        await escrouter.handle("test question")

        assert len(received_inputs) == 1
        assert "simple_chat" in received_inputs[0]


# ---------------------------------------------------------------------------
# HandlerRegistry tests
# ---------------------------------------------------------------------------


class TestHandlerRegistry:
    def test_registry_returns_correct_handler(self):
        registry = HandlerRegistry()
        registry.register("simple_chat", _noop_handler, HandlerConfig())
        registry.register("agent_task", _noop_handler, HandlerConfig(model="gpt-4o"))

        handler = registry.get_handler("agent_task")
        assert handler.config.model == "gpt-4o"

    def test_registry_unknown_intent_falls_back(self):
        registry = HandlerRegistry()
        registry.register("simple_chat", _noop_handler, HandlerConfig())

        handler = registry.get_handler("totally_unknown_intent")
        assert handler is not None  # falls back to simple_chat

    def test_handler_config_applied(self):
        registry = HandlerRegistry()
        cfg = HandlerConfig(model="gpt-4o", max_tokens=4096, temperature=0.2)
        registry.register("agent_task", _noop_handler, cfg)

        rh = registry.get_handler("agent_task")
        assert rh.config.model == "gpt-4o"
        assert rh.config.max_tokens == 4096
        assert rh.config.temperature == 0.2


# ---------------------------------------------------------------------------
# RoutingEvaluator tests
# ---------------------------------------------------------------------------


class TestRoutingEvaluator:
    @pytest.mark.anyio
    async def test_evaluator_calculates_accuracy(self):
        """90 correct out of 100 → overall_accuracy == 0.90."""
        router = HybridRouter()

        # 90 greeted correctly, 10 wrong
        cases = (
            [RoutingTestCase("hi", "greeting")] * 90
            + [RoutingTestCase("hi", "goodbye")] * 10
        )

        async def _stub_classify(user_input, history=None):
            return make_route_result("greeting", confidence=0.9)

        router.llm.classify = _stub_classify
        router.deterministic.classify = lambda _: make_route_result(
            "greeting", confidence=0.9, method="deterministic"
        )

        evaluator = RoutingEvaluator(router)
        report = await evaluator.evaluate(cases)
        assert abs(report.overall_accuracy - 0.90) < 0.01

    @pytest.mark.anyio
    async def test_evaluator_per_intent_breakdown(self):
        """Report must contain per-intent accuracy."""
        router = HybridRouter()
        cases = [
            RoutingTestCase("hi", "greeting"),
            RoutingTestCase("bye", "goodbye"),
        ]

        async def _stub(user_input, history=None):
            return make_route_result("greeting" if "hi" in user_input else "goodbye")

        router.llm.classify = _stub
        router.deterministic.classify = lambda _: None  # force LLM

        evaluator = RoutingEvaluator(router)
        report = await evaluator.evaluate(cases)
        assert "greeting" in report.by_intent
        assert "goodbye" in report.by_intent

    @pytest.mark.anyio
    async def test_evaluator_top_misclassifications(self):
        """Misclassifications should appear in the report."""
        router = HybridRouter()

        async def _stub(user_input, history=None):
            return make_route_result("greeting")  # always wrong for non-greeting

        router.llm.classify = _stub
        router.deterministic.classify = lambda _: None

        cases = [RoutingTestCase("what is your return policy?", "knowledge_question")] * 5
        evaluator = RoutingEvaluator(router)
        report = await evaluator.evaluate(cases)
        assert len(report.top_misclassifications) > 0
        assert report.top_misclassifications[0][1] == 5

    @pytest.mark.anyio
    async def test_evaluator_deterministic_rate(self):
        """Deterministic rate should reflect what the router actually used."""
        router = HybridRouter()

        evaluator = RoutingEvaluator(router)
        cases = [RoutingTestCase("hello", "greeting")] * 10
        report = await evaluator.evaluate(cases)
        # All "hello" inputs match deterministic patterns
        assert report.deterministic_rate == 1.0

    @pytest.mark.anyio
    async def test_deterministic_route_faster_than_llm(self):
        """Deterministic classification must be faster than a (mocked) LLM call."""
        det_router = DeterministicRouter()

        async def _slow_llm(user_input, history=None):
            await asyncio.sleep(0.05)  # simulate 50 ms LLM latency
            return make_route_result("simple_chat")

        t0 = time.monotonic()
        det_result = det_router.classify("hello there")
        det_time = time.monotonic() - t0

        t0 = time.monotonic()
        await _slow_llm("hello there")
        llm_time = time.monotonic() - t0

        assert det_time < llm_time


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestIntegration:
    @pytest.mark.anyio
    async def test_complete_flow(self):
        """Full pipeline: deterministic route → handler → response."""

        async def _chat_handler(user_input, history, config):
            return HandlerResponse(
                content="Hi there!",
                handler_used="simple_chat",
                metadata={},
            )

        registry = HandlerRegistry()
        registry.register("greeting", _chat_handler, HandlerConfig())
        registry.register("simple_chat", _chat_handler, HandlerConfig())

        router = HybridRouter()
        escrouter = EscalatingRouter(router, registry)

        resp = await escrouter.handle("hello!")
        assert resp.content == "Hi there!"
        assert resp.handler_used == "simple_chat" or resp.handler_used == "simple_chat"

    @pytest.mark.anyio
    async def test_100_cases_accuracy_above_threshold(self):
        """Overall routing accuracy across 100 varied cases must exceed 85%."""
        from routing_test_suite import ALL_TEST_CASES, _stub_llm_classify

        router = HybridRouter()
        LLMRouter.classify = _stub_llm_classify  # type: ignore[method-assign]

        evaluator = RoutingEvaluator(router)
        report = await evaluator.evaluate(ALL_TEST_CASES[:100])
        assert report.overall_accuracy >= 0.85, (
            f"Accuracy {report.overall_accuracy:.1%} is below the 85% threshold"
        )
