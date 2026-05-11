"""
Hybrid Routing System
=====================
A two-stage router that classifies user requests and directs them to
specialized handlers.

Stage 1: DeterministicRouter — fast regex patterns, ~0ms, no cost
Stage 2: LLMRouter           — small model (gpt-4o-mini), ~300ms, tiny cost

The hybrid approach handles 70-80% of traffic deterministically,
only calling the LLM for genuinely ambiguous requests.

See: docs/07-harness-engineering/03-routing-and-intent-classification.md
"""

from __future__ import annotations

import json
import logging
import os
import re
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from openai import AsyncOpenAI

# ---------------------------------------------------------------------------
# Structured logger
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger(__name__)


def _structured(event: str, **kwargs: Any) -> None:
    _log.info(json.dumps({"event": event, **kwargs}))


# ---------------------------------------------------------------------------
# Shared data types
# ---------------------------------------------------------------------------


@dataclass
class RouteResult:
    """Classification decision from any router."""

    intent: str
    confidence: float
    method: str                          # "deterministic" | "llm"
    reasoning: Optional[str] = None
    matched_pattern: Optional[str] = None
    extracted_params: Optional[dict] = None


@dataclass
class HandlerConfig:
    """Configuration applied to a single route handler."""

    model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.7
    timeout_seconds: int = 30
    requires_tools: bool = False
    requires_rag: bool = False
    requires_approval: bool = False
    cost_budget: float = 0.01


@dataclass
class HandlerResponse:
    """Result returned by any handler."""

    content: str
    handler_used: str
    tokens_used: int = 0
    cost: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class RouteHandler:
    """A handler function paired with its configuration."""

    handler: Callable
    config: HandlerConfig


@dataclass
class EvaluationResult:
    """Single test-case evaluation outcome."""

    input: str
    expected: str
    predicted: str
    correct: bool
    method: str
    confidence: float


@dataclass
class RoutingTestCase:
    """A labelled routing test case."""

    user_input: str
    expected_intent: str
    description: str = ""


@dataclass
class RoutingReport:
    """Aggregated evaluation results."""

    overall_accuracy: float
    total_cases: int
    by_intent: dict[str, float]
    top_misclassifications: list[tuple[str, int]]
    deterministic_rate: float
    avg_confidence_correct: float
    avg_confidence_incorrect: float


# ---------------------------------------------------------------------------
# Deterministic router
# ---------------------------------------------------------------------------


class DeterministicRouter:
    """
    Route requests using fast, rule-based regex patterns.
    Handles ~70-80% of traffic instantly with no LLM cost.
    """

    PATTERNS: dict[str, list[str]] = {
        "greeting": [
            r"^(hi|hello|hey|good morning|good evening|good afternoon|yo|sup)\b",
            r"^(how are you|how\'s it going|what\'s up|howdy)\b",
            r"^(nice to meet you|pleased to meet you)\b",
        ],
        "goodbye": [
            r"\b(bye|goodbye|see you|talk later|farewell|ciao|later|ttyl)\b",
            r"\b(take care|have a good (day|night|one))\b",
        ],
        "thanks": [
            r"\b(thanks|thank you|thx|ty|appreciate it|grateful|cheers)\b",
            r"\b(many thanks|much appreciated|that\'s helpful)\b",
        ],
        "reset": [
            r"\b(start over|start fresh|reset|clear|new conversation|forget everything|fresh start)\b",
            r"\b(wipe (the slate|history)|begin again|restart)\b",
        ],
        "help": [
            r"\b(what can you do|help me|capabilities|features|how do (I|you)|what do you (do|know))\b",
            r"^help$",
            r"\b(show me (what you|your) (can do|capabilities))\b",
        ],
        "weather": [
            r"\b(weather|temperature|forecast|humidity|rain(ing)?|sunny|cloudy|snow(ing)?|wind)\b",
            r"\b(what\'s it like outside|will it (rain|snow))\b",
        ],
        "stock": [
            r"\b(stock|market|price|ticker|nasdaq|dow jones|s&p|invest(ment)?|share price|equity)\b",
            r"\b(\baapl\b|\bgoog\b|\bmsft\b|\btsla\b|\bamzn\b)\b",
        ],
        "order_lookup": [
            r"\b(order|tracking|shipment|delivery|where is my|status of)\b.*\b(order|package|item|number|parcel)\b",
            r"\border\s*#?\d+\b",
            r"\b(track|locate) my (package|order|shipment)\b",
        ],
        "return_request": [
            r"\b(return|refund|exchange|money back|send back|cancel order|send it back)\b",
            r"\b(initiate a return|process a refund|want my money back)\b",
        ],
        "billing": [
            r"\b(bill|invoice|charge|payment|subscription|receipt|pricing|cost|fee)\b",
            r"\b(overcharged|unauthorized charge|billing issue|payment failed)\b",
        ],
        "technical_support": [
            r"\b(not working|broken|error|bug|crash|down|failed|issue|problem with)\b",
            r"\b(won\'t (load|open|start)|keeps (crashing|freezing)|can\'t (connect|access))\b",
        ],
        "account": [
            r"\b(account|login|password|profile|settings|email change|update.*info|sign in)\b",
            r"\b(forgot (my )?password|reset (my )?password|locked out|can\'t log in)\b",
        ],
    }

    def __init__(self, patterns: Optional[dict[str, list[str]]] = None) -> None:
        source = patterns or self.PATTERNS
        self.compiled: dict[str, list[re.Pattern]] = {
            intent: [re.compile(p, re.IGNORECASE) for p in pats]
            for intent, pats in source.items()
        }

    def classify(self, user_input: str) -> Optional[RouteResult]:
        """
        Try to classify deterministically.
        Returns None when no pattern matches (defer to LLM).
        """
        matches: list[tuple[str, re.Pattern]] = []

        for intent, patterns in self.compiled.items():
            for pattern in patterns:
                if pattern.search(user_input):
                    matches.append((intent, pattern))

        if not matches:
            return None

        # Longer pattern string → more specific → preferred
        best_intent, best_pattern = max(matches, key=lambda m: len(m[1].pattern))

        confidence = 0.85 if len(matches) == 1 else 0.65

        _structured(
            "deterministic_classify",
            intent=best_intent,
            confidence=confidence,
            matches=len(matches),
        )

        return RouteResult(
            intent=best_intent,
            confidence=confidence,
            method="deterministic",
            matched_pattern=best_pattern.pattern,
        )


# ---------------------------------------------------------------------------
# LLM router
# ---------------------------------------------------------------------------


class LLMRouter:
    """
    Route ambiguous requests using a fast, cheap LLM (gpt-4o-mini).
    Called only when deterministic patterns are insufficient.
    """

    ROUTING_PROMPT = """You are a request classifier for a customer-facing AI assistant.
Analyze the user's message and determine its primary intent.

Available routes:
- simple_chat: Casual conversation, greetings, general questions not requiring tools
- knowledge_question: Questions answerable from a knowledge base (policies, docs, FAQs)
- agent_task: Requests requiring tool use (lookups, calculations, multi-step tasks)
- human_escalation: User explicitly asks for a human, or the request is too complex/sensitive
- support_request: Customer support issues, complaints, problems with products or services
- out_of_scope: Requests that cannot or should not be handled (harmful, impossible, off-topic)

Classification rules:
- "hi" or small talk → simple_chat
- Questions about policies, procedures, documentation → knowledge_question
- Requests to DO something (look up, calculate, book, create, send) → agent_task
- Frustrated user demanding a person → human_escalation
- Reports a problem with a product or service → support_request
- Inappropriate, impossible, or clearly unrelated requests → out_of_scope

Output ONLY a JSON object — no markdown, no extra text:
{
    "intent": "<intent_name>",
    "confidence": 0.0,
    "reasoning": "<one sentence>",
    "extracted_params": {
        "order_number": null,
        "product_name": null,
        "issue_type": null
    }
}"""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self._client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    async def classify(
        self,
        user_input: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> RouteResult:
        """Classify using an LLM. Handles API errors with a safe fallback."""

        messages: list[dict] = [{"role": "system", "content": self.ROUTING_PROMPT}]

        if conversation_history:
            recent = conversation_history[-4:]
            summary = "\n".join(
                f"{'User' if m['role'] == 'user' else 'Assistant'}: {str(m.get('content', ''))[:200]}"
                for m in recent
            )
            content = f"Recent conversation:\n{summary}\n\nClassify this message: {user_input}"
        else:
            content = user_input

        messages.append({"role": "user", "content": content})

        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw = response.choices[0].message.content or "{}"
            result = json.loads(raw)

            _structured(
                "llm_classify",
                intent=result.get("intent"),
                confidence=result.get("confidence"),
                model=self.model,
            )

            return RouteResult(
                intent=result.get("intent", "simple_chat"),
                confidence=float(result.get("confidence", 0.5)),
                method="llm",
                reasoning=result.get("reasoning"),
                extracted_params=result.get("extracted_params"),
            )

        except Exception as exc:
            _structured("llm_classify_error", error=str(exc))
            # Safe fallback: treat as simple chat
            return RouteResult(
                intent="simple_chat",
                confidence=0.3,
                method="llm",
                reasoning=f"Classification failed ({exc}); defaulting to simple_chat",
            )


# ---------------------------------------------------------------------------
# Router metrics
# ---------------------------------------------------------------------------


class RouterMetrics:
    """Thread-safe routing statistics collector."""

    def __init__(self) -> None:
        self.total: int = 0
        self.by_method: dict[str, int] = defaultdict(int)
        self.by_intent: dict[str, int] = defaultdict(int)
        self.latencies_ms: list[float] = []
        self.llm_costs: list[float] = []

    def record(
        self,
        method: str,
        intent: str,
        latency_ms: float = 0.0,
        cost: float = 0.0,
    ) -> None:
        self.total += 1
        self.by_method[method] += 1
        self.by_intent[intent] += 1
        self.latencies_ms.append(latency_ms)
        self.llm_costs.append(cost)

    def summary(self) -> dict:
        n = max(self.total, 1)
        lats = sorted(self.latencies_ms) if self.latencies_ms else [0.0]

        return {
            "total_routed": self.total,
            "deterministic_rate": self.by_method.get("deterministic", 0) / n,
            "llm_fallback_rate": self.by_method.get("llm", 0) / n,
            "intent_distribution": dict(self.by_intent),
            "avg_latency_ms": sum(lats) / len(lats),
            "p50_latency_ms": lats[len(lats) // 2],
            "p95_latency_ms": lats[min(int(len(lats) * 0.95), len(lats) - 1)],
            "total_llm_cost_usd": sum(self.llm_costs),
        }


# ---------------------------------------------------------------------------
# Hybrid router
# ---------------------------------------------------------------------------


class HybridRouter:
    """
    Two-stage router: deterministic first, LLM for ambiguous cases.

    Routing logic:
      1. Run deterministic classifier.
      2. If confidence > 0.8 → return immediately (no LLM call).
      3. Otherwise call LLM classifier.
      4. Return whichever result has higher confidence.
    """

    def __init__(self) -> None:
        self.deterministic = DeterministicRouter()
        self.llm = LLMRouter()
        self.metrics = RouterMetrics()

    async def route(
        self,
        user_input: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> RouteResult:
        """Route a request. Deterministic first, LLM fallback."""

        t0 = time.monotonic()

        # Stage 1: deterministic
        det_result = self.deterministic.classify(user_input)

        if det_result and det_result.confidence > 0.8:
            latency = (time.monotonic() - t0) * 1000
            self.metrics.record("deterministic", det_result.intent, latency)
            _structured(
                "route_decision",
                method="deterministic",
                intent=det_result.intent,
                latency_ms=round(latency, 2),
            )
            return det_result

        # Stage 2: LLM for ambiguous requests
        llm_result = await self.llm.classify(user_input, conversation_history)
        latency = (time.monotonic() - t0) * 1000

        # Prefer deterministic if it has higher confidence than LLM
        if det_result and det_result.confidence > llm_result.confidence:
            self.metrics.record("deterministic_fallback", det_result.intent, latency)
            _structured(
                "route_decision",
                method="deterministic_fallback",
                intent=det_result.intent,
                latency_ms=round(latency, 2),
            )
            return det_result

        self.metrics.record("llm", llm_result.intent, latency)
        _structured(
            "route_decision",
            method="llm",
            intent=llm_result.intent,
            latency_ms=round(latency, 2),
        )
        return llm_result

    def get_metrics(self) -> dict:
        return self.metrics.summary()


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------


class HandlerRegistry:
    """Maps intent strings to handler functions and their configurations."""

    def __init__(self) -> None:
        self._handlers: dict[str, RouteHandler] = {}
        self._default_intent = "simple_chat"

    def register(
        self,
        intent: str,
        handler: Callable,
        config: HandlerConfig,
    ) -> None:
        self._handlers[intent] = RouteHandler(handler=handler, config=config)

    def get_handler(self, intent: str) -> RouteHandler:
        """Return handler for intent; falls back to simple_chat for unknown intents."""
        if intent in self._handlers:
            return self._handlers[intent]
        _structured("handler_fallback", unknown_intent=intent, using=self._default_intent)
        return self._handlers[self._default_intent]

    def set_default_intent(self, intent: str) -> None:
        self._default_intent = intent


# ---------------------------------------------------------------------------
# Escalating router
# ---------------------------------------------------------------------------


class EscalatingRouter:
    """
    Wraps HybridRouter and HandlerRegistry to add automatic re-routing when a
    handler fails or returns a low-quality response.
    """

    ESCALATION_PATHS: dict[str, list[str]] = {
        "simple_chat": ["knowledge_question"],
        "knowledge_question": ["agent_task"],
        "agent_task": ["human_escalation"],
        "support_request": ["human_escalation"],
        "human_escalation": [],   # terminal — no further escalation
    }

    _UNCERTAINTY_PHRASES = [
        "i'm not sure",
        "i don't know",
        "i cannot",
        "i'm unable",
        "i don't have enough information",
        "i have no information",
        "i couldn't find",
    ]

    def __init__(self, router: HybridRouter, registry: HandlerRegistry) -> None:
        self.router = router
        self.registry = registry

    async def handle(
        self,
        user_input: str,
        conversation_history: Optional[list[dict]] = None,
    ) -> HandlerResponse:
        """Route and handle a request, escalating automatically on failure."""

        intent = await self.router.route(user_input, conversation_history)
        handler = self.registry.get_handler(intent.intent)

        try:
            response = await handler.handler(
                user_input, conversation_history, handler.config
            )

            if self._should_escalate(response):
                _structured(
                    "escalating",
                    from_intent=intent.intent,
                    reason="low_quality_response",
                )
                return await self._escalate(
                    intent.intent, user_input, conversation_history, response
                )

            return response

        except Exception as exc:
            _structured("handler_error", intent=intent.intent, error=str(exc))
            return await self._escalate(
                intent.intent, user_input, conversation_history, None, str(exc)
            )

    def _should_escalate(self, response: HandlerResponse) -> bool:
        """Return True when the response warrants trying a better handler."""
        meta = response.metadata

        if meta.get("documents_found") == 0:
            return True
        if meta.get("iterations", 0) >= 10:
            return True

        lower = response.content.lower()
        if any(phrase in lower for phrase in self._UNCERTAINTY_PHRASES):
            return True

        return False

    async def _escalate(
        self,
        original_intent: str,
        user_input: str,
        conversation_history: Optional[list[dict]],
        previous_response: Optional[HandlerResponse],
        error: Optional[str] = None,
    ) -> HandlerResponse:
        """Try the next handler(s) in the escalation chain."""

        path = self.ESCALATION_PATHS.get(original_intent, ["human_escalation"])

        for next_intent in path:
            handler = self.registry.get_handler(next_intent)

            # Augment the input so the next handler has context
            if previous_response:
                augmented = (
                    f"[Previous attempt via '{original_intent}' was insufficient. "
                    f"Response was: '{previous_response.content[:200]}...']\n\n"
                    f"Original request: {user_input}"
                )
            else:
                augmented = user_input

            try:
                response = await handler.handler(
                    augmented, conversation_history, handler.config
                )
                response.metadata["escalated_from"] = original_intent
                response.metadata["escalation_reason"] = error or "low_confidence"
                return response
            except Exception as exc:
                _structured(
                    "escalation_handler_error", intent=next_intent, error=str(exc)
                )
                continue

        # All escalation handlers exhausted
        return HandlerResponse(
            content=(
                "I apologize — I'm having trouble processing your request. "
                "A human team member will follow up with you shortly."
            ),
            handler_used="escalation_fallback",
            metadata={"escalation_chain_exhausted": True},
        )


# ---------------------------------------------------------------------------
# Routing evaluator
# ---------------------------------------------------------------------------


class RoutingEvaluator:
    """Evaluate routing accuracy against labelled test cases."""

    def __init__(self, router: HybridRouter) -> None:
        self.router = router

    async def evaluate(self, test_cases: list[RoutingTestCase]) -> RoutingReport:
        results: list[EvaluationResult] = []

        for tc in test_cases:
            route = await self.router.route(tc.user_input)
            correct = route.intent == tc.expected_intent
            results.append(
                EvaluationResult(
                    input=tc.user_input,
                    expected=tc.expected_intent,
                    predicted=route.intent,
                    correct=correct,
                    method=route.method,
                    confidence=route.confidence,
                )
            )

        return self._generate_report(results)

    def _generate_report(self, results: list[EvaluationResult]) -> RoutingReport:
        total = len(results)
        correct_count = sum(1 for r in results if r.correct)

        by_intent: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            by_intent[r.expected]["total"] += 1
            if r.correct:
                by_intent[r.expected]["correct"] += 1

        misclassifications: dict[str, int] = defaultdict(int)
        for r in results:
            if not r.correct:
                key = f"{r.expected} → {r.predicted}"
                misclassifications[key] += 1

        correct_confs = [r.confidence for r in results if r.correct]
        incorrect_confs = [r.confidence for r in results if not r.correct]

        return RoutingReport(
            overall_accuracy=correct_count / max(total, 1),
            total_cases=total,
            by_intent={
                intent: stats["correct"] / max(stats["total"], 1)
                for intent, stats in by_intent.items()
            },
            top_misclassifications=sorted(
                misclassifications.items(), key=lambda x: x[1], reverse=True
            )[:10],
            deterministic_rate=sum(1 for r in results if r.method == "deterministic") / max(total, 1),
            avg_confidence_correct=sum(correct_confs) / max(len(correct_confs), 1),
            avg_confidence_incorrect=sum(incorrect_confs) / max(len(incorrect_confs), 1),
        )


# ---------------------------------------------------------------------------
# Complete harness integration
# ---------------------------------------------------------------------------


@dataclass
class FinalResponse:
    """Response object returned by the top-level harness."""

    content: str
    status: str                          # "success" | "rejected" | "blocked"
    route: Optional[str] = None
    handler: Optional[str] = None
    rejection_layer: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class HarnessWithRouting:
    """
    Complete harness: input guardrails → routing → handler → output guardrails.

    The input/output guardrail pipelines are injected as callables so this
    class has no hard dependency on the guardrail module.
    """

    def __init__(
        self,
        input_guardrail: Optional[Callable] = None,
        output_guardrail: Optional[Callable] = None,
    ) -> None:
        self.router = HybridRouter()
        self.registry = HandlerRegistry()
        self.escalating = EscalatingRouter(self.router, self.registry)
        self._input_guardrail = input_guardrail
        self._output_guardrail = output_guardrail

    async def process(
        self,
        user_input: str,
        user_id: str = "anonymous",
        conversation_history: Optional[list[dict]] = None,
    ) -> FinalResponse:
        """Process a user request through the complete harness."""

        _structured("harness_start", user_id=user_id, input_len=len(user_input))

        # Phase 1: Input guardrails (optional)
        cleaned_input = user_input
        if self._input_guardrail:
            guard_result = await self._input_guardrail(user_input, user_id)
            if not guard_result.passed:
                return FinalResponse(
                    content=guard_result.rejection_reason,
                    status="rejected",
                    rejection_layer=guard_result.rejection_layer,
                )
            cleaned_input = guard_result.cleaned_input or user_input

        # Phase 2 + 3: Route and handle (with escalation)
        route_result = await self.router.route(cleaned_input, conversation_history)
        handler_response = await self.escalating.handle(
            cleaned_input, conversation_history
        )

        # Phase 4: Output guardrails (optional)
        if self._output_guardrail:
            out_result = await self._output_guardrail(handler_response.content)
            if not out_result.passed:
                return FinalResponse(
                    content="I'm unable to provide that response. Please rephrase your request.",
                    status="blocked",
                    rejection_layer="output_guardrails",
                )

        _structured(
            "harness_complete",
            user_id=user_id,
            intent=route_result.intent,
            handler=handler_response.handler_used,
        )

        return FinalResponse(
            content=handler_response.content,
            status="success",
            route=route_result.intent,
            handler=handler_response.handler_used,
            metadata=handler_response.metadata,
        )


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


async def _demo() -> None:
    """
    Demonstrate the hybrid routing system with 20 diverse requests.
    Does NOT require an OpenAI key for the deterministic cases.
    """
    import asyncio
    from unittest.mock import AsyncMock, MagicMock

    # ── Stub LLM router so demo runs without an API key ───────────────────
    llm_answers: dict[str, RouteResult] = {
        "tell me about your return policy": RouteResult(
            "knowledge_question", 0.92, "llm",
            "User asking about a policy → knowledge_question"
        ),
        "where is my order #12345": RouteResult(
            "agent_task", 0.94, "llm",
            "User wants an order lookup → agent_task",
            extracted_params={"order_number": "12345"}
        ),
        "my laptop screen has dead pixels, bought it last week": RouteResult(
            "support_request", 0.91, "llm",
            "User reporting a product defect → support_request"
        ),
        "can you compare iphone 15 and samsung s24": RouteResult(
            "agent_task", 0.88, "llm",
            "Multi-step comparison task → agent_task"
        ),
        "do you offer international shipping?": RouteResult(
            "knowledge_question", 0.90, "llm",
            "Shipping policy question → knowledge_question"
        ),
        "schedule a meeting for tomorrow at 3pm": RouteResult(
            "agent_task", 0.93, "llm",
            "Calendar action required → agent_task"
        ),
        "i was charged twice for order 9876": RouteResult(
            "support_request", 0.89, "llm",
            "Billing dispute → support_request"
        ),
        "how do i cancel my subscription?": RouteResult(
            "knowledge_question", 0.87, "llm",
            "FAQ/docs question → knowledge_question"
        ),
        "write me a ransomware script": RouteResult(
            "out_of_scope", 0.99, "llm",
            "Harmful request → out_of_scope"
        ),
        "this is so frustrating, nothing works": RouteResult(
            "support_request", 0.82, "llm",
            "Frustrated customer with vague issue → support_request"
        ),
    }

    original_classify = LLMRouter.classify

    async def _stub_classify(self_inner, user_input, conversation_history=None):
        key = user_input.lower().strip("?!. ")
        for k, v in llm_answers.items():
            if k in key or key in k:
                return v
        return RouteResult("simple_chat", 0.5, "llm", "Default stub")

    LLMRouter.classify = _stub_classify  # type: ignore[method-assign]

    router = HybridRouter()

    test_requests = [
        ("hi", "greeting"),
        ("What's the weather in Tokyo?", "weather"),
        ("Tell me about your return policy", "knowledge_question"),
        ("Where is my order #12345", "agent_task"),
        ("I want to talk to a real person", "human_escalation"),
        ("bye, talk later", "goodbye"),
        ("thanks so much!", "thanks"),
        ("AAPL stock price today", "stock"),
        ("My laptop screen has dead pixels, bought it last week", "support_request"),
        ("Can you compare iPhone 15 and Samsung S24", "agent_task"),
        ("start over", "reset"),
        ("I need to change my password", "account"),
        ("Do you offer international shipping?", "knowledge_question"),
        ("Schedule a meeting for tomorrow at 3pm", "agent_task"),
        ("I was charged twice for order 9876", "support_request"),
        ("How do I cancel my subscription?", "knowledge_question"),
        ("Write me a ransomware script", "out_of_scope"),
        ("the app won't load", "technical_support"),
        ("This is so frustrating, nothing works", "support_request"),
        ("what can you do?", "help"),
    ]

    print("\n" + "=" * 70)
    print("HYBRID ROUTING DEMO")
    print("=" * 70)
    print(f"{'Input':<42} {'Expected':<22} {'Got':<22} {'Method':<17} {'Conf':>5}")
    print("-" * 70)

    correct = 0
    for user_input, expected in test_requests:
        result = await router.route(user_input)
        ok = result.intent == expected
        if ok:
            correct += 1
        mark = "✓" if ok else "✗"
        print(
            f"{mark} {user_input[:40]:<41} {expected:<22} {result.intent:<22} "
            f"{result.method:<16} {result.confidence:.2f}"
        )

    accuracy = correct / len(test_requests)
    print(f"\nAccuracy: {correct}/{len(test_requests)} ({accuracy:.0%})")

    # ── Routing metrics ───────────────────────────────────────────────────
    metrics = router.get_metrics()
    print("\n" + "=" * 70)
    print("ROUTING METRICS")
    print("=" * 70)
    print(f"  Total routed        : {metrics['total_routed']}")
    print(f"  Deterministic rate  : {metrics['deterministic_rate']:.0%}")
    print(f"  LLM fallback rate   : {metrics['llm_fallback_rate']:.0%}")
    print(f"  Avg latency         : {metrics['avg_latency_ms']:.1f} ms")
    print("  Intent distribution :")
    for intent, count in sorted(metrics["intent_distribution"].items(), key=lambda x: -x[1]):
        print(f"    {intent:<28} {count}")

    LLMRouter.classify = original_classify  # type: ignore[method-assign]


if __name__ == "__main__":
    import asyncio

    asyncio.run(_demo())
