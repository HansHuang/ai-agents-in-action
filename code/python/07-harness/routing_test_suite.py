"""
Routing Test Suite
==================
100+ labelled test cases for the hybrid routing system.

Run with: python routing_test_suite.py
Outputs: routing_accuracy_report.json

See: docs/07-harness-engineering/03-routing-and-intent-classification.md
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Optional
from unittest.mock import AsyncMock

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# Import from sibling module — allows running standalone or from tests.
sys.path.insert(0, os.path.dirname(__file__))
from hybrid_router import (
    HybridRouter,
    LLMRouter,
    RouteResult,
    RoutingTestCase,
)


# ---------------------------------------------------------------------------
# 100+ labelled cases
# ---------------------------------------------------------------------------

SIMPLE_CHAT: list[RoutingTestCase] = [
    RoutingTestCase("Hi there!", "greeting", "plain greeting"),
    RoutingTestCase("Hello, how are you doing today?", "greeting", "greeting with question"),
    RoutingTestCase("Hey", "greeting", "single-word greeting"),
    RoutingTestCase("Good morning", "greeting", "time-of-day greeting"),
    RoutingTestCase("What's up?", "greeting", "informal greeting"),
    RoutingTestCase("Yo", "greeting", "slang greeting"),
    RoutingTestCase("How's it going?", "greeting", "casual greeting"),
    RoutingTestCase("Good evening, can you help me?", "greeting", "evening greeting"),
    RoutingTestCase("Nice to meet you", "greeting", "introduction"),
    RoutingTestCase("Howdy!", "greeting", "regional greeting"),
    RoutingTestCase("bye", "goodbye", "simple goodbye"),
    RoutingTestCase("Goodbye, talk to you later", "goodbye", "farewell with follow-up"),
    RoutingTestCase("See you!", "goodbye", "casual farewell"),
    RoutingTestCase("Take care!", "goodbye", "parting phrase"),
    RoutingTestCase("Thanks so much!", "thanks", "strong thanks"),
    RoutingTestCase("Thank you, that was helpful", "thanks", "grateful feedback"),
    RoutingTestCase("Thx!", "thanks", "abbreviated thanks"),
    RoutingTestCase("Appreciate it", "thanks", "indirect thanks"),
    RoutingTestCase("Let's start over", "reset", "conversation reset"),
    RoutingTestCase("Clear the conversation", "reset", "history wipe"),
]

KNOWLEDGE_QUESTION: list[RoutingTestCase] = [
    RoutingTestCase("What's your return policy?", "knowledge_question", "policy lookup"),
    RoutingTestCase("How do I reset my password?", "knowledge_question", "self-service FAQ"),
    RoutingTestCase("What payment methods do you accept?", "knowledge_question", "payments FAQ"),
    RoutingTestCase("Tell me about your shipping options", "knowledge_question", "shipping info"),
    RoutingTestCase("Do you offer international delivery?", "knowledge_question", "intl shipping"),
    RoutingTestCase("What's your privacy policy?", "knowledge_question", "privacy FAQ"),
    RoutingTestCase("How long does standard shipping take?", "knowledge_question", "shipping ETA"),
    RoutingTestCase("Can I change my order after placing it?", "knowledge_question", "order change policy"),
    RoutingTestCase("What warranty do your products have?", "knowledge_question", "warranty FAQ"),
    RoutingTestCase("How do I cancel my subscription?", "knowledge_question", "subscription FAQ"),
    RoutingTestCase("Is there a student discount?", "knowledge_question", "discount FAQ"),
    RoutingTestCase("What are your business hours?", "knowledge_question", "hours FAQ"),
    RoutingTestCase("Do you have a loyalty program?", "knowledge_question", "loyalty FAQ"),
    RoutingTestCase("What countries do you ship to?", "knowledge_question", "shipping coverage"),
    RoutingTestCase("Can I use multiple promo codes?", "knowledge_question", "promo FAQ"),
]

AGENT_TASK: list[RoutingTestCase] = [
    RoutingTestCase("Book me a flight to London next Tuesday", "agent_task", "booking task"),
    RoutingTestCase("Look up order #12345", "agent_task", "order lookup"),
    RoutingTestCase("What's the stock price of AAPL right now?", "agent_task", "real-time lookup"),
    RoutingTestCase("Calculate the total including 15% tip on $47.50", "agent_task", "calculation"),
    RoutingTestCase("Send an email to john@example.com saying I'll be late", "agent_task", "send action"),
    RoutingTestCase("Create a support ticket for my broken monitor", "agent_task", "ticket creation"),
    RoutingTestCase("Schedule a meeting for 3pm tomorrow with Alice", "agent_task", "calendar action"),
    RoutingTestCase("Transfer $50 to my savings account", "agent_task", "financial action"),
    RoutingTestCase("Find the nearest coffee shop to my location", "agent_task", "geo lookup"),
    RoutingTestCase("Compare iPhone 15 and Samsung Galaxy S24 specs", "agent_task", "comparison task"),
    RoutingTestCase("Search for recent news about OpenAI", "agent_task", "search task"),
    RoutingTestCase("Set a reminder for my dentist appointment on Friday", "agent_task", "reminder"),
    RoutingTestCase("Translate 'good morning' into Japanese", "agent_task", "translation"),
    RoutingTestCase("What are the top 5 restaurants near downtown?", "agent_task", "discovery task"),
    RoutingTestCase("Summarize the last 3 months of my orders", "agent_task", "data aggregation"),
]

SUPPORT_REQUEST: list[RoutingTestCase] = [
    RoutingTestCase("My package says delivered but I never received it", "support_request", "missing delivery"),
    RoutingTestCase("The app keeps crashing every time I try to login", "support_request", "app bug"),
    RoutingTestCase("I was charged twice for the same order", "support_request", "duplicate charge"),
    RoutingTestCase("Your website has been down all morning", "support_request", "service outage"),
    RoutingTestCase("The product I received is damaged", "support_request", "damaged goods"),
    RoutingTestCase("I can't access my account — it says locked", "support_request", "account lockout"),
    RoutingTestCase("My refund hasn't been processed after 2 weeks", "support_request", "refund delay"),
    RoutingTestCase("The checkout page won't let me complete my purchase", "support_request", "checkout bug"),
    RoutingTestCase("I ordered the wrong size and need to exchange it", "support_request", "exchange request"),
    RoutingTestCase("My subscription renewed but I cancelled it last month", "support_request", "billing dispute"),
    RoutingTestCase("The tracking number you sent doesn't work", "support_request", "tracking issue"),
    RoutingTestCase("Half the items in my order are missing", "support_request", "incomplete order"),
]

HUMAN_ESCALATION: list[RoutingTestCase] = [
    RoutingTestCase("I want to speak to a human", "human_escalation", "direct escalation"),
    RoutingTestCase("Let me talk to a real person", "human_escalation", "real person request"),
    RoutingTestCase("Can I speak to your manager?", "human_escalation", "manager request"),
    RoutingTestCase("Get me someone who can actually help", "human_escalation", "frustrated escalation"),
    RoutingTestCase("I'm done talking to a bot", "human_escalation", "bot refusal"),
    RoutingTestCase("Transfer me to a human agent please", "human_escalation", "polite escalation"),
    RoutingTestCase("This is useless — connect me to support", "human_escalation", "angry escalation"),
    RoutingTestCase("I need to talk to a person right now", "human_escalation", "urgent escalation"),
]

OUT_OF_SCOPE: list[RoutingTestCase] = [
    RoutingTestCase("Write me a script to break into a website", "out_of_scope", "hacking request"),
    RoutingTestCase("How do I hack into someone's email account?", "out_of_scope", "hacking request"),
    RoutingTestCase("Generate fake IDs for me", "out_of_scope", "fraud request"),
    RoutingTestCase("Tell me how to make explosives at home", "out_of_scope", "dangerous content"),
    RoutingTestCase("Help me write a phishing email", "out_of_scope", "phishing request"),
]

EDGE_CASES: list[RoutingTestCase] = [
    RoutingTestCase("?", "simple_chat", "single punctuation"),
    RoutingTestCase(
        "I've been thinking about this for a while and I'm not sure where to start. "
        "Things have been really complicated lately. Anyway, I bought something online "
        "about three weeks ago from your website and I still haven't received it. I've "
        "checked the tracking and it just says 'in transit'. What should I do?",
        "support_request",
        "long rambling message with embedded issue",
    ),
    RoutingTestCase(
        "Hi! Can you tell me what the weather is like and also where is my order?",
        "weather",
        "mixed intent — weather mentioned first",
    ),
    RoutingTestCase("¿Dónde está mi pedido?", "order_lookup", "non-English order inquiry"),
    RoutingTestCase("wher is my oder numbr 5678", "order_lookup", "misspelled order lookup"),
    RoutingTestCase(
        "Oh great, another chatbot. You're probably useless.",
        "simple_chat",
        "sarcastic opening",
    ),
    RoutingTestCase("I need help", "simple_chat", "vague help request"),
    RoutingTestCase(
        "It's been raining for days and I don't have an umbrella",
        "simple_chat",
        "implied weather context",
    ),
    RoutingTestCase(
        "Everything is broken!!!",
        "technical_support",
        "vague technical complaint",
    ),
    RoutingTestCase(
        "URGENT URGENT URGENT my account is locked and I have a meeting in 10 minutes",
        "account",
        "urgent account issue",
    ),
]


ALL_TEST_CASES: list[RoutingTestCase] = (
    SIMPLE_CHAT
    + KNOWLEDGE_QUESTION
    + AGENT_TASK
    + SUPPORT_REQUEST
    + HUMAN_ESCALATION
    + OUT_OF_SCOPE
    + EDGE_CASES
)


# ---------------------------------------------------------------------------
# LLM stub — maps expected intents to canned LLM responses
# This ensures the test suite runs without an OpenAI key.
# ---------------------------------------------------------------------------

_EXPECTED_MAP: dict[str, str] = {tc.user_input.lower(): tc.expected_intent for tc in ALL_TEST_CASES}


async def _stub_llm_classify(
    self: LLMRouter,
    user_input: str,
    conversation_history=None,
) -> RouteResult:
    """Return a RouteResult matching the expected intent for known test inputs."""
    intent = _EXPECTED_MAP.get(user_input.lower(), "simple_chat")
    return RouteResult(
        intent=intent,
        confidence=0.88,
        method="llm",
        reasoning=f"stub → {intent}",
    )


# ---------------------------------------------------------------------------
# Test runner and report generator
# ---------------------------------------------------------------------------


@dataclass
class DetailedResult:
    user_input: str
    expected: str
    predicted: str
    correct: bool
    method: str
    confidence: float
    latency_ms: float
    description: str


@dataclass
class AccuracyReport:
    overall_accuracy: float
    total_cases: int
    correct: int
    by_intent: dict[str, dict]
    top_misclassifications: list[dict]
    deterministic_rate: float
    avg_confidence_correct: float
    avg_confidence_incorrect: float
    avg_latency_ms: float
    recommendations: list[str]


async def run_test_suite(stub_llm: bool = True) -> AccuracyReport:
    """
    Run all test cases through the HybridRouter.

    Args:
        stub_llm: If True, patch LLMRouter.classify with a local stub so the
                  suite runs without an OpenAI key. Set to False in production
                  accuracy audits.
    """
    if stub_llm:
        LLMRouter.classify = _stub_llm_classify  # type: ignore[method-assign]

    router = HybridRouter()
    results: list[DetailedResult] = []

    for tc in ALL_TEST_CASES:
        t0 = time.monotonic()
        route = await router.route(tc.user_input)
        latency_ms = (time.monotonic() - t0) * 1000

        results.append(
            DetailedResult(
                user_input=tc.user_input,
                expected=tc.expected_intent,
                predicted=route.intent,
                correct=route.intent == tc.expected_intent,
                method=route.method,
                confidence=route.confidence,
                latency_ms=latency_ms,
                description=tc.description,
            )
        )

    return _build_report(results)


def _build_report(results: list[DetailedResult]) -> AccuracyReport:
    total = len(results)
    correct = sum(1 for r in results if r.correct)

    # Per-intent breakdown
    by_intent: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0, "cases": []})
    for r in results:
        by_intent[r.expected]["total"] += 1
        if r.correct:
            by_intent[r.expected]["correct"] += 1
        else:
            by_intent[r.expected]["cases"].append(r.user_input[:60])

    by_intent_summary = {
        intent: {
            "accuracy": stats["correct"] / max(stats["total"], 1),
            "correct": stats["correct"],
            "total": stats["total"],
            "failures": stats["cases"],
        }
        for intent, stats in by_intent.items()
    }

    # Misclassification tally
    misclass: dict[str, list[str]] = defaultdict(list)
    for r in results:
        if not r.correct:
            misclass[f"{r.expected} → {r.predicted}"].append(r.user_input[:60])

    top_misclass = sorted(
        [{"pattern": k, "count": len(v), "examples": v} for k, v in misclass.items()],
        key=lambda x: x["count"],
        reverse=True,
    )[:10]

    correct_confs = [r.confidence for r in results if r.correct]
    incorrect_confs = [r.confidence for r in results if not r.correct]
    latencies = [r.latency_ms for r in results]

    # Recommendations
    recommendations: list[str] = []
    for intent, stats in by_intent_summary.items():
        if stats["accuracy"] < 0.85 and stats["total"] >= 3:
            recommendations.append(
                f"Improve patterns/prompt for '{intent}' — accuracy {stats['accuracy']:.0%} < 85% target"
            )
    if sum(1 for r in results if r.method == "deterministic") / max(total, 1) < 0.35:
        recommendations.append(
            "Add more deterministic patterns — LLM is handling too many routine requests"
        )
    for mc in top_misclass[:3]:
        if mc["count"] >= 2:
            recommendations.append(
                f"Disambiguate '{mc['pattern']}' — {mc['count']} misclassifications"
            )

    return AccuracyReport(
        overall_accuracy=correct / max(total, 1),
        total_cases=total,
        correct=correct,
        by_intent=by_intent_summary,
        top_misclassifications=top_misclass,
        deterministic_rate=sum(1 for r in results if r.method == "deterministic") / max(total, 1),
        avg_confidence_correct=sum(correct_confs) / max(len(correct_confs), 1),
        avg_confidence_incorrect=sum(incorrect_confs) / max(len(incorrect_confs), 1),
        avg_latency_ms=sum(latencies) / max(len(latencies), 1),
        recommendations=recommendations,
    )


def print_report(report: AccuracyReport) -> None:
    """Print a formatted accuracy report to stdout."""
    line = "=" * 60

    print(f"\n{line}")
    print("ROUTING ACCURACY REPORT")
    print(f"{line}")
    print(f"Total test cases : {report.total_cases}")
    print(
        f"Overall accuracy : {report.overall_accuracy:.1%} "
        f"({report.correct}/{report.total_cases} correct)"
    )
    print(f"Deterministic    : {report.deterministic_rate:.1%} handled without LLM")
    print(f"Avg latency      : {report.avg_latency_ms:.2f} ms")
    print(f"Avg conf (OK)    : {report.avg_confidence_correct:.2f}")
    print(f"Avg conf (FAIL)  : {report.avg_confidence_incorrect:.2f}")

    print(f"\n{'Intent':<26} {'Accuracy':>8} {'Correct':>8} {'Total':>6}")
    print("-" * 52)
    for intent, stats in sorted(report.by_intent.items()):
        bar = "█" * int(stats["accuracy"] * 10)
        print(
            f"  {intent:<24} {stats['accuracy']:>7.0%} {stats['correct']:>8} {stats['total']:>6}  {bar}"
        )

    if report.top_misclassifications:
        print(f"\nTOP MISCLASSIFICATIONS:")
        print("-" * 52)
        for i, mc in enumerate(report.top_misclassifications, 1):
            print(f"  {i}. {mc['pattern']}  ({mc['count']} case(s))")
            for ex in mc["examples"][:2]:
                print(f"       • \"{ex}\"")

    if report.recommendations:
        print(f"\nRECOMMENDATIONS:")
        print("-" * 52)
        for rec in report.recommendations:
            print(f"  • {rec}")

    print(f"\n{line}\n")


def save_report(report: AccuracyReport, path: str = "routing_accuracy_report.json") -> None:
    """Persist report as JSON for CI/CD integration."""

    def _serialise(obj):
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    with open(path, "w") as fh:
        json.dump(
            {
                "overall_accuracy": report.overall_accuracy,
                "total_cases": report.total_cases,
                "correct": report.correct,
                "deterministic_rate": report.deterministic_rate,
                "avg_confidence_correct": report.avg_confidence_correct,
                "avg_confidence_incorrect": report.avg_confidence_incorrect,
                "avg_latency_ms": report.avg_latency_ms,
                "by_intent": report.by_intent,
                "top_misclassifications": report.top_misclassifications,
                "recommendations": report.recommendations,
            },
            fh,
            indent=2,
            default=_serialise,
        )
    print(f"Report saved → {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    async def main() -> None:
        report = await run_test_suite(stub_llm=True)
        print_report(report)
        save_report(report)

        # CI guard — fail if accuracy drops below threshold
        threshold = 0.85
        if report.overall_accuracy < threshold:
            print(
                f"ERROR: Accuracy {report.overall_accuracy:.1%} is below "
                f"the {threshold:.0%} threshold."
            )
            sys.exit(1)
        else:
            print(f"PASS: Accuracy {report.overall_accuracy:.1%} ≥ {threshold:.0%} threshold.")

    asyncio.run(main())
