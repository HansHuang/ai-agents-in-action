"""
Handler Implementations
=======================
Specialized handlers for each routing intent.

Each handler accepts (user_input, conversation_history, config) and returns
a HandlerResponse with content, tokens_used, cost, and structured metadata.

Handlers are intentionally independent — they can be swapped, tested,
and scaled separately.

See: docs/07-harness-engineering/03-routing-and-intent-classification.md
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from openai import AsyncOpenAI

from hybrid_router import HandlerConfig, HandlerRegistry, HandlerResponse, HybridRouter

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(message)s")
_log = logging.getLogger(__name__)


def _structured(event: str, **kwargs) -> None:
    _log.info(json.dumps({"event": event, **kwargs}))


# ---------------------------------------------------------------------------
# Cost model (approximate USD per 1 000 tokens)
# ---------------------------------------------------------------------------

_COST_PER_1K: dict[str, tuple[float, float]] = {
    # model: (input_cost, output_cost)
    "gpt-4o-mini": (0.00015, 0.00060),
    "gpt-4o":      (0.00250, 0.01000),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    inp, out = _COST_PER_1K.get(model, (0.002, 0.002))
    return (input_tokens * inp + output_tokens * out) / 1000


# ---------------------------------------------------------------------------
# Shared OpenAI client
# ---------------------------------------------------------------------------

_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))


# ---------------------------------------------------------------------------
# 1. simple_chat_handler
# ---------------------------------------------------------------------------

async def simple_chat_handler(
    user_input: str,
    conversation_history: Optional[list[dict]],
    config: HandlerConfig,
) -> HandlerResponse:
    """
    Handle casual conversation.
    Fast, cheap, no tools, minimal context.
    """
    t0 = time.monotonic()

    messages = [
        {
            "role": "system",
            "content": "You are a friendly, concise assistant. Keep responses brief and warm.",
        },
        *(conversation_history or [])[-6:],  # Only last 6 messages
        {"role": "user", "content": user_input},
    ]

    response = await _client.chat.completions.create(
        model=config.model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
    )

    content = response.choices[0].message.content or ""
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    total_tokens = input_tokens + output_tokens
    cost = _estimate_cost(config.model, input_tokens, output_tokens)

    latency = (time.monotonic() - t0) * 1000
    _structured(
        "handler_complete",
        handler="simple_chat",
        latency_ms=round(latency, 1),
        tokens=total_tokens,
        cost_usd=round(cost, 6),
    )

    return HandlerResponse(
        content=content,
        handler_used="simple_chat",
        tokens_used=total_tokens,
        cost=cost,
        metadata={"latency_ms": round(latency, 1)},
    )


# ---------------------------------------------------------------------------
# 2. rag_handler
# ---------------------------------------------------------------------------

@dataclass
class Document:
    """A retrieved document from the vector database."""

    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class _StubVectorDB:
    """
    Minimal in-memory vector store used when no real vector DB is configured.
    Replace with your actual vector DB client (e.g. Pinecone, Weaviate, pgvector).
    """

    _DOCS = [
        Document(
            "Our return policy allows returns within 30 days of purchase for a full refund.",
            0.92,
            {"source": "returns-policy.md"},
        ),
        Document(
            "Standard shipping takes 5-7 business days. Express shipping is 1-2 business days.",
            0.89,
            {"source": "shipping-info.md"},
        ),
        Document(
            "We accept Visa, Mastercard, American Express, PayPal, and Apple Pay.",
            0.91,
            {"source": "payments-faq.md"},
        ),
        Document(
            "To reset your password, visit account settings and click 'Forgot password'.",
            0.88,
            {"source": "account-help.md"},
        ),
        Document(
            "Warranties cover manufacturing defects for 1 year from the purchase date.",
            0.87,
            {"source": "warranty-policy.md"},
        ),
    ]

    async def search(self, query: str, k: int = 5, threshold: float = 0.7) -> list[Document]:
        # Simple keyword overlap scoring
        query_words = set(query.lower().split())
        scored = []
        for doc in self._DOCS:
            doc_words = set(doc.text.lower().split())
            overlap = len(query_words & doc_words) / max(len(query_words), 1)
            adjusted = doc.score * (0.5 + overlap)
            if adjusted >= threshold:
                scored.append(Document(doc.text, adjusted, doc.metadata))

        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:k]


_vector_db = _StubVectorDB()


async def rag_handler(
    user_input: str,
    conversation_history: Optional[list[dict]],
    config: HandlerConfig,
) -> HandlerResponse:
    """
    Answer questions from the knowledge base using RAG.
    Retrieves relevant documents, builds context, then generates a grounded answer.
    """
    t0 = time.monotonic()

    documents = await _vector_db.search(user_input, k=5, threshold=0.7)

    if not documents:
        _structured("rag_no_documents", query=user_input[:80])
        return HandlerResponse(
            content=(
                "I couldn't find information about that in our knowledge base. "
                "Let me try a different approach to help you."
            ),
            handler_used="rag",
            metadata={"documents_found": 0},
        )

    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.text}"
        for doc in documents
    )

    messages = [
        {
            "role": "system",
            "content": (
                "Answer using only the provided documents. "
                "If the answer is not in the documents, say so clearly. "
                "Cite the source when relevant.\n\n"
                f"Documents:\n{context}"
            ),
        },
        {"role": "user", "content": user_input},
    ]

    response = await _client.chat.completions.create(
        model=config.model,
        messages=messages,  # type: ignore[arg-type]
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        timeout=config.timeout_seconds,
    )

    content = response.choices[0].message.content or ""
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage else 0
    output_tokens = usage.completion_tokens if usage else 0
    total_tokens = input_tokens + output_tokens
    cost = _estimate_cost(config.model, input_tokens, output_tokens)
    latency = (time.monotonic() - t0) * 1000

    _structured(
        "handler_complete",
        handler="rag",
        documents=len(documents),
        latency_ms=round(latency, 1),
        tokens=total_tokens,
        cost_usd=round(cost, 6),
    )

    return HandlerResponse(
        content=content,
        handler_used="rag",
        tokens_used=total_tokens,
        cost=cost,
        metadata={
            "documents_found": len(documents),
            "sources": [d.metadata.get("source") for d in documents],
            "similarity_scores": [round(d.score, 3) for d in documents],
            "latency_ms": round(latency, 1),
        },
    )


# ---------------------------------------------------------------------------
# 3. agent_loop_handler
# ---------------------------------------------------------------------------

_TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_order",
            "description": "Look up the status of a customer order by order number.",
            "parameters": {
                "type": "object",
                "properties": {"order_number": {"type": "string", "description": "The order ID"}},
                "required": ["order_number"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Stock ticker symbol"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient.",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                },
                "required": ["to", "subject", "body"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


def _execute_tool(name: str, args: dict) -> str:
    """Stub tool executor. Replace with real tool implementations."""
    if name == "lookup_order":
        order_num = args.get("order_number", "UNKNOWN")
        return json.dumps({
            "order_number": order_num,
            "status": "in_transit",
            "estimated_delivery": "2026-05-12",
            "carrier": "UPS",
            "tracking": f"1Z{order_num}99",
        })
    if name == "get_stock_price":
        ticker = args.get("ticker", "UNKNOWN")
        return json.dumps({"ticker": ticker, "price": 182.41, "change": "+1.23%", "currency": "USD"})
    if name == "send_email":
        return json.dumps({"status": "sent", "message_id": str(uuid.uuid4())})
    if name == "calculate":
        try:
            result = eval(args.get("expression", "0"), {"__builtins__": {}})  # noqa: S307
            return json.dumps({"result": result})
        except Exception as exc:
            return json.dumps({"error": str(exc)})
    return json.dumps({"error": f"Unknown tool: {name}"})


async def agent_loop_handler(
    user_input: str,
    conversation_history: Optional[list[dict]],
    config: HandlerConfig,
) -> HandlerResponse:
    """
    Handle complex multi-step tasks using a tool-calling agent loop.
    Runs up to config.max_tokens // 256 iterations (default 10).
    """
    t0 = time.monotonic()
    max_iterations = 10
    iterations = 0
    tool_calls_made: list[str] = []
    total_input_tokens = 0
    total_output_tokens = 0

    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                "You are a capable AI assistant with access to tools. "
                "Use the tools available to complete the user's request accurately. "
                "When you have all the information needed, provide a final answer."
            ),
        },
        *(conversation_history or [])[-8:],
        {"role": "user", "content": user_input},
    ]

    while iterations < max_iterations:
        iterations += 1

        response = await _client.chat.completions.create(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            tools=_TOOL_DEFINITIONS,  # type: ignore[arg-type]
            tool_choice="auto",
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout_seconds,
        )

        usage = response.usage
        total_input_tokens += usage.prompt_tokens if usage else 0
        total_output_tokens += usage.completion_tokens if usage else 0

        choice = response.choices[0]
        messages.append({"role": "assistant", "content": choice.message.content or "", "tool_calls": [
            tc.model_dump() for tc in (choice.message.tool_calls or [])
        ]})

        # No tool calls → agent is done
        if not choice.message.tool_calls:
            content = choice.message.content or ""
            break

        # Execute all tool calls in this round
        for tc in choice.message.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            tool_calls_made.append(tool_name)

            _structured("tool_call", tool=tool_name, args=tool_args)

            tool_result = _execute_tool(tool_name, tool_args)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })
    else:
        content = (
            "I reached the maximum number of steps trying to complete your request. "
            "Here is what I found so far: " + (messages[-1].get("content") or "")
        )

    total_tokens = total_input_tokens + total_output_tokens
    cost = _estimate_cost(config.model, total_input_tokens, total_output_tokens)
    latency = (time.monotonic() - t0) * 1000

    _structured(
        "handler_complete",
        handler="agent_loop",
        iterations=iterations,
        tools_called=tool_calls_made,
        latency_ms=round(latency, 1),
        tokens=total_tokens,
        cost_usd=round(cost, 6),
    )

    return HandlerResponse(
        content=content,
        handler_used="agent_loop",
        tokens_used=total_tokens,
        cost=cost,
        metadata={
            "iterations": iterations,
            "tools_called": tool_calls_made,
            "latency_ms": round(latency, 1),
        },
    )


# ---------------------------------------------------------------------------
# 4. escalation_handler
# ---------------------------------------------------------------------------

async def escalation_handler(
    user_input: str,
    conversation_history: Optional[list[dict]],
    config: HandlerConfig,
) -> HandlerResponse:
    """
    Route the user to a human support agent.
    No LLM call — just create a ticket and return expected wait time.
    """
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    # Summarise conversation for the human agent
    history_summary = ""
    if conversation_history:
        recent = conversation_history[-6:]
        history_summary = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Bot'}: {str(m.get('content', ''))[:200]}"
            for m in recent
        )

    # Determine priority from signal words
    urgent_words = ["urgent", "asap", "immediately", "emergency", "critical", "now"]
    priority = "high" if any(w in user_input.lower() for w in urgent_words) else "normal"

    _structured(
        "handler_complete",
        handler="escalation",
        ticket_id=ticket_id,
        priority=priority,
    )

    content = (
        f"I've created a support ticket ({ticket_id}) and am connecting you with a human agent.\n\n"
        f"Priority: {priority.upper()}\n"
        f"Estimated wait time: {'< 5 minutes' if priority == 'high' else '10-15 minutes'}\n\n"
        "A team member will continue this conversation shortly. "
        "Please don't close this window."
    )

    return HandlerResponse(
        content=content,
        handler_used="escalation",
        tokens_used=0,
        cost=0.0,
        metadata={
            "ticket_id": ticket_id,
            "priority": priority,
            "conversation_summary": history_summary,
        },
    )


# ---------------------------------------------------------------------------
# 5. out_of_scope_handler
# ---------------------------------------------------------------------------

async def out_of_scope_handler(
    user_input: str,
    conversation_history: Optional[list[dict]],
    config: HandlerConfig,
) -> HandlerResponse:
    """
    Gracefully decline requests that cannot or should not be handled.
    Uses a short LLM call to generate a polite, context-aware refusal.
    """
    t0 = time.monotonic()

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. The user has made a request you cannot fulfill "
                "because it is outside your scope, violates policies, or is harmful. "
                "Politely decline and suggest an alternative if possible. "
                "Keep the response under 60 words. Do not lecture or moralize."
            ),
        },
        {"role": "user", "content": user_input},
    ]

    try:
        response = await _client.chat.completions.create(
            model=config.model,
            messages=messages,  # type: ignore[arg-type]
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            timeout=config.timeout_seconds,
        )
        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        total_tokens = input_tokens + output_tokens
        cost = _estimate_cost(config.model, input_tokens, output_tokens)
    except Exception:
        content = (
            "I'm not able to help with that request. "
            "If you have a different question, I'm happy to assist."
        )
        total_tokens = 0
        cost = 0.0

    latency = (time.monotonic() - t0) * 1000
    _structured(
        "handler_complete",
        handler="out_of_scope",
        latency_ms=round(latency, 1),
        tokens=total_tokens,
    )

    return HandlerResponse(
        content=content,
        handler_used="out_of_scope",
        tokens_used=total_tokens,
        cost=cost,
        metadata={"latency_ms": round(latency, 1)},
    )


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def build_handler_registry() -> HandlerRegistry:
    """Create and return a HandlerRegistry with all handlers configured."""
    registry = HandlerRegistry()

    registry.register(
        "simple_chat",
        simple_chat_handler,
        HandlerConfig(
            model="gpt-4o-mini",
            max_tokens=512,
            temperature=0.7,
            timeout_seconds=15,
            cost_budget=0.001,
        ),
    )
    registry.register(
        "greeting",
        simple_chat_handler,
        HandlerConfig(model="gpt-4o-mini", max_tokens=256, temperature=0.7, timeout_seconds=10),
    )
    registry.register(
        "goodbye",
        simple_chat_handler,
        HandlerConfig(model="gpt-4o-mini", max_tokens=128, temperature=0.7, timeout_seconds=10),
    )
    registry.register(
        "thanks",
        simple_chat_handler,
        HandlerConfig(model="gpt-4o-mini", max_tokens=128, temperature=0.7, timeout_seconds=10),
    )
    registry.register(
        "help",
        simple_chat_handler,
        HandlerConfig(model="gpt-4o-mini", max_tokens=512, temperature=0.5, timeout_seconds=15),
    )
    registry.register(
        "reset",
        simple_chat_handler,
        HandlerConfig(model="gpt-4o-mini", max_tokens=128, temperature=0.5, timeout_seconds=10),
    )
    registry.register(
        "knowledge_question",
        rag_handler,
        HandlerConfig(
            model="gpt-4o-mini",
            max_tokens=2048,
            temperature=0.3,
            timeout_seconds=45,
            requires_rag=True,
            cost_budget=0.05,
        ),
    )
    registry.register(
        "agent_task",
        agent_loop_handler,
        HandlerConfig(
            model="gpt-4o",
            max_tokens=4096,
            temperature=0.2,
            timeout_seconds=120,
            requires_tools=True,
            requires_approval=True,
            cost_budget=0.25,
        ),
    )
    registry.register(
        "order_lookup",
        agent_loop_handler,
        HandlerConfig(
            model="gpt-4o-mini",
            max_tokens=1024,
            temperature=0.1,
            timeout_seconds=30,
            requires_tools=True,
            cost_budget=0.02,
        ),
    )
    registry.register(
        "return_request",
        rag_handler,
        HandlerConfig(
            model="gpt-4o-mini",
            max_tokens=1024,
            temperature=0.3,
            timeout_seconds=30,
            requires_rag=True,
            cost_budget=0.02,
        ),
    )
    registry.register(
        "billing",
        rag_handler,
        HandlerConfig(
            model="gpt-4o-mini",
            max_tokens=1024,
            temperature=0.3,
            timeout_seconds=30,
            requires_rag=True,
            cost_budget=0.02,
        ),
    )
    registry.register(
        "technical_support",
        rag_handler,
        HandlerConfig(
            model="gpt-4o-mini",
            max_tokens=2048,
            temperature=0.3,
            timeout_seconds=45,
            requires_rag=True,
            cost_budget=0.05,
        ),
    )
    registry.register(
        "account",
        rag_handler,
        HandlerConfig(
            model="gpt-4o-mini",
            max_tokens=1024,
            temperature=0.3,
            timeout_seconds=30,
            requires_rag=True,
            cost_budget=0.02,
        ),
    )
    registry.register(
        "support_request",
        escalation_handler,
        HandlerConfig(timeout_seconds=10, cost_budget=0.0),
    )
    registry.register(
        "human_escalation",
        escalation_handler,
        HandlerConfig(timeout_seconds=10, cost_budget=0.0),
    )
    registry.register(
        "out_of_scope",
        out_of_scope_handler,
        HandlerConfig(
            model="gpt-4o-mini",
            max_tokens=256,
            temperature=0.5,
            timeout_seconds=10,
            cost_budget=0.001,
        ),
    )
    registry.register(
        "weather",
        agent_loop_handler,
        HandlerConfig(
            model="gpt-4o-mini",
            max_tokens=512,
            temperature=0.3,
            timeout_seconds=30,
            requires_tools=True,
            cost_budget=0.01,
        ),
    )
    registry.register(
        "stock",
        agent_loop_handler,
        HandlerConfig(
            model="gpt-4o-mini",
            max_tokens=512,
            temperature=0.1,
            timeout_seconds=30,
            requires_tools=True,
            cost_budget=0.01,
        ),
    )

    registry.set_default_intent("simple_chat")
    return registry


# ---------------------------------------------------------------------------
# Demo — compare handlers side by side
# ---------------------------------------------------------------------------

async def _demo() -> None:
    """Run the same query through multiple handlers and compare cost + latency."""
    from unittest.mock import AsyncMock, patch

    # ── Mock OpenAI so demo runs without a key ────────────────────────────
    _MOCK_CONTENT = "This is a demo response from the handler."

    class _MockUsage:
        prompt_tokens = 50
        completion_tokens = 30

    class _MockMessage:
        content = _MOCK_CONTENT
        tool_calls = None

    class _MockChoice:
        message = _MockMessage()

    class _MockResponse:
        choices = [_MockChoice()]
        usage = _MockUsage()

    async def _mock_create(**kwargs):
        return _MockResponse()

    _client.chat.completions.create = _mock_create  # type: ignore[method-assign]
    _vector_db._DOCS  # ensure docs exist

    registry = build_handler_registry()
    query = "What is your return policy?"

    print("\n" + "=" * 65)
    print("HANDLER COMPARISON DEMO")
    print("=" * 65)
    print(f"Query: \"{query}\"\n")
    print(f"{'Handler':<20} {'Tokens':>7} {'Cost (USD)':>12} {'Latency (ms)':>14}")
    print("-" * 56)

    for intent in ["simple_chat", "knowledge_question", "out_of_scope"]:
        rh = registry.get_handler(intent)
        t0 = time.monotonic()
        resp = await rh.handler(query, [], rh.config)
        elapsed = (time.monotonic() - t0) * 1000

        print(
            f"  {intent:<18} {resp.tokens_used:>7} "
            f"${resp.cost:>10.6f} {elapsed:>13.1f}"
        )

    print()


if __name__ == "__main__":
    asyncio.run(_demo())
