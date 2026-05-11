"""Delegation multi-agent system: coordinator delegates to specialist agents.

The coordinator runs a standard ReAct loop, but its "tools" are other agents.
Each specialist agent is a focused single-purpose LLM with its own tools and
system prompt.

Implements Pattern 1 from:
docs/02-the-agent-loop/04-multi-agent-patterns.md

Run:
    python delegation_agent.py
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

MAX_SPECIALIST_ITERATIONS = 5
MAX_DELEGATIONS_PER_AGENT = 3


# ---------------------------------------------------------------------------
# Mock tools for specialist agents
# ---------------------------------------------------------------------------

_STOCK_MOCK: dict[str, dict] = {
    "AAPL":  {"price_usd": 192.35, "change_percent":  1.2, "weekly_change_percent":  3.1},
    "MSFT":  {"price_usd": 415.10, "change_percent":  0.8, "weekly_change_percent":  2.4},
    "GOOGL": {"price_usd": 171.80, "change_percent": -0.5, "weekly_change_percent":  1.5},
    "TSLA":  {"price_usd": 175.20, "change_percent": -2.3, "weekly_change_percent": -4.1},
    "AMZN":  {"price_usd": 188.40, "change_percent":  0.3, "weekly_change_percent":  0.8},
}

_FINANCIALS_MOCK: dict[str, dict] = {
    "Apple": {
        "revenue_ttm_b": 391.0, "net_income_b": 97.0,
        "gross_margin_pct": 45.0, "yoy_revenue_growth_pct": 5.1,
    },
    "Microsoft": {
        "revenue_ttm_b": 245.0, "net_income_b": 88.0,
        "gross_margin_pct": 70.0, "yoy_revenue_growth_pct": 15.7,
    },
    "Google": {
        "revenue_ttm_b": 350.0, "net_income_b": 76.0,
        "gross_margin_pct": 58.0, "yoy_revenue_growth_pct": 14.3,
    },
}

_NEWS_MOCK: dict[str, list[str]] = {
    "Apple": [
        "Apple reports record services revenue of $25B in Q3.",
        "Apple Vision Pro sales expected to reach 1M units by year-end.",
        "iPhone 17 pre-orders exceed 15M in opening weekend.",
    ],
    "Microsoft": [
        "Microsoft Copilot adds 5M enterprise users in Q3.",
        "Azure cloud revenue surpasses $30B quarterly run rate.",
        "Microsoft acquires AI research lab for $2.1B.",
    ],
}


def _get_stock_price(ticker: str) -> dict:
    upper = ticker.upper()
    data = _STOCK_MOCK.get(upper, {"price_usd": 100.0, "change_percent": 0.0})
    return {"ticker": upper, **data}


def _get_company_financials(company: str) -> dict:
    title = company.title()
    # Try exact match first, then partial
    for key in _FINANCIALS_MOCK:
        if title.lower() in key.lower():
            return {"company": key, **_FINANCIALS_MOCK[key]}
    return {"company": company, "error": "No financials found"}


def _web_search(query: str) -> dict:
    # Mock: return canned headlines matching the query
    for key, articles in _NEWS_MOCK.items():
        if key.lower() in query.lower():
            return {"query": query, "results": articles}
    return {"query": query, "results": [f"No mock results found for '{query}'"]}


def _fetch_article(url: str) -> dict:
    return {
        "url": url,
        "title": "Mock article",
        "content": (
            "This is a mock article body. In a real implementation this would "
            "contain the full text of the article at the given URL."
        ),
    }


# OpenAI tool schemas
_FINANCE_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price and weekly change for a ticker symbol.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Uppercase ticker symbol, e.g. 'AAPL'."}
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_financials",
            "description": (
                "Get TTM financials for a company: revenue, net income, "
                "gross margin, and year-over-year revenue growth."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "company": {"type": "string", "description": "Company name, e.g. 'Apple'."}
                },
                "required": ["company"],
            },
        },
    },
]

_RESEARCH_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for recent news and information about a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query string."}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_article",
            "description": "Fetch the full text of a web article by URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL of the article."}
                },
                "required": ["url"],
            },
        },
    },
]

_TOOL_DISPATCH = {
    "get_stock_price":       lambda args: _get_stock_price(args["ticker"]),
    "get_company_financials": lambda args: _get_company_financials(args["company"]),
    "web_search":            lambda args: _web_search(args["query"]),
    "fetch_article":         lambda args: _fetch_article(args["url"]),
}


# ---------------------------------------------------------------------------
# Structured handoff
# ---------------------------------------------------------------------------

@dataclass
class Handoff:
    """Carries the minimum context a specialist needs — nothing more."""

    from_agent: str
    to_agent: str
    task: str
    context: dict[str, Any] = field(default_factory=dict)

    def to_user_message(self) -> str:
        if self.context:
            ctx_text = json.dumps(self.context, indent=2)
            return f"{self.task}\n\nContext provided:\n{ctx_text}"
        return self.task


@dataclass
class HandoffResult:
    """Response from a specialist back to the coordinator."""

    from_agent: str
    status: str   # "complete" | "failed" | "need_clarification"
    result: str
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# SpecialistAgent
# ---------------------------------------------------------------------------


class SpecialistAgent:
    """A focused single-purpose agent with its own tools and system prompt.

    Args:
        name:          Short identifier, e.g. "finance".
        role:          Human-readable role description.
        tools:         OpenAI tool definitions available to this specialist.
        system_prompt: System prompt that scopes the specialist's behaviour.
        task_guidance: One-sentence description of what tasks to delegate here.
    """

    def __init__(
        self,
        name: str,
        role: str,
        tools: list[dict],
        system_prompt: str,
        task_guidance: str = "",
    ) -> None:
        self.name = name
        self.role = role
        self.tools = tools
        self.system_prompt = system_prompt
        self.task_guidance = task_guidance
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        return self._client

    def run(self, handoff: Handoff) -> HandoffResult:
        """Execute a delegated task within the specialist's domain.

        Runs a lightweight ReAct loop bounded by MAX_SPECIALIST_ITERATIONS.
        """
        logger.info("[%s] Starting task: %s", self.name, handoff.task[:80])
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": handoff.to_user_message()},
        ]

        try:
            for iteration in range(1, MAX_SPECIALIST_ITERATIONS + 1):
                kwargs: dict[str, Any] = {
                    "model": "gpt-4o",
                    "messages": messages,
                }
                if self.tools:
                    kwargs["tools"] = self.tools
                    kwargs["tool_choice"] = "auto"

                response = self.client.chat.completions.create(**kwargs)
                msg = response.choices[0].message

                assistant_msg: dict = {"role": "assistant", "content": msg.content}
                if msg.tool_calls:
                    assistant_msg["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in msg.tool_calls
                    ]
                messages.append(assistant_msg)

                if not msg.tool_calls:
                    logger.info("[%s] Task complete in %d iteration(s)", self.name, iteration)
                    return HandoffResult(
                        from_agent=self.name,
                        status="complete",
                        result=msg.content or "",
                    )

                # Execute tool calls
                for tc in msg.tool_calls:
                    args = json.loads(tc.function.arguments)
                    fn = _TOOL_DISPATCH.get(tc.function.name)
                    if fn:
                        try:
                            result = fn(args)
                            content = json.dumps(result)
                        except Exception as exc:  # noqa: BLE001
                            content = json.dumps({"error": str(exc)})
                    else:
                        content = json.dumps({"error": f"Unknown tool: {tc.function.name}"})

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content,
                    })

            return HandoffResult(
                from_agent=self.name,
                status="complete",
                result=messages[-1].get("content", ""),
            )

        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] Task failed: %s", self.name, exc)
            return HandoffResult(
                from_agent=self.name,
                status="failed",
                result="",
                error=str(exc),
            )


# ---------------------------------------------------------------------------
# CoordinatorAgent
# ---------------------------------------------------------------------------

_COORDINATOR_SYSTEM = """\
You are a coordinator agent. Your ONLY job is to delegate tasks to specialist agents.
Do NOT answer questions directly. Do NOT perform analysis yourself.

Your specialists:
- delegate_to_finance_agent:   financial data, stock prices, earnings, revenue figures
- delegate_to_research_agent:  web search, news, recent events, background research
- delegate_to_writing_agent:   synthesis, summaries, reports, polished prose

Rules:
1. Always delegate immediately — never answer from your own knowledge.
2. If a request spans multiple domains, delegate to multiple specialists.
3. Give each specialist a complete, self-contained task with all the context they need.
4. After all specialists have responded, synthesize their results into a final answer.
5. If a specialist reports failure, acknowledge it in your final answer and continue."""


class CoordinatorAgent:
    """Routes user requests to specialist agents and synthesizes their results.

    Specialists are exposed as tools; the coordinator runs a standard ReAct
    loop where every tool call is a delegation to a specialist.
    """

    def __init__(self, specialists: dict[str, SpecialistAgent]) -> None:
        self.specialists = specialists
        self.tools = self._build_delegation_tools()
        self._delegation_counts: dict[str, int] = {name: 0 for name in specialists}
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        return self._client

    def _build_delegation_tools(self) -> list[dict]:
        tools = []
        for name, agent in self.specialists.items():
            tools.append({
                "type": "function",
                "function": {
                    "name": f"delegate_to_{name}_agent",
                    "description": (
                        f"Delegate a task to the {name} specialist. "
                        f"{agent.task_guidance}"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "task": {
                                "type": "string",
                                "description": (
                                    f"Complete, self-contained task for the {name} agent. "
                                    "Include all relevant context."
                                ),
                            },
                            "context": {
                                "type": "object",
                                "description": "Optional structured context to pass (key-value pairs).",
                            },
                        },
                        "required": ["task"],
                    },
                },
            })
        return tools

    def _dispatch_delegation(self, tool_name: str, args: dict) -> str:
        """Execute a delegation tool call and return a result string."""
        agent_name = tool_name.replace("delegate_to_", "").replace("_agent", "")
        specialist = self.specialists.get(agent_name)
        if specialist is None:
            return json.dumps({"status": "failed", "error": f"No specialist named '{agent_name}'"})

        count = self._delegation_counts.get(agent_name, 0)
        if count >= MAX_DELEGATIONS_PER_AGENT:
            return json.dumps({
                "status": "failed",
                "error": f"Max delegations ({MAX_DELEGATIONS_PER_AGENT}) reached for {agent_name}",
            })
        self._delegation_counts[agent_name] = count + 1

        handoff = Handoff(
            from_agent="coordinator",
            to_agent=agent_name,
            task=args.get("task", ""),
            context=args.get("context", {}),
        )
        result = specialist.run(handoff)
        return json.dumps({
            "status": result.status,
            "result": result.result,
            "error": result.error,
        })

    def run(self, user_input: str) -> dict[str, Any]:
        """Run the coordinator ReAct loop.

        Returns:
            A dict with:
                ``answer``       — final synthesised answer
                ``delegations``  — list of {agent, task, result} records
        """
        self._delegation_counts = {name: 0 for name in self.specialists}
        delegations: list[dict] = []

        messages: list[dict] = [
            {"role": "system", "content": _COORDINATOR_SYSTEM},
            {"role": "user", "content": user_input},
        ]

        max_iters = MAX_DELEGATIONS_PER_AGENT * len(self.specialists) + 2
        for iteration in range(1, max_iters + 1):
            logger.debug("[coordinator] Iteration %d", iteration)
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
            )
            msg = response.choices[0].message
            assistant_msg: dict = {"role": "assistant", "content": msg.content}
            if msg.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in msg.tool_calls
                ]
            messages.append(assistant_msg)

            if not msg.tool_calls:
                return {"answer": msg.content or "", "delegations": delegations}

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                agent_name = tc.function.name.replace("delegate_to_", "").replace("_agent", "")
                task = args.get("task", "")

                logger.info("[coordinator] → [%s]: %s", agent_name, task[:80])
                tool_result_str = self._dispatch_delegation(tc.function.name, args)
                tool_result = json.loads(tool_result_str)

                delegations.append({
                    "agent": agent_name,
                    "task": task,
                    "result": tool_result.get("result") or tool_result.get("error", ""),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result_str,
                })

        return {"answer": "Max iterations reached.", "delegations": delegations}


# ---------------------------------------------------------------------------
# Specialist factory
# ---------------------------------------------------------------------------

def build_specialists() -> dict[str, SpecialistAgent]:
    """Build the three default specialist agents."""
    finance = SpecialistAgent(
        name="finance",
        role="Financial analyst",
        tools=_FINANCE_TOOLS,
        system_prompt="""\
You are a financial analyst specialist. Use your tools to retrieve accurate
financial data. Always cite specific numbers (prices, percentages, revenue
figures). Never speculate — only report what the tools return.""",
        task_guidance="Use for stock prices, company revenue, earnings, and financial metrics.",
    )

    research = SpecialistAgent(
        name="research",
        role="Research analyst",
        tools=_RESEARCH_TOOLS,
        system_prompt="""\
You are a research specialist. Use your tools to find relevant news and
background information. Summarise key findings clearly. Attribute claims
to specific sources. Do not include your own opinions.""",
        task_guidance="Use for recent news, background research, and fact-finding.",
    )

    writing = SpecialistAgent(
        name="writing",
        role="Content writer",
        tools=[],  # Pure LLM — no tools
        system_prompt="""\
You are a professional writer and synthesizer. You receive structured data
and findings and turn them into polished, well-organised prose. Your output
should be suitable for a business audience. Use clear headings and bullet
points where appropriate.""",
        task_guidance="Use for summaries, reports, and polished written content.",
    )

    return {"finance": finance, "research": research, "writing": writing}


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    specialists = build_specialists()
    coordinator = CoordinatorAgent(specialists)

    query = "Research Apple's financial performance and write a summary"

    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print("=" * 60)

    output = coordinator.run(query)

    print("\n--- Delegation Trace ---")
    for i, d in enumerate(output["delegations"], 1):
        preview = (d["result"] or "")[:150].replace("\n", " ")
        print(f"  {i}. [{d['agent']}] {d['task'][:60]}")
        print(f"     → {preview}…")

    print(f"\n--- Final Answer ---\n{output['answer']}")


if __name__ == "__main__":
    main()
