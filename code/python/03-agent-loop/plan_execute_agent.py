"""Plan-and-Execute agent: separate planning from execution.

Implements the three-phase pattern described in:
docs/02-the-agent-loop/03-planning-strategies.md

Phase 1 PLAN:   LLM generates a structured list of steps.
Phase 2 EXECUTE: Steps run in dependency order; independent steps run in parallel.
Phase 3 SYNTHESIZE: LLM combines all results into a final answer.

Run:
    python plan_execute_agent.py
"""

from __future__ import annotations

import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

from openai import OpenAI
from pydantic import ValidationError

from plan_schema import AgentPlan, PlanStep, StepResult

logger = logging.getLogger(__name__)

MAX_STEPS = 20


# ---------------------------------------------------------------------------
# Mock tools (same data as tools.py so the agent is self-contained)
# ---------------------------------------------------------------------------

_WEATHER_MOCK: dict[str, dict] = {
    "Apple":     {"price_usd": 192.35, "change_percent":  1.2, "currency": "USD"},  # kept for compat
    "Shanghai":  {"temperature_c": 22, "condition": "light rain",   "humidity_percent": 85, "wind_kph": 15},
    "London":    {"temperature_c": 14, "condition": "overcast",     "humidity_percent": 78, "wind_kph": 20},
    "New York":  {"temperature_c": 18, "condition": "partly cloudy","humidity_percent": 60, "wind_kph": 25},
    "Tokyo":     {"temperature_c": 22, "condition": "light rain",   "humidity_percent": 85, "wind_kph": 15},
    "Paris":     {"temperature_c": 16, "condition": "sunny",        "humidity_percent": 55, "wind_kph": 12},
    "Sydney":    {"temperature_c": 28, "condition": "clear",        "humidity_percent": 45, "wind_kph": 18},
}

_STOCK_MOCK: dict[str, dict] = {
    "AAPL":  {"price_usd": 192.35, "change_percent":  1.2, "currency": "USD", "weekly_change_percent": 3.1},
    "GOOGL": {"price_usd": 171.80, "change_percent": -0.5, "currency": "USD", "weekly_change_percent": 1.5},
    "MSFT":  {"price_usd": 415.10, "change_percent":  0.8, "currency": "USD", "weekly_change_percent": 2.4},
    "TSLA":  {"price_usd": 175.20, "change_percent": -2.3, "currency": "USD", "weekly_change_percent": -4.1},
    "AMZN":  {"price_usd": 188.40, "change_percent":  0.3, "currency": "USD", "weekly_change_percent": 0.8},
}

_NEWS_MOCK: dict[str, list[dict]] = {
    "AAPL": [
        {"headline": "Apple unveils M4 Ultra chip with record AI performance", "sentiment": "positive"},
        {"headline": "iPhone 17 pre-orders exceed expectations", "sentiment": "positive"},
    ],
    "MSFT": [
        {"headline": "Microsoft Azure revenue grows 33% year-over-year", "sentiment": "positive"},
        {"headline": "Copilot+ PC sales drive record Surface quarter", "sentiment": "positive"},
    ],
    "TSLA": [
        {"headline": "Tesla delays Robotaxi launch to Q3", "sentiment": "negative"},
        {"headline": "Tesla energy storage deployments hit record", "sentiment": "positive"},
    ],
}


def _get_weather(city: str) -> dict:
    key = city.split(",")[0].strip()
    data = _WEATHER_MOCK.get(key, {"temperature_c": 20, "condition": "clear", "humidity_percent": 55, "wind_kph": 10})
    return {"city": city, **data}


def _get_stock_price(ticker: str) -> dict:
    upper = ticker.upper()
    data = _STOCK_MOCK.get(upper, {"price_usd": 100.0, "change_percent": 0.0, "currency": "USD", "weekly_change_percent": 0.0})
    return {"ticker": upper, "market_status": "open", **data}


def _search_news(query: str) -> dict:
    ticker = query.upper().split()[0]
    articles = _NEWS_MOCK.get(ticker, [
        {"headline": f"No major news found for '{query}'", "sentiment": "neutral"}
    ])
    return {"query": query, "articles": articles, "count": len(articles)}


_TOOL_DISPATCH: dict[str, Any] = {
    "get_weather":     _get_weather,
    "get_stock_price": _get_stock_price,
    "search_news":     _search_news,
}

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather conditions for a city. "
                "Returns temperature (C), humidity, wind speed, and condition."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name, e.g. 'Tokyo' or 'New York'.",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": (
                "Get current stock price and weekly percentage change. "
                "Returns price_usd, change_percent, and weekly_change_percent."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": "Stock ticker symbol in uppercase, e.g. 'AAPL'.",
                    }
                },
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Search for recent news headlines about a company or topic. "
                "Returns a list of headlines with sentiment labels."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Company name or ticker, e.g. 'AAPL' or 'Apple'.",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Planner and synthesizer system prompts
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """\
You are a planning assistant. Break the user's request into concrete, sequential steps.

Rules:
- Each step must have a unique sequential step_number starting at 1.
- tool_name must be one of: get_weather, get_stock_price, search_news — or null for reasoning/synthesis steps.
- tool_params must contain the exact arguments to pass to the tool (or null for reasoning steps).
- depends_on lists the step_numbers this step must wait for.
- expected_output describes what a successful result looks like (at least 15 characters).
- Keep the plan to at most 10 steps. Consolidate where possible.

Output ONLY valid JSON matching this schema:
{
  "user_question": "<the user's question>",
  "steps": [
    {
      "step_number": 1,
      "description": "...",
      "tool_name": "get_stock_price" | null,
      "tool_params": {"ticker": "AAPL"} | null,
      "depends_on": [],
      "expected_output": "AAPL price and weekly percentage change"
    }
  ],
  "estimated_tool_calls": <int>
}"""

_EXECUTOR_SYSTEM = """\
You are an executor. Complete the given reasoning step using the provided context.
Be concise and factual. Return only the answer to this specific step."""

_SYNTHESIZER_SYSTEM = """\
You are a synthesizer. Given the original question and all execution results,
write a complete, well-structured answer. Cite specific numbers and data from
the results. Use markdown formatting where it improves readability."""


# ---------------------------------------------------------------------------
# PlanAndExecuteAgent
# ---------------------------------------------------------------------------


class PlanAndExecuteAgent:
    """Three-phase agent: Plan → Execute → Synthesize.

    Args:
        tools:      List of OpenAI tool definition dicts (the function schemas).
        model:      OpenAI model name for all LLM calls.
        max_steps:  Safety limit on plan length.
    """

    def __init__(
        self,
        tools: list[dict] = TOOLS,
        model: str = "gpt-4o",
        max_steps: int = MAX_STEPS,
    ) -> None:
        self.tools = tools
        self.model = model
        self.max_steps = max_steps
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        return self._client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> dict[str, Any]:
        """Run the full Plan → Execute → Synthesize cycle.

        Args:
            user_input: The user's question or request.

        Returns:
            A dict with keys:
                ``plan``    — list of PlanStep dicts
                ``results`` — list of StepResult dicts
                ``answer``  — final synthesised answer string
        """
        logger.info("[PlanAndExecute] Planning for: %s", user_input)
        plan = self._generate_plan(user_input)
        logger.info("[PlanAndExecute] Plan has %d steps", len(plan))

        results = self._execute_plan(plan)

        successes = sum(1 for r in results if r.success)
        logger.info("[PlanAndExecute] Executed %d/%d steps successfully", successes, len(results))

        answer = self._synthesize(user_input, plan, results)

        return {
            "plan": [s.model_dump() for s in plan],
            "results": [r.model_dump() for r in results],
            "answer": answer,
        }

    # ------------------------------------------------------------------
    # Phase 1: Plan
    # ------------------------------------------------------------------

    def _generate_plan(self, user_input: str) -> list[PlanStep]:
        """Ask the LLM to produce a structured JSON plan.

        Falls back to a single-step plan if the LLM output cannot be parsed.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _PLANNER_SYSTEM},
                {"role": "user", "content": user_input},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = response.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
            plan_obj = AgentPlan.model_validate(data)
            steps = plan_obj.steps[: self.max_steps]
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("[PlanAndExecute] Plan parsing failed (%s); using fallback", exc)
            steps = [
                PlanStep(
                    step_number=1,
                    description="Answer the user's question directly using available tools.",
                    tool_name=None,
                    tool_params=None,
                    depends_on=[],
                    expected_output="A complete answer to the user's question",
                )
            ]
        return steps

    # ------------------------------------------------------------------
    # Phase 2: Execute
    # ------------------------------------------------------------------

    def _execute_plan(self, steps: list[PlanStep]) -> list[StepResult]:
        """Execute plan steps, respecting dependencies and running independent
        steps in parallel.
        """
        results: dict[int, StepResult] = {}

        # Topological execution: keep finding steps whose dependencies are met
        remaining = list(steps)
        max_rounds = len(steps) + 1  # safety

        for _ in range(max_rounds):
            if not remaining:
                break
            ready = [s for s in remaining if all(dep in results for dep in s.depends_on)]
            if not ready:
                # Unresolvable — mark everything left as failed
                for s in remaining:
                    results[s.step_number] = StepResult(
                        step_number=s.step_number,
                        success=False,
                        error="Could not execute: unresolved dependencies",
                    )
                break

            # Execute all ready steps in parallel
            with ThreadPoolExecutor(max_workers=min(len(ready), 8)) as pool:
                futures = {
                    pool.submit(self._execute_step, step, results): step
                    for step in ready
                }
                for future in as_completed(futures):
                    step = futures[future]
                    result = future.result()
                    results[step.step_number] = result

            for s in ready:
                remaining.remove(s)

        return [results[s.step_number] for s in steps]

    def _execute_step(
        self, step: PlanStep, prior_results: dict[int, StepResult]
    ) -> StepResult:
        """Execute a single plan step: call a tool or ask the LLM to reason."""
        start = time.monotonic()
        try:
            if step.tool_name:
                data = self._call_tool(step.tool_name, step.tool_params or {})
            else:
                data = self._reason(step, prior_results)

            duration_ms = int((time.monotonic() - start) * 1000)
            return StepResult(
                step_number=step.step_number,
                success=True,
                data=data,
                duration_ms=duration_ms,
            )
        except Exception as exc:  # noqa: BLE001
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.warning("[PlanAndExecute] Step %d failed: %s", step.step_number, exc)
            return StepResult(
                step_number=step.step_number,
                success=False,
                error=str(exc),
                duration_ms=duration_ms,
            )

    def _call_tool(self, tool_name: str, params: dict[str, Any]) -> dict[str, Any]:
        """Dispatch a tool call to the local mock implementations."""
        fn = _TOOL_DISPATCH.get(tool_name)
        if fn is None:
            raise ValueError(f"Unknown tool: '{tool_name}'")
        result = fn(**params)
        if not isinstance(result, dict):
            result = {"result": result}
        return result

    def _reason(
        self, step: PlanStep, prior_results: dict[int, StepResult]
    ) -> dict[str, Any]:
        """Ask the LLM to reason about a non-tool step, given prior results."""
        context_parts = []
        for dep_num in step.depends_on:
            dep_result = prior_results.get(dep_num)
            if dep_result and dep_result.success:
                context_parts.append(
                    f"Step {dep_num} result:\n{json.dumps(dep_result.data, indent=2)}"
                )

        user_message = step.description
        if context_parts:
            user_message += "\n\nContext from previous steps:\n" + "\n\n".join(context_parts)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _EXECUTOR_SYSTEM},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
        )
        return {"reasoning": response.choices[0].message.content or ""}

    # ------------------------------------------------------------------
    # Phase 3: Synthesize
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        question: str,
        steps: list[PlanStep],
        results: list[StepResult],
    ) -> str:
        """Combine all step results into a final answer."""
        lines = [f"Original question: {question}\n"]
        for step, result in zip(steps, results):
            status = "✓" if result.success else "✗"
            lines.append(f"Step {step.step_number} [{status}]: {step.description}")
            if result.success and result.data:
                lines.append(f"Result: {json.dumps(result.data)}")
            elif result.error:
                lines.append(f"Error: {result.error}")
            lines.append("")

        context = "\n".join(lines)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYNTHESIZER_SYSTEM},
                {"role": "user", "content": context},
            ],
            temperature=0.5,
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    agent = PlanAndExecuteAgent()
    query = "Compare Apple and Microsoft stock performance this week"

    print(f"\n{'=' * 60}")
    print(f"Query: {query}")
    print("=" * 60)

    output = agent.run(query)

    print("\n--- Plan ---")
    for step in output["plan"]:
        deps = f" (depends on {step['depends_on']})" if step["depends_on"] else ""
        tool = f" [{step['tool_name']}]" if step["tool_name"] else " [reasoning]"
        print(f"  {step['step_number']}. {step['description']}{tool}{deps}")

    print("\n--- Results ---")
    for result in output["results"]:
        status = "✓" if result["success"] else "✗"
        duration = result.get("duration_ms", 0)
        print(f"  Step {result['step_number']} [{status}] ({duration}ms)")
        if result["success"] and result.get("data"):
            print(f"    {json.dumps(result['data'])}")
        elif result.get("error"):
            print(f"    ERROR: {result['error']}")

    print("\n--- Final Answer ---")
    print(output["answer"])


if __name__ == "__main__":
    main()
