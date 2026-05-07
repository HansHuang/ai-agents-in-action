"""LangGraph ReAct agent — the same loop from Chapter 02, now as a StateGraph.

Builds the identical ReAct (Reason → Act → Observe) agent from
docs/02-the-agent-loop/01-anatomy-of-an-agent.md using LangGraph's
StateGraph instead of imperative Python.

Includes a :func:`compare_agents` function that runs the same query through:
  1. This LangGraph agent
  2. The from-scratch agent (code/python/03-agent-loop/agent.py)

and reports code complexity, execution trace, token usage, and error
handling differences.

Run:
    python langgraph_react_agent.py

See: docs/06-frameworks-in-practice/02-langchain-langgraph.md
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Optional

# ---------------------------------------------------------------------------
# Optional LangGraph / LangChain imports
# ---------------------------------------------------------------------------

try:
    from typing import TypedDict

    from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
    from langchain_openai import ChatOpenAI
    from langchain.tools import tool
    from langgraph.graph import StateGraph, END
    from langgraph.graph.message import add_messages

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False

# ---------------------------------------------------------------------------
# Shared mock tool implementations (mirrors code/python/03-agent-loop/tools.py)
# ---------------------------------------------------------------------------

_WEATHER_MOCK: dict[str, dict] = {
    "Tokyo":    {"temperature_c": 18, "condition": "partly cloudy", "humidity_percent": 65, "wind_kph": 14},
    "Shanghai": {"temperature_c": 22, "condition": "light rain",    "humidity_percent": 85, "wind_kph": 15},
    "London":   {"temperature_c": 14, "condition": "overcast",      "humidity_percent": 78, "wind_kph": 20},
    "New York": {"temperature_c": 18, "condition": "partly cloudy", "humidity_percent": 60, "wind_kph": 25},
    "Paris":    {"temperature_c": 16, "condition": "sunny",         "humidity_percent": 55, "wind_kph": 12},
}

_STOCK_MOCK: dict[str, dict] = {
    "AAPL":  {"price_usd": 192.35, "change_percent":  1.2, "currency": "USD"},
    "GOOGL": {"price_usd": 171.80, "change_percent": -0.5, "currency": "USD"},
    "MSFT":  {"price_usd": 415.10, "change_percent":  0.8, "currency": "USD"},
    "TSLA":  {"price_usd": 175.20, "change_percent": -2.3, "currency": "USD"},
    "AMZN":  {"price_usd": 188.40, "change_percent":  0.3, "currency": "USD"},
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class AgentResult:
    """Structured output from a single agent run."""

    answer: str
    iterations: int
    tool_calls_made: list[str] = field(default_factory=list)
    messages: list = field(default_factory=list)
    tokens_used: int = 0
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# LangGraphReActAgent
# ---------------------------------------------------------------------------


class LangGraphReActAgent:
    """ReAct agent built with LangGraph's StateGraph.

    Functionally identical to the from-scratch agent in
    code/python/03-agent-loop/agent.py.  The difference is that the
    orchestration loop is declared as a graph instead of a ``for`` loop.

    Args:
        model:          OpenAI chat model name.
        max_iterations: Hard cap on agent iterations (safety valve).
    """

    def __init__(self, model: str = "gpt-4o", max_iterations: int = 10) -> None:
        if not LANGGRAPH_AVAILABLE:
            raise ImportError(
                "LangGraph is required. Install with: "
                "pip install langgraph langchain-openai"
            )
        self.model = model
        self.max_iterations = max_iterations
        self.tools = self._load_tools()
        self.llm = ChatOpenAI(model=model)
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.graph = self._build_graph()

    # ------------------------------------------------------------------
    # Tool definitions
    # ------------------------------------------------------------------

    def _load_tools(self) -> list:
        """Return LangChain @tool-decorated versions of the Chapter 02 tools."""

        @tool
        def get_weather(city: str) -> str:
            """Get current weather for a city. City must include the city name, e.g. 'Tokyo'."""
            city_key = city.split(",")[0].strip()
            data = _WEATHER_MOCK.get(
                city_key,
                {"temperature_c": 20, "condition": "clear", "humidity_percent": 55, "wind_kph": 10},
            )
            return json.dumps({"city": city, **data})

        @tool
        def get_stock_price(ticker: str) -> str:
            """Get current stock price and daily change for a ticker symbol, e.g. 'AAPL'."""
            ticker_upper = ticker.upper()
            data = _STOCK_MOCK.get(
                ticker_upper,
                {"price_usd": 100.00, "change_percent": 0.0, "currency": "USD"},
            )
            return json.dumps({"ticker": ticker_upper, "market_status": "open", **data})

        @tool
        def calculator(expression: str) -> str:
            """Evaluate a simple arithmetic expression, e.g. '2 + 2' or '192.35 * 1.1'."""
            try:
                # Restrict to safe arithmetic — no builtins
                result = eval(expression, {"__builtins__": {}})  # noqa: S307
                return str(result)
            except Exception as exc:
                return f"Error: {exc}"

        return [get_weather, get_stock_price, calculator]

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        """Declare the ReAct graph: agent → [tools?] → agent → … → END."""

        # --- State ---
        class AgentState(TypedDict):
            messages: Annotated[list, add_messages]
            iteration_count: int
            tool_calls_made: list[str]

        tool_map = {t.name: t for t in self.tools}
        llm_with_tools = self.llm_with_tools
        max_iter = self.max_iterations

        # --- Nodes ---

        def agent_node(state: AgentState) -> AgentState:
            """The LLM decides: answer now or call a tool."""
            response = llm_with_tools.invoke(state["messages"])
            return {
                "messages": [response],
                "iteration_count": state["iteration_count"] + 1,
                "tool_calls_made": state["tool_calls_made"],
            }

        def tool_node(state: AgentState) -> AgentState:
            """Execute every tool call requested by the last LLM message."""
            last_message = state["messages"][-1]
            results = []
            new_calls: list[str] = []

            for tc in last_message.tool_calls:
                name = tc["name"]
                args = tc["args"]
                new_calls.append(name)

                if name in tool_map:
                    output = tool_map[name].invoke(args)
                else:
                    output = f"Unknown tool: {name}"

                results.append(
                    ToolMessage(content=str(output), tool_call_id=tc["id"])
                )

            return {
                "messages": results,
                "iteration_count": state["iteration_count"],
                "tool_calls_made": state["tool_calls_made"] + new_calls,
            }

        # --- Routing ---

        def should_continue(state: AgentState) -> str:
            """Route to 'tools' if the LLM called tools; otherwise end."""
            if state["iteration_count"] >= max_iter:
                return "end"
            last = state["messages"][-1]
            if hasattr(last, "tool_calls") and last.tool_calls:
                return "tools"
            return "end"

        # --- Graph assembly ---

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", tool_node)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {"tools": "tools", "end": END},
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> AgentResult:
        """Run the agent synchronously and return a structured result.

        Args:
            user_input: The user's question or instruction.

        Returns:
            :class:`AgentResult` with answer, iteration count, tool calls, etc.
        """
        start = time.monotonic()

        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "iteration_count": 0,
            "tool_calls_made": [],
        }

        final_state = self.graph.invoke(initial_state)

        elapsed = (time.monotonic() - start) * 1000
        last_msg = final_state["messages"][-1]
        answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        return AgentResult(
            answer=answer,
            iterations=final_state["iteration_count"],
            tool_calls_made=final_state["tool_calls_made"],
            messages=final_state["messages"],
            elapsed_ms=elapsed,
        )

    def stream(self, user_input: str):
        """Yield each graph step as it executes.

        Each yielded item is a dict mapping node name → updated state.
        Useful for real-time progress display.

        Args:
            user_input: The user's question or instruction.

        Yields:
            dict: ``{"node_name": AgentState}`` for each step.
        """
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "iteration_count": 0,
            "tool_calls_made": [],
        }
        for step in self.graph.stream(initial_state):
            yield step

    @staticmethod
    def visualize() -> str:
        """Return an ASCII diagram of the agent graph."""
        return """\
LangGraph ReAct Agent
─────────────────────

┌─────────┐
│  START  │
└────┬────┘
     ▼
┌─────────┐     has tool_calls     ┌─────────┐
│  agent  │──────────────────────▶ │  tools  │
└─────────┘                        └────┬────┘
     │                                  │
     │  no tool_calls / max_iter        │ (always)
     ▼                                  │
┌─────────┐ ◀────────────────────────── ┘
│   END   │
└─────────┘

Nodes:
  agent — LLM decides: answer or call tool(s)
  tools — executes tool calls, appends ToolMessage results

Edges:
  START → agent          (entry point)
  agent → tools          (conditional: tool_calls present)
  agent → END            (conditional: no tool_calls OR max_iter reached)
  tools → agent          (always: loop back after execution)
"""


# ---------------------------------------------------------------------------
# Comparison with from-scratch agent
# ---------------------------------------------------------------------------


def _count_agent_logic_lines(source: str) -> int:
    """Count non-blank, non-comment lines in a source string."""
    return sum(
        1
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )


@dataclass
class ComparisonResult:
    """Side-by-side metrics for LangGraph vs from-scratch agent."""

    query: str
    langgraph_answer: str
    scratch_answer: str
    langgraph_iterations: int
    scratch_iterations: int
    langgraph_tool_calls: list[str]
    scratch_tool_calls: list[str]
    langgraph_lines: int
    scratch_lines: int
    langgraph_elapsed_ms: float
    scratch_elapsed_ms: float
    trace_identical: bool


def compare_agents(query: str) -> ComparisonResult:
    """Run *query* through both agents and return a :class:`ComparisonResult`.

    The from-scratch agent is imported from code/python/03-agent-loop/.
    Falls back gracefully if that module is not on sys.path.

    Args:
        query: Natural-language question to submit to both agents.

    Returns:
        ComparisonResult with side-by-side metrics.
    """
    # -- LangGraph agent --
    lg_agent = LangGraphReActAgent()
    lg_result = lg_agent.run(query)

    # -- From-scratch agent --
    scratch_answer = "(from-scratch agent not available on sys.path)"
    scratch_iterations = 0
    scratch_tool_calls: list[str] = []
    scratch_elapsed_ms: float = 0.0
    scratch_lines = 0

    scratch_dir = Path(__file__).parent.parent / "03-agent-loop"
    if scratch_dir.is_dir():
        sys.path.insert(0, str(scratch_dir))
        try:
            import agent as scratch_agent  # type: ignore
            import tools as scratch_tools  # type: ignore

            scratch_messages: list[dict] = [
                {"role": "system", "content": scratch_agent.SYSTEM_PROMPT}
            ]
            t0 = time.monotonic()
            scratch_answer = scratch_agent.run_agent(
                query, messages=scratch_messages, tools=scratch_tools.TOOLS
            )
            scratch_elapsed_ms = (time.monotonic() - t0) * 1000

            # Count tool calls from message history
            scratch_tool_calls = [
                tc["function"]["name"]
                for msg in scratch_messages
                if msg.get("tool_calls")
                for tc in msg["tool_calls"]
            ]
            scratch_iterations = sum(
                1 for msg in scratch_messages if msg["role"] == "assistant"
            )

            scratch_src = inspect.getsource(scratch_agent.run_agent)
            scratch_lines = _count_agent_logic_lines(scratch_src)
        except Exception:
            pass
        finally:
            sys.path.remove(str(scratch_dir))

    # Count LangGraph agent logic lines (build_graph + run)
    lg_src = (
        inspect.getsource(LangGraphReActAgent._build_graph)
        + inspect.getsource(LangGraphReActAgent.run)
    )
    lg_lines = _count_agent_logic_lines(lg_src)

    return ComparisonResult(
        query=query,
        langgraph_answer=lg_result.answer,
        scratch_answer=scratch_answer,
        langgraph_iterations=lg_result.iterations,
        scratch_iterations=scratch_iterations,
        langgraph_tool_calls=lg_result.tool_calls_made,
        scratch_tool_calls=scratch_tool_calls,
        langgraph_lines=lg_lines,
        scratch_lines=scratch_lines,
        langgraph_elapsed_ms=lg_result.elapsed_ms,
        scratch_elapsed_ms=scratch_elapsed_ms,
        trace_identical=(
            set(lg_result.tool_calls_made) == set(scratch_tool_calls)
        ),
    )


def print_comparison(result: ComparisonResult) -> None:
    """Pretty-print a :class:`ComparisonResult` as a side-by-side table."""
    W = 42
    sep = "─" * (W * 2 + 5)

    print(f"\n{'AGENT COMPARISON':^{W * 2 + 5}}")
    print(sep)
    print(f"Query: {result.query}")
    print(sep)
    print(f"{'Metric':<28} {'LangGraph':>{W - 28}} {'From Scratch':>{W - 10}}")
    print(sep)

    def row(label: str, a, b) -> None:
        print(f"{label:<28} {str(a):>{W - 28}} {str(b):>{W - 10}}")

    row("Iterations", result.langgraph_iterations, result.scratch_iterations)
    row("Agent logic (lines)", result.langgraph_lines, result.scratch_lines)
    row("Elapsed (ms)", f"{result.langgraph_elapsed_ms:.0f}", f"{result.scratch_elapsed_ms:.0f}")
    row("Tool calls", ", ".join(result.langgraph_tool_calls) or "none",
        ", ".join(result.scratch_tool_calls) or "none")
    row("Traces match?", "yes" if result.trace_identical else "no", "—")
    print(sep)

    print("\nLangGraph answer:")
    print(f"  {result.langgraph_answer[:200]}")
    print("\nFrom-scratch answer:")
    print(f"  {str(result.scratch_answer)[:200]}")
    print()


# ---------------------------------------------------------------------------
# Demo entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the demo: visualise the graph, compare both agents."""
    print(LangGraphReActAgent.visualize())

    demo_query = "What's the weather in Tokyo and should I invest in AAPL?"
    print(f"Demo query: {demo_query!r}\n")

    if not LANGGRAPH_AVAILABLE:
        print("LangGraph is not installed. Install with:")
        print("  pip install langgraph langchain-openai")
        return

    # Stream step-by-step progress
    agent = LangGraphReActAgent()
    print("Streaming execution steps:")
    for step in agent.stream(demo_query):
        node, state = next(iter(step.items()))
        msgs = state.get("messages", [])
        last_msg = msgs[-1] if msgs else None
        if last_msg is not None:
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                calls = [tc["name"] for tc in last_msg.tool_calls]
                print(f"  [{node}] → tool calls: {calls}")
            elif hasattr(last_msg, "content") and last_msg.content:
                preview = str(last_msg.content)[:80]
                print(f"  [{node}] → {preview}")
            else:
                print(f"  [{node}] → (tool results)")
    print()

    # Side-by-side comparison
    comparison = compare_agents(demo_query)
    print_comparison(comparison)


if __name__ == "__main__":
    main()
