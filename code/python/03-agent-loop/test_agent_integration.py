"""Integration tests for the ReAct agent loop using a MockLLM.

MockLLM accepts a list of pre-programmed responses. Each response is either:
  - A tool call response (tool name + arguments)
  - A content response (final answer string)

The mock simulates the full OpenAI response object structure so the agent
code runs unchanged — no real API calls, no OPENAI_API_KEY required.

Test cases:
  1. Full weather query lifecycle (tool → answer)
  2. Agent retries when a tool fails on the first attempt
  3. Agent switches between two different tools in one session
"""

from __future__ import annotations

import json
from typing import Union
from unittest.mock import MagicMock

import pytest

from agent import SYSTEM_PROMPT, run_agent
from tools import TOOLS, get_weather, get_stock_price


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------


class MockLLM:
    """Simulates the OpenAI chat completions API with pre-programmed responses.

    Each entry in ``responses`` is a mock ChatCompletion object returned
    in sequence. Raises AssertionError if called more times than planned.

    Usage::

        llm = MockLLM([
            make_tool_response("get_weather", {"city": "Shanghai"}),
            make_content_response("Shanghai is sunny."),
        ])
        # Pass as the `create` callable to the agent under test.
    """

    def __init__(self, responses: list) -> None:
        self.responses = responses
        self.call_count = 0

    def create(self, **kwargs) -> object:  # noqa: ANN001
        if self.call_count >= len(self.responses):
            raise AssertionError(
                f"MockLLM called {self.call_count + 1} time(s) but only "
                f"{len(self.responses)} response(s) were programmed."
            )
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp


# ---------------------------------------------------------------------------
# Response builders (same pattern as test_agent.py for consistency)
# ---------------------------------------------------------------------------


def make_content_response(content: str) -> MagicMock:
    """Build a mock final-answer response."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def make_tool_response(
    tool_name: str,
    args: dict,
    call_id: str = "call_mock1",
) -> MagicMock:
    """Build a mock tool-call response."""
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function.name = tool_name
    tc.function.arguments = json.dumps(args)

    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tc]

    choice = MagicMock()
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_messages() -> list[dict]:
    """Fresh message history starting with only the system prompt."""
    return [{"role": "system", "content": SYSTEM_PROMPT}]


@pytest.fixture
def real_registry() -> dict:
    """Registry using the actual (mocked-data) tool implementations."""
    return {"get_weather": get_weather, "get_stock_price": get_stock_price}


# ---------------------------------------------------------------------------
# Test 1: Full weather query lifecycle
# ---------------------------------------------------------------------------


def test_full_weather_query_lifecycle(monkeypatch, base_messages, real_registry):
    """Complete lifecycle: 2 LLM calls, 1 tool executed, correct final answer."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    llm = MockLLM([
        make_tool_response("get_weather", {"city": "Shanghai"}, call_id="call_1"),
        make_content_response("Shanghai is 22°C with light rain. Bring an umbrella!"),
    ])

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = llm.create
    monkeypatch.setattr("agent.OpenAI", lambda **kw: mock_client)

    result = run_agent(
        "Weather in Shanghai?",
        messages=base_messages,
        tools=TOOLS,
        registry=real_registry,
    )

    assert llm.call_count == 2
    tool_msgs = [m for m in base_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    weather = json.loads(tool_msgs[0]["content"])
    assert weather["temperature_c"] == 22
    assert "umbrella" in result.lower() or "rain" in result.lower() or "22" in result


# ---------------------------------------------------------------------------
# Test 2: Agent retries when tool fails on first attempt
# ---------------------------------------------------------------------------


def test_agent_retries_on_tool_failure(monkeypatch, base_messages):
    """
    Sequence: call(weather) → call(weather) → final answer.
    Tool fails on the first execution, succeeds on the second.
    Agent should make 3 LLM calls and 2 tool executions.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    llm = MockLLM([
        make_tool_response("get_weather", {"city": "London"}, call_id="call_a"),
        make_tool_response("get_weather", {"city": "London"}, call_id="call_b"),
        make_content_response("London is 14°C and overcast."),
    ])

    call_count = {"n": 0}

    def sometimes_fails_weather(city: str) -> dict:
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("Connection timeout")
        return get_weather(city)

    registry = {"get_weather": sometimes_fails_weather}
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = llm.create
    monkeypatch.setattr("agent.OpenAI", lambda **kw: mock_client)

    result = run_agent(
        "What's the weather in London?",
        messages=base_messages,
        tools=TOOLS,
        registry=registry,
    )

    assert llm.call_count == 3
    assert call_count["n"] == 2

    tool_msgs = [m for m in base_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2

    # First tool message is an error, second is a valid result.
    first = json.loads(tool_msgs[0]["content"])
    assert "error" in first

    second = json.loads(tool_msgs[1]["content"])
    assert "temperature_c" in second

    assert result != ""


# ---------------------------------------------------------------------------
# Test 3: Agent switches between two different tools
# ---------------------------------------------------------------------------


def test_agent_switches_tools(monkeypatch, base_messages, real_registry):
    """
    Sequence: get_weather → get_stock_price → final answer.
    Verifies both tools are called and the final answer synthesises both results.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    llm = MockLLM([
        make_tool_response("get_weather",    {"city": "Shanghai"},  call_id="call_w"),
        make_tool_response("get_stock_price", {"ticker": "AAPL"}, call_id="call_s"),
        make_content_response(
            "Shanghai is 22°C with light rain. Apple (AAPL) is at $192.35, up 1.2%."
        ),
    ])

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = llm.create
    monkeypatch.setattr("agent.OpenAI", lambda **kw: mock_client)

    result = run_agent(
        "What's the weather in Shanghai and Apple's stock price?",
        messages=base_messages,
        tools=TOOLS,
        registry=real_registry,
    )

    assert llm.call_count == 3

    tool_msgs = [m for m in base_messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2

    tool_names = {
        m["tool_call_id"]: json.loads(m["content"])
        for m in tool_msgs
    }

    # Verify weather data
    weather_msg = next(
        (json.loads(m["content"]) for m in tool_msgs if "temperature_c" in m["content"]),
        None,
    )
    assert weather_msg is not None
    assert weather_msg["temperature_c"] == 22

    # Verify stock data
    stock_msg = next(
        (json.loads(m["content"]) for m in tool_msgs if "price_usd" in m["content"]),
        None,
    )
    assert stock_msg is not None
    assert stock_msg["ticker"] == "AAPL"
    assert stock_msg["price_usd"] == 192.35

    # Final answer should mention both results
    assert "192" in result or "AAPL" in result or "Apple" in result
