"""Unit tests for the ReAct agent orchestration loop.

All OpenAI API calls are mocked — no OPENAI_API_KEY is required.

Test cases:
  1. Agent answers directly when no tools are needed
  2. Agent calls a tool and incorporates the result
  3. Agent aborts after MAX_ITERATIONS (no infinite loops)
  4. Agent handles a tool execution error gracefully
  5. Agent calls multiple tools in sequence
  6. Messages list has the correct structure after a full run
  7. Empty user input raises ValueError
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent import MAX_ITERATIONS, SYSTEM_PROMPT, run_agent
from tool_dispatcher import dispatch_tool
from tools import get_weather, get_stock_price, TOOLS


# ---------------------------------------------------------------------------
# Mock builders
# ---------------------------------------------------------------------------


def _content_response(content: str) -> MagicMock:
    """Build a mock LLM response containing only a final answer (no tool calls)."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _tool_call_response(
    tool_name: str,
    args: dict,
    call_id: str = "call_abc123",
) -> MagicMock:
    """Build a mock LLM response containing a single tool call."""
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
def mock_openai(monkeypatch):
    """Patch OpenAI inside agent.py and return the mock create() callable."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    mock_client = MagicMock()
    monkeypatch.setattr("agent.OpenAI", lambda **kw: mock_client)
    return mock_client.chat.completions.create


# ---------------------------------------------------------------------------
# Test 1: Direct answer (no tools needed)
# ---------------------------------------------------------------------------


def test_agent_answers_directly_when_no_tools_needed(mock_openai):
    """When the LLM returns content with no tool_calls, the answer is returned immediately."""
    mock_openai.return_value = _content_response("4")

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    result = run_agent("What is 2+2?", messages=messages, tools=TOOLS)

    assert result == "4"
    assert mock_openai.call_count == 1
    # No tool messages should be in history
    roles = [m["role"] for m in messages]
    assert "tool" not in roles


# ---------------------------------------------------------------------------
# Test 2: One tool call then final answer
# ---------------------------------------------------------------------------


def test_agent_calls_tool_and_returns_result(mock_openai):
    """Agent calls get_weather, appends the result, then returns the final answer."""
    mock_openai.side_effect = [
        _tool_call_response("get_weather", {"city": "Shanghai"}, call_id="call_w1"),
        _content_response("Shanghai is 22°C with light rain."),
    ]

    registry = {"get_weather": get_weather, "get_stock_price": get_stock_price}
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    result = run_agent(
        "What's the weather in Shanghai?",
        messages=messages,
        tools=TOOLS,
        registry=registry,
    )

    assert "22" in result or "Shanghai" in result or "rain" in result
    assert mock_openai.call_count == 2

    # Verify the tool result was appended with the right city
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    tool_data = json.loads(tool_msgs[0]["content"])
    assert tool_data["temperature_c"] == 22


# ---------------------------------------------------------------------------
# Test 3: Max iterations safety valve
# ---------------------------------------------------------------------------


def test_agent_stops_at_max_iterations(mock_openai):
    """Agent returns a graceful error after MAX_ITERATIONS without a final answer."""
    # Always respond with a tool call — the agent should never finish naturally.
    mock_openai.return_value = _tool_call_response(
        "get_weather", {"city": "Shanghai"}, call_id="call_loop"
    )

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    result = run_agent("Keep checking the weather.", messages=messages, tools=TOOLS)

    assert mock_openai.call_count == MAX_ITERATIONS
    assert "unable" in result.lower() or "steps" in result.lower()


# ---------------------------------------------------------------------------
# Test 4: Tool execution error — agent explains gracefully
# ---------------------------------------------------------------------------


def test_agent_handles_tool_error_gracefully(mock_openai):
    """When a tool raises, dispatch_tool returns an error message; the LLM explains it."""
    mock_openai.side_effect = [
        _tool_call_response("get_weather", {"city": "Shanghai"}, call_id="call_err"),
        _content_response("Sorry, the weather service is down. Please try again later."),
    ]

    def broken_weather(city: str) -> dict:
        raise RuntimeError("Weather API is unavailable")

    registry = {"get_weather": broken_weather}
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    result = run_agent(
        "What's the weather in Shanghai?",
        messages=messages,
        tools=TOOLS,
        registry=registry,
    )

    # The error tool message should contain the exception text
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    err_data = json.loads(tool_msgs[0]["content"])
    assert "error" in err_data

    # The final answer should acknowledge the failure
    assert result != ""


# ---------------------------------------------------------------------------
# Test 5: Multiple tools in sequence
# ---------------------------------------------------------------------------


def test_agent_calls_multiple_tools_in_sequence(mock_openai):
    """Agent calls get_weather for two cities before delivering the final answer."""
    mock_openai.side_effect = [
        _tool_call_response("get_weather", {"city": "Shanghai"}, call_id="call_t1"),
        _tool_call_response("get_weather", {"city": "London"}, call_id="call_t2"),
        _content_response("Shanghai is 22°C (light rain); London is 14°C (overcast)."),
    ]

    registry = {"get_weather": get_weather}
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    result = run_agent(
        "What's the weather in Shanghai and London?",
        messages=messages,
        tools=TOOLS,
        registry=registry,
    )

    assert mock_openai.call_count == 3

    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert len(tool_msgs) == 2

    cities = {json.loads(m["content"])["city"] for m in tool_msgs}
    assert "Shanghai" in cities
    assert "London" in cities

    assert "Shanghai" in result or "22" in result


# ---------------------------------------------------------------------------
# Test 6: Messages list structure after a full run
# ---------------------------------------------------------------------------


def test_messages_list_grows_correctly(mock_openai):
    """Verify the full message structure: system → user → assistant(tool) → tool → assistant."""
    mock_openai.side_effect = [
        _tool_call_response("get_weather", {"city": "Shanghai"}, call_id="call_s1"),
        _content_response("It's raining in Shanghai."),
    ]

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    run_agent("Weather in Shanghai?", messages=messages, tools=TOOLS)

    roles = [m["role"] for m in messages]
    assert roles[0] == "system"
    assert roles[1] == "user"
    assert roles[2] == "assistant"        # has tool_calls
    assert roles[3] == "tool"             # tool result
    assert roles[4] == "assistant"        # final answer

    # The first assistant message must contain tool_calls
    assert messages[2].get("tool_calls") is not None
    # The final assistant message must contain content
    assert messages[4]["content"] == "It's raining in Shanghai."


# ---------------------------------------------------------------------------
# Test 7: Empty input
# ---------------------------------------------------------------------------


def test_agent_handles_empty_user_input(monkeypatch):
    """Empty or whitespace-only input raises ValueError before any API call."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with pytest.raises(ValueError, match="empty"):
        run_agent("")
    with pytest.raises(ValueError, match="empty"):
        run_agent("   \n\t")
