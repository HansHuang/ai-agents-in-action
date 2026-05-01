"""Edge-case tests for ToolRegistry.execute_tool().

Covers unusual but realistic situations: missing tools, type mismatches,
extra parameters, exceptions inside tools, non-dict return values, and
concurrent execution.
"""

from __future__ import annotations

import json
import threading

import pytest

from tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool_call(name: str, arguments: dict, tool_call_id: str = "call_1") -> dict:
    return {
        "function": {"name": name, "arguments": json.dumps(arguments)},
        "id": tool_call_id,
    }


def content(result: dict) -> dict:
    return json.loads(result["content"])


# ---------------------------------------------------------------------------
# Test 1: Execute unregistered tool
# ---------------------------------------------------------------------------


def test_execute_unregistered_tool() -> None:
    """Executing a tool that was never registered returns a structured error."""
    registry = ToolRegistry()

    # Register one tool so the error message can list available tools.
    @registry.register(name="existing_tool", description="Exists. Returns nothing.", parameters={})
    def existing_tool() -> None:
        return None

    result = registry.execute_tool(make_tool_call("phantom_tool", {}))
    c = content(result)

    assert c["success"] is False
    assert c["error"] == "tool_not_found"
    # The message should tell the caller what to look for.
    assert "phantom_tool" in c["message"]
    assert "existing_tool" in c["message"], "Error message should list available tools"
    assert result["tool_call_id"] == "call_1"


# ---------------------------------------------------------------------------
# Test 2: Missing required parameter
# ---------------------------------------------------------------------------


def test_execute_with_missing_required_param() -> None:
    """Omitting a required parameter returns an invalid_args error."""
    registry = ToolRegistry()

    @registry.register(
        name="get_weather",
        description="Get weather. Returns conditions.",
        parameters={"city": {"type": "string", "required": True, "description": "City, e.g. 'Tokyo'"}},
    )
    def get_weather(city: str) -> dict:
        return {"temperature": 22}

    result = registry.execute_tool(make_tool_call("get_weather", {}))
    c = content(result)

    assert c["success"] is False
    assert c["error"] == "invalid_args"
    assert "city" in c["message"]


# ---------------------------------------------------------------------------
# Test 3: Wrong parameter type
# ---------------------------------------------------------------------------


def test_execute_with_wrong_param_type() -> None:
    """Passing an integer where a string is expected returns a type-mismatch error."""
    registry = ToolRegistry()

    @registry.register(
        name="get_weather",
        description="Get weather. Returns conditions.",
        parameters={"city": {"type": "string", "required": True, "description": "City, e.g. 'Tokyo'"}},
    )
    def get_weather(city: str) -> dict:
        return {"temperature": 22}

    result = registry.execute_tool(make_tool_call("get_weather", {"city": 123}))
    c = content(result)

    assert c["success"] is False
    assert c["error"] == "invalid_args"
    # Message must describe both expected and actual types.
    assert "string" in c["message"]
    assert "int" in c["message"]


# ---------------------------------------------------------------------------
# Test 4: Extra parameters are silently ignored
# ---------------------------------------------------------------------------


def test_execute_with_extra_params() -> None:
    """Extra parameters not declared in the tool spec are stripped and ignored.

    Behaviour: execute_tool logs a warning and strips the unknown keys before
    calling the tool function, so the call succeeds.  The caller does NOT
    receive an error for sending extra keys; this mirrors how most REST APIs
    handle unexpected query parameters.
    """
    registry = ToolRegistry()

    @registry.register(
        name="get_weather",
        description="Get weather. Returns conditions.",
        parameters={"city": {"type": "string", "required": True, "description": "City, e.g. 'Tokyo'"}},
    )
    def get_weather(city: str) -> dict:
        return {"temperature": 22, "city": city}

    # "format" is an extra key not declared in the spec.
    result = registry.execute_tool(
        make_tool_call("get_weather", {"city": "Shanghai", "format": "json"})
    )
    c = content(result)

    assert c["success"] is True, (
        "Extra parameters should be silently ignored, not treated as an error"
    )
    assert c["data"]["temperature"] == 22


# ---------------------------------------------------------------------------
# Test 5: Tool raises RuntimeError
# ---------------------------------------------------------------------------


def test_tool_raises_exception() -> None:
    """An unexpected exception inside a tool is converted to an internal_error message."""
    registry = ToolRegistry()

    @registry.register(
        name="flaky_tool",
        description="Flaky. Returns data when it works.",
        parameters={},
    )
    def flaky_tool() -> dict:
        raise RuntimeError("External API timeout")

    result = registry.execute_tool(make_tool_call("flaky_tool", {}, "call_timeout"))
    c = content(result)

    assert c["success"] is False
    assert c["error"] == "internal_error"
    # The original error message must be surfaced so the LLM can explain it.
    assert "External API timeout" in c["message"]
    # The tool_call_id must be preserved for the message array to be valid.
    assert result["tool_call_id"] == "call_timeout"


# ---------------------------------------------------------------------------
# Test 6: Tool returns a non-dict (plain string)
# ---------------------------------------------------------------------------


def test_tool_returns_non_dict() -> None:
    """Tools that return a plain string are wrapped in {success, data}."""
    registry = ToolRegistry()

    @registry.register(name="ping", description="Ping the service. Returns 'OK'.", parameters={})
    def ping() -> str:
        return "OK"

    result = registry.execute_tool(make_tool_call("ping", {}))
    c = content(result)

    assert c["success"] is True
    assert c["data"] == "OK"


# ---------------------------------------------------------------------------
# Test 7: Tool returns None
# ---------------------------------------------------------------------------


def test_tool_returns_none() -> None:
    """Tools that return None (void) are wrapped in {success: true, data: null}."""
    registry = ToolRegistry()

    @registry.register(name="void_tool", description="Side-effect only. Returns nothing.", parameters={})
    def void_tool() -> None:
        return None  # Explicit for clarity

    result = registry.execute_tool(make_tool_call("void_tool", {}))
    c = content(result)

    assert c["success"] is True
    assert c["data"] is None


# ---------------------------------------------------------------------------
# Test 8: Concurrent tool execution — no race conditions
# ---------------------------------------------------------------------------


def test_concurrent_tool_execution() -> None:
    """Three tools executed in parallel all return correct results with no data corruption."""
    registry = ToolRegistry()

    @registry.register(name="tool_a", description="Tool A. Returns {'tool': 'a'}.", parameters={})
    def tool_a() -> dict:
        return {"tool": "a"}

    @registry.register(name="tool_b", description="Tool B. Returns {'tool': 'b'}.", parameters={})
    def tool_b() -> dict:
        return {"tool": "b"}

    @registry.register(name="tool_c", description="Tool C. Returns {'tool': 'c'}.", parameters={})
    def tool_c() -> dict:
        return {"tool": "c"}

    results: dict[str, dict] = {}
    lock = threading.Lock()

    def run(tool_name: str, key: str) -> None:
        r = registry.execute_tool(make_tool_call(tool_name, {}, f"call_{key}"))
        with lock:
            results[key] = r

    threads = [
        threading.Thread(target=run, args=("tool_a", "a")),
        threading.Thread(target=run, args=("tool_b", "b")),
        threading.Thread(target=run, args=("tool_c", "c")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 3
    for key in ("a", "b", "c"):
        assert key in results
        c = content(results[key])
        assert c["success"] is True
        assert c["data"]["tool"] == key
        assert results[key]["tool_call_id"] == f"call_{key}"


# ---------------------------------------------------------------------------
# Test 9: list_tools
# ---------------------------------------------------------------------------


def test_tool_registry_list_tools() -> None:
    """list_tools() returns one entry per registered tool with name and description."""
    registry = ToolRegistry()

    @registry.register(name="alpha", description="Alpha tool. Returns alpha.", parameters={})
    def alpha() -> str:
        return "a"

    @registry.register(name="beta", description="Beta tool. Returns beta.", parameters={})
    def beta() -> str:
        return "b"

    @registry.register(name="gamma", description="Gamma tool. Returns gamma.", parameters={})
    def gamma() -> str:
        return "c"

    tools = registry.list_tools()
    assert len(tools) == 3
    names = {t["name"] for t in tools}
    assert names == {"alpha", "beta", "gamma"}
    for t in tools:
        assert "description" in t


# ---------------------------------------------------------------------------
# Test 10: get_openai_schemas
# ---------------------------------------------------------------------------


def test_tool_registry_get_schemas() -> None:
    """get_openai_schemas() returns valid OpenAI function definitions for all tools."""
    registry = ToolRegistry()

    @registry.register(
        name="get_weather",
        description="Get current weather for a city. Returns temperature and conditions.",
        parameters={
            "city": {"type": "string", "required": True, "description": "City, e.g. 'Tokyo'"},
        },
    )
    def get_weather(city: str) -> dict:
        return {}

    @registry.register(
        name="get_stock_price",
        description="Get current stock price. Returns price and change percent.",
        parameters={
            "ticker": {"type": "string", "required": True, "description": "Ticker, e.g. 'AAPL'"},
        },
    )
    def get_stock_price(ticker: str) -> dict:
        return {}

    schemas = registry.get_openai_schemas()
    assert len(schemas) == 2

    for schema in schemas:
        assert schema["type"] == "function"
        fn = schema["function"]
        assert "name" in fn
        assert "description" in fn
        params = fn["parameters"]
        assert params["type"] == "object"
        assert "properties" in params
        assert "required" in params

    names = {s["function"]["name"] for s in schemas}
    assert names == {"get_weather", "get_stock_price"}
