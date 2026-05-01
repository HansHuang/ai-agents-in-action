"""Basic tests for ToolRegistry — registration, schemas, and execution."""

from __future__ import annotations

import json

import pytest

from tool_registry import (
    InvalidInputError,
    NotFoundError,
    ToolRegistry,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_tool_call(name: str, arguments: dict, tool_call_id: str = "call_1") -> dict:
    """Build a plain-dict tool call (same shape as OpenAI's object)."""
    return {
        "function": {"name": name, "arguments": json.dumps(arguments)},
        "id": tool_call_id,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registry() -> ToolRegistry:
    r = ToolRegistry()

    @r.register(
        name="get_weather",
        description="Get current weather for a city. Returns temperature and conditions.",
        parameters={
            "city": {
                "type": "string",
                "required": True,
                "description": "City name with country code, e.g. 'Tokyo, JP'",
            },
        },
    )
    def get_weather(city: str) -> dict:
        return {"temperature": 22, "condition": "sunny", "city": city}

    return r


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_and_list_tools(registry: ToolRegistry) -> None:
    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0]["name"] == "get_weather"
    assert "description" in tools[0]


def test_multiple_registrations() -> None:
    r = ToolRegistry()

    @r.register(name="tool_a", description="Tool A. Returns data.", parameters={})
    def tool_a() -> str:
        return "a"

    @r.register(name="tool_b", description="Tool B. Returns data.", parameters={})
    def tool_b() -> str:
        return "b"

    assert len(r.list_tools()) == 2


# ---------------------------------------------------------------------------
# get_openai_schemas
# ---------------------------------------------------------------------------


def test_get_openai_schemas_structure(registry: ToolRegistry) -> None:
    schemas = registry.get_openai_schemas()
    assert len(schemas) == 1
    s = schemas[0]
    assert s["type"] == "function"
    fn = s["function"]
    assert fn["name"] == "get_weather"
    assert "properties" in fn["parameters"]
    assert "city" in fn["parameters"]["properties"]
    assert "city" in fn["parameters"]["required"]


def test_get_openai_schemas_additional_properties_false(registry: ToolRegistry) -> None:
    schemas = registry.get_openai_schemas()
    assert schemas[0]["function"]["parameters"]["additionalProperties"] is False


# ---------------------------------------------------------------------------
# execute_tool — success
# ---------------------------------------------------------------------------


def test_execute_tool_success(registry: ToolRegistry) -> None:
    result = registry.execute_tool(make_tool_call("get_weather", {"city": "Tokyo"}))
    assert result["role"] == "tool"
    assert result["tool_call_id"] == "call_1"
    content = json.loads(result["content"])
    assert content["success"] is True
    assert content["data"]["temperature"] == 22


def test_execute_tool_preserves_tool_call_id(registry: ToolRegistry) -> None:
    result = registry.execute_tool(make_tool_call("get_weather", {"city": "Tokyo"}, "call_abc"))
    assert result["tool_call_id"] == "call_abc"


# ---------------------------------------------------------------------------
# execute_tool — tool not found
# ---------------------------------------------------------------------------


def test_execute_tool_not_found(registry: ToolRegistry) -> None:
    result = registry.execute_tool(make_tool_call("nonexistent", {}))
    content = json.loads(result["content"])
    assert content["success"] is False
    assert content["error"] == "tool_not_found"
    assert "nonexistent" in content["message"]
    assert "tool_call_id" in result


# ---------------------------------------------------------------------------
# execute_tool — validation failures
# ---------------------------------------------------------------------------


def test_execute_tool_missing_required_param(registry: ToolRegistry) -> None:
    result = registry.execute_tool(make_tool_call("get_weather", {}))
    content = json.loads(result["content"])
    assert content["success"] is False
    assert content["error"] == "invalid_args"
    assert "city" in content["message"]


def test_execute_tool_wrong_param_type(registry: ToolRegistry) -> None:
    result = registry.execute_tool(make_tool_call("get_weather", {"city": 123}))
    content = json.loads(result["content"])
    assert content["success"] is False
    assert content["error"] == "invalid_args"
    assert "string" in content["message"]


# ---------------------------------------------------------------------------
# execute_tool — tool-raised exceptions
# ---------------------------------------------------------------------------


def test_execute_tool_not_found_error() -> None:
    r = ToolRegistry()

    @r.register(
        name="lookup",
        description="Look up an item by ID. Returns item details.",
        parameters={"id": {"type": "string", "required": True, "description": "Item ID, e.g. 'item_001'"}},
    )
    def lookup(id: str) -> dict:
        raise NotFoundError(f"Item '{id}' not found", suggestion="Try 'item_001'")

    result = r.execute_tool(make_tool_call("lookup", {"id": "bad_id"}))
    content = json.loads(result["content"])
    assert content["success"] is False
    assert content["error"] == "not_found"
    assert content["suggestion"] == "Try 'item_001'"


def test_execute_tool_invalid_input_error() -> None:
    r = ToolRegistry()

    @r.register(
        name="set_status",
        description="Set item status. Returns updated item.",
        parameters={"status": {"type": "string", "required": True, "description": "Status, e.g. 'active'"}},
    )
    def set_status(status: str) -> dict:
        raise InvalidInputError("Invalid status", allowed_values=["active", "inactive"])

    result = r.execute_tool(make_tool_call("set_status", {"status": "unknown"}))
    content = json.loads(result["content"])
    assert content["success"] is False
    assert content["error"] == "invalid_input"
    assert "active" in content["valid_values"]
