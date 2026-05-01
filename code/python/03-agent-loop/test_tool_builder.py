"""Tests for tool_builder.py — ToolDef and Param classes."""

from __future__ import annotations

import json

import pytest

from tool_builder import Param, ToolDef


# ---------------------------------------------------------------------------
# Param construction
# ---------------------------------------------------------------------------


def test_param_valid_type_accepted():
    p = Param("city", "string", description="A city")
    assert p.type == "string"


def test_param_invalid_type_raises():
    with pytest.raises(ValueError, match="type must be one of"):
        Param("x", "unsupported_type")


def test_param_to_schema_minimal():
    p = Param("city", "string", description="City name, e.g. 'Tokyo'")
    schema = p.to_schema()
    assert schema["type"] == "string"
    assert schema["description"] == "City name, e.g. 'Tokyo'"
    assert "enum" not in schema


def test_param_to_schema_with_enum():
    p = Param("units", "string", enum=["celsius", "fahrenheit"],
              description="Temperature unit, e.g. 'celsius'")
    schema = p.to_schema()
    assert schema["enum"] == ["celsius", "fahrenheit"]


def test_param_to_schema_with_min_max():
    p = Param("limit", "integer", minimum=1, maximum=50,
              description="Result count, e.g. 10")
    schema = p.to_schema()
    assert schema["minimum"] == 1
    assert schema["maximum"] == 50


def test_param_to_schema_with_default():
    p = Param("limit", "integer", required=False, default=10,
              description="Result count, e.g. 10")
    schema = p.to_schema()
    assert schema["default"] == 10


# ---------------------------------------------------------------------------
# to_openai_schema structure
# ---------------------------------------------------------------------------


def test_to_openai_schema_top_level_structure():
    tool = ToolDef(
        name="get_weather",
        description="Get weather. Returns temperature and conditions.",
        parameters=[
            Param("city", "string", required=True,
                  description="City name, e.g. 'Tokyo, JP'"),
        ],
    )
    schema = tool.to_openai_schema()
    assert schema["type"] == "function"
    fn = schema["function"]
    assert fn["name"] == "get_weather"
    assert "city" in fn["parameters"]["properties"]
    assert "city" in fn["parameters"]["required"]


def test_to_openai_schema_additional_properties_false():
    tool = ToolDef("t", "desc", [])
    schema = tool.to_openai_schema()
    assert schema["function"]["parameters"]["additionalProperties"] is False


def test_to_openai_schema_optional_param_not_in_required():
    tool = ToolDef(
        "t", "desc",
        [Param("units", "string", required=False, description="Units, e.g. 'celsius'")],
    )
    schema = tool.to_openai_schema()
    assert "units" not in schema["function"]["parameters"]["required"]


def test_to_openai_schema_strict_forces_all_required():
    tool = ToolDef(
        "t", "desc",
        parameters=[
            Param("city", "string", required=True, description="City, e.g. 'Tokyo'"),
            Param("units", "string", required=False, description="Units, e.g. 'celsius'"),
        ],
        strict=True,
    )
    schema = tool.to_openai_schema()
    assert schema["function"]["strict"] is True
    # strict=True must include every property in required
    assert "units" in schema["function"]["parameters"]["required"]


def test_to_openai_schema_serialisable():
    tool = ToolDef(
        "get_weather",
        "Get weather. Returns temperature in Celsius.",
        [Param("city", "string", description="City, e.g. 'Tokyo'")],
    )
    # Should not raise
    json_str = json.dumps(tool.to_openai_schema())
    assert "get_weather" in json_str


# ---------------------------------------------------------------------------
# validate_args — happy path
# ---------------------------------------------------------------------------


def test_validate_args_valid_string():
    tool = ToolDef("t", "d", [Param("city", "string", description="City, e.g. 'Tokyo'")])
    tool.validate_args({"city": "Tokyo"})  # must not raise


def test_validate_args_optional_param_absent_is_ok():
    tool = ToolDef("t", "d", [Param("units", "string", required=False, description="d")])
    tool.validate_args({})  # must not raise


def test_validate_args_extra_keys_are_ignored():
    tool = ToolDef("t", "d", [Param("city", "string", description="d")])
    tool.validate_args({"city": "Tokyo", "unexpected": "value"})  # must not raise


# ---------------------------------------------------------------------------
# validate_args — type violations
# ---------------------------------------------------------------------------


def test_validate_args_missing_required():
    tool = ToolDef("t", "d", [Param("city", "string", required=True, description="d")])
    with pytest.raises(ValueError, match="Missing required parameter: 'city'"):
        tool.validate_args({})


def test_validate_args_wrong_type_string():
    tool = ToolDef("t", "d", [Param("city", "string", description="d")])
    with pytest.raises(ValueError, match="must be a string.*got int"):
        tool.validate_args({"city": 123})


def test_validate_args_wrong_type_integer():
    tool = ToolDef("t", "d", [Param("count", "integer", description="Count, e.g. 5")])
    with pytest.raises(ValueError, match="must be an integer"):
        tool.validate_args({"count": 3.14})


def test_validate_args_bool_rejected_as_integer():
    """bool is a subclass of int in Python; it must not pass an integer check."""
    tool = ToolDef("t", "d", [Param("count", "integer", description="d")])
    with pytest.raises(ValueError, match="must be an integer.*got bool"):
        tool.validate_args({"count": True})


def test_validate_args_wrong_type_boolean():
    tool = ToolDef("t", "d", [Param("flag", "boolean", description="d")])
    with pytest.raises(ValueError, match="must be a boolean"):
        tool.validate_args({"flag": 1})


# ---------------------------------------------------------------------------
# validate_args — constraint violations
# ---------------------------------------------------------------------------


def test_validate_args_invalid_enum():
    tool = ToolDef(
        "t", "d",
        [Param("units", "string", enum=["celsius", "fahrenheit"], description="d")],
    )
    with pytest.raises(ValueError, match="must be one of"):
        tool.validate_args({"units": "kelvin"})


def test_validate_args_below_minimum():
    tool = ToolDef("t", "d", [Param("limit", "integer", minimum=1, description="d")])
    with pytest.raises(ValueError, match=">= 1"):
        tool.validate_args({"limit": 0})


def test_validate_args_above_maximum():
    tool = ToolDef("t", "d", [Param("limit", "integer", maximum=50, description="d")])
    with pytest.raises(ValueError, match="<= 50"):
        tool.validate_args({"limit": 100})


def test_validate_args_within_range_is_ok():
    tool = ToolDef("t", "d", [Param("limit", "integer", minimum=1, maximum=50, description="d")])
    tool.validate_args({"limit": 25})  # must not raise


# ---------------------------------------------------------------------------
# from_dict
# ---------------------------------------------------------------------------


def test_from_dict_basic():
    data = {
        "name": "get_weather",
        "description": "Get weather. Returns temperature and conditions.",
        "parameters": [
            {"name": "city", "type": "string", "required": True,
             "description": "City name, e.g. 'Tokyo'"},
        ],
    }
    tool = ToolDef.from_dict(data)
    assert tool.name == "get_weather"
    assert len(tool.parameters) == 1
    assert tool.parameters[0].name == "city"
    assert tool.parameters[0].required is True


def test_from_dict_optional_fields_default():
    data = {
        "name": "t",
        "description": "d",
        "parameters": [{"name": "x", "type": "string"}],
    }
    tool = ToolDef.from_dict(data)
    assert tool.strict is False
    assert tool.parameters[0].required is True
    assert tool.parameters[0].enum is None


def test_from_dict_with_enum_and_range():
    data = {
        "name": "t",
        "description": "d",
        "parameters": [
            {
                "name": "limit",
                "type": "integer",
                "required": False,
                "description": "Limit, e.g. 10",
                "minimum": 1,
                "maximum": 100,
            }
        ],
    }
    tool = ToolDef.from_dict(data)
    p = tool.parameters[0]
    assert p.minimum == 1
    assert p.maximum == 100
    assert p.required is False


def test_from_dict_roundtrip():
    """Parsing a dict then generating a schema should produce a valid structure."""
    data = {
        "name": "search_orders",
        "description": "Search orders by email. Returns a list of orders.",
        "parameters": [
            {"name": "email", "type": "string", "required": True,
             "description": "Email address, e.g. 'user@example.com'"},
            {"name": "status", "type": "string", "required": False,
             "enum": ["pending", "shipped", "delivered"],
             "description": "Filter status, e.g. 'shipped'"},
        ],
    }
    tool = ToolDef.from_dict(data)
    schema = tool.to_openai_schema()
    assert schema["function"]["parameters"]["properties"]["status"]["enum"] == [
        "pending", "shipped", "delivered"
    ]
