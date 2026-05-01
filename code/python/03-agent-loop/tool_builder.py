"""Programmatic tool definition builder with built-in validation.

Provides ToolDef and Param classes for constructing OpenAI-compatible
function-calling tool definitions and validating arguments at runtime.

Replaces hand-authoring raw dicts and catches schema errors early,
before they reach the API.

Usage::

    weather_tool = ToolDef(
        name="get_weather",
        description=(
            "Get current weather for a city. "
            "Returns temperature (C/F), humidity, and conditions."
        ),
        parameters=[
            Param("city", "string", required=True,
                  description="City name with country code. "
                              "Format: 'City, CC'. Example: 'Shanghai, SH'"),
            Param("units", "string", required=False,
                  enum=["celsius", "fahrenheit"],
                  description="Temperature unit. Defaults to celsius."),
        ],
    )

    schema = weather_tool.to_openai_schema()          # OpenAI-ready dict
    weather_tool.validate_args({"city": "Shanghai, SH"})  # OK
    weather_tool.validate_args({"city": 123})             # raises ValueError

See docs/02-the-agent-loop/02-tool-design-patterns.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Param
# ---------------------------------------------------------------------------

_VALID_TYPES = frozenset(
    {"string", "integer", "number", "boolean", "array", "object"}
)


@dataclass
class Param:
    """A single parameter in a tool definition.

    Attributes:
        name:        Parameter name (must match the function argument name).
        type:        JSON Schema primitive type: string | integer | number |
                     boolean | array | object.
        required:    Whether the LLM must supply this parameter. Default True.
        description: Human-readable explanation *including* an example value.
                     The LLM reads this to decide what to pass.
        enum:        If provided, restricts the value to this list.
        minimum:     Minimum value for numeric parameters (inclusive).
        maximum:     Maximum value for numeric parameters (inclusive).
        default:     Default value if the parameter is omitted.
    """

    name: str
    type: str
    required: bool = True
    description: str = ""
    enum: Optional[list[Any]] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    default: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.type not in _VALID_TYPES:
            raise ValueError(
                f"Param '{self.name}': type must be one of "
                f"{sorted(_VALID_TYPES)}, got '{self.type}'"
            )

    def to_schema(self) -> dict[str, Any]:
        """Return the JSON Schema fragment for this parameter."""
        schema: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.enum is not None:
            schema["enum"] = self.enum
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.default is not None:
            schema["default"] = self.default
        return schema


# ---------------------------------------------------------------------------
# ToolDef
# ---------------------------------------------------------------------------


@dataclass
class ToolDef:
    """An OpenAI-compatible function-calling tool definition.

    Attributes:
        name:        Snake-case tool name shown to the LLM.
        description: Full tool description including what it returns and when
                     to use it.
        parameters:  List of :class:`Param` objects.
        strict:      When True, the model is constrained to the exact schema.
                     All properties are automatically added to ``required``
                     and ``additionalProperties`` is set to False.
    """

    name: str
    description: str
    parameters: list[Param] = field(default_factory=list)
    strict: bool = False

    # ------------------------------------------------------------------
    # Schema generation
    # ------------------------------------------------------------------

    def to_openai_schema(self) -> dict[str, Any]:
        """Generate the exact dict expected by the OpenAI ``tools`` parameter.

        Returns a dict of the form::

            {
                "type": "function",
                "function": {
                    "name": "...",
                    "description": "...",
                    "strict": True | False,
                    "parameters": {
                        "type": "object",
                        "properties": {...},
                        "required": [...],
                        "additionalProperties": False,
                    }
                }
            }
        """
        properties = {p.name: p.to_schema() for p in self.parameters}

        if self.strict:
            # Strict mode: every property must be required.
            required = [p.name for p in self.parameters]
        else:
            required = [p.name for p in self.parameters if p.required]

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": self.strict,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                    "additionalProperties": False,
                },
            },
        }

    # ------------------------------------------------------------------
    # Argument validation
    # ------------------------------------------------------------------

    def validate_args(self, args: dict[str, Any]) -> None:
        """Validate *args* against this tool's parameter definitions.

        Checks:
        - Required parameters are present.
        - Each value matches its declared type.
        - Values respect enum, minimum, and maximum constraints.

        Raises:
            ValueError: On the first violation, with a human-readable message
                        of the form "Parameter 'x' must be a string, got int (123)".
        """
        # --- Required presence ---
        for param in self.parameters:
            if param.required and param.name not in args:
                raise ValueError(
                    f"Missing required parameter: '{param.name}'"
                )

        # --- Per-parameter checks ---
        for param in self.parameters:
            if param.name not in args:
                continue
            value = args[param.name]
            self._check_type(param, value)
            self._check_enum(param, value)
            self._check_range(param, value)

    # ------------------------------------------------------------------
    # Deserialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolDef":
        """Create a :class:`ToolDef` from a plain dict (e.g. parsed from YAML/JSON).

        Expected format::

            {
                "name": "get_weather",
                "description": "...",
                "strict": false,          # optional
                "parameters": [
                    {
                        "name": "city",
                        "type": "string",
                        "required": true,
                        "description": "...",
                        "enum": null,
                        "minimum": null,
                        "maximum": null,
                        "default": null
                    }
                ]
            }
        """
        params: list[Param] = []
        for p in data.get("parameters", []):
            params.append(
                Param(
                    name=p["name"],
                    type=p["type"],
                    required=p.get("required", True),
                    description=p.get("description", ""),
                    enum=p.get("enum"),
                    minimum=p.get("minimum"),
                    maximum=p.get("maximum"),
                    default=p.get("default"),
                )
            )
        return cls(
            name=data["name"],
            description=data["description"],
            parameters=params,
            strict=data.get("strict", False),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_type(param: Param, value: Any) -> None:
        t = param.type
        actual = type(value).__name__

        if t == "string":
            if not isinstance(value, str):
                raise ValueError(
                    f"Parameter '{param.name}' must be a string, "
                    f"got {actual} ({value!r})"
                )
        elif t == "integer":
            # bool is a subclass of int in Python — reject it explicitly.
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"Parameter '{param.name}' must be an integer, "
                    f"got {actual} ({value!r})"
                )
        elif t == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Parameter '{param.name}' must be a number, "
                    f"got {actual} ({value!r})"
                )
        elif t == "boolean":
            if not isinstance(value, bool):
                raise ValueError(
                    f"Parameter '{param.name}' must be a boolean, "
                    f"got {actual} ({value!r})"
                )
        elif t == "array":
            if not isinstance(value, list):
                raise ValueError(
                    f"Parameter '{param.name}' must be an array, "
                    f"got {actual} ({value!r})"
                )
        elif t == "object":
            if not isinstance(value, dict):
                raise ValueError(
                    f"Parameter '{param.name}' must be an object, "
                    f"got {actual} ({value!r})"
                )

    @staticmethod
    def _check_enum(param: Param, value: Any) -> None:
        if param.enum is not None and value not in param.enum:
            raise ValueError(
                f"Parameter '{param.name}' must be one of {param.enum!r}, "
                f"got {value!r}"
            )

    @staticmethod
    def _check_range(param: Param, value: Any) -> None:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return
        if param.minimum is not None and value < param.minimum:
            raise ValueError(
                f"Parameter '{param.name}' must be >= {param.minimum}, "
                f"got {value}"
            )
        if param.maximum is not None and value > param.maximum:
            raise ValueError(
                f"Parameter '{param.name}' must be <= {param.maximum}, "
                f"got {value}"
            )
