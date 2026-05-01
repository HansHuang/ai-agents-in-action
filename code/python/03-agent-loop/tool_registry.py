"""Tool registration, dispatch, and structured error handling for the agent loop.

Provides a ToolRegistry class with a decorator-based registration API, OpenAI
schema generation, and execution that converts any tool failure into a
structured error message the LLM can read and explain to the user.

Usage::

    registry = ToolRegistry()

    @registry.register(
        name="get_weather",
        description="Get current weather for a city. Returns temperature and conditions.",
        parameters={
            "city": {"type": "string", "required": True,
                     "description": "City name with country code, e.g. 'Shanghai, SH'"},
        },
    )
    def get_weather(city: str) -> dict:
        if not valid_city(city):
            raise NotFoundError(f"City '{city}' not found", suggestion="Try 'Shanghai, SH'")
        return {"temperature": 22, "condition": "sunny"}

    schemas = registry.get_openai_schemas()          # → list of OpenAI tool dicts
    result  = registry.execute_tool(tool_call)       # → {"role": "tool", ...}

See docs/02-the-agent-loop/02-tool-design-patterns.md — "Error Handling"
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class ToolNotFoundError(Exception):
    """Raised internally when a tool name is not in the registry."""


class InvalidArgsError(Exception):
    """Raised when argument validation fails.

    Attributes:
        allowed_values: Valid options for the offending parameter, if applicable.
    """

    def __init__(self, message: str, allowed_values: Optional[list[Any]] = None) -> None:
        super().__init__(message)
        self.allowed_values: list[Any] = allowed_values or []


class ToolExecutionError(Exception):
    """Wraps an unexpected exception raised inside a tool function."""


class NotFoundError(Exception):
    """Tool functions may raise this when a resource cannot be found.

    The registry converts it to a structured ``not_found`` error message so
    the LLM can suggest alternatives to the user.

    Attributes:
        suggestion: A corrected value or alternative the model should try.
    """

    def __init__(self, message: str, suggestion: Optional[str] = None) -> None:
        super().__init__(message)
        self.suggestion = suggestion


class InvalidInputError(Exception):
    """Tool functions may raise this when a parameter value is logically invalid.

    Attributes:
        allowed_values: The valid values the model should choose from.
    """

    def __init__(self, message: str, allowed_values: Optional[list[Any]] = None) -> None:
        super().__init__(message)
        self.allowed_values: list[Any] = allowed_values or []


# ---------------------------------------------------------------------------
# Registry internals
# ---------------------------------------------------------------------------


@dataclass
class _Entry:
    name: str
    description: str
    fn: Callable[..., Any]
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Manages tool registration, schema generation, and dispatch.

    All execution errors — including missing tools, invalid arguments, and
    unexpected exceptions — are converted to structured tool messages so the
    agent loop never crashes due to a tool failure.
    """

    def __init__(self) -> None:
        self._tools: dict[str, _Entry] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        description: str,
        parameters: Optional[dict[str, dict[str, Any]]] = None,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """Decorator factory that registers a function as a named tool.

        Args:
            name:        Tool name shown to the LLM (must be unique).
            description: Full tool description including what it returns.
            parameters:  Dict mapping parameter names to spec dicts with keys:
                         ``type``, ``required``, ``description``, and
                         optionally ``enum``, ``minimum``, ``maximum``.

        Example::

            @registry.register(
                name="get_weather",
                description="Get weather. Returns temperature and conditions.",
                parameters={
                    "city": {"type": "string", "required": True,
                             "description": "City, e.g. 'Tokyo, JP'"},
                },
            )
            def get_weather(city: str) -> dict:
                ...
        """

        def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            self._tools[name] = _Entry(
                name=name,
                description=description,
                fn=fn,
                parameters=parameters or {},
            )
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_tools(self) -> list[dict[str, str]]:
        """Return a list of ``{name, description}`` dicts for all registered tools."""
        return [{"name": e.name, "description": e.description} for e in self._tools.values()]

    def get_openai_schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI function-calling tool definitions for all registered tools."""
        schemas: list[dict[str, Any]] = []
        for entry in self._tools.values():
            properties: dict[str, Any] = {}
            required: list[str] = []

            for param_name, spec in entry.parameters.items():
                prop: dict[str, Any] = {
                    "type": spec.get("type", "string"),
                    "description": spec.get("description", ""),
                }
                for key in ("enum", "minimum", "maximum", "default"):
                    if key in spec:
                        prop[key] = spec[key]

                properties[param_name] = prop
                if spec.get("required", False):
                    required.append(param_name)

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": entry.name,
                        "description": entry.description,
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False,
                        },
                    },
                }
            )
        return schemas

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute_tool(self, tool_call: Any) -> dict[str, str]:
        """Execute a tool call from the LLM and return a formatted tool message.

        Accepts either an OpenAI ``ChatCompletionMessageToolCall`` object
        (with ``.function.name``, ``.function.arguments``, ``.id``) or a plain
        dict with equivalent keys.

        Always returns a dict suitable for the messages array::

            {"role": "tool", "content": "<json string>", "tool_call_id": "..."}

        Tool execution errors are returned as structured JSON content — they are
        never raised as Python exceptions — so the agent loop can continue and
        the LLM can explain the failure to the user.
        """
        # --- Extract fields ---
        if hasattr(tool_call, "function"):
            name: str = tool_call.function.name
            arguments_str: str = tool_call.function.arguments
            tool_call_id: str = tool_call.id
        else:
            name = tool_call["function"]["name"]
            arguments_str = tool_call["function"]["arguments"]
            tool_call_id = tool_call["id"]

        def _msg(content: dict[str, Any]) -> dict[str, str]:
            return {
                "role": "tool",
                "content": json.dumps(content),
                "tool_call_id": tool_call_id,
            }

        # --- Tool lookup ---
        if name not in self._tools:
            available = ", ".join(self._tools.keys()) or "(none registered)"
            logger.warning("execute_tool: '%s' not found. Available: %s", name, available)
            return _msg(
                {
                    "success": False,
                    "error": "tool_not_found",
                    "message": (
                        f"Tool '{name}' is not available. "
                        f"Available tools: {available}"
                    ),
                }
            )

        entry = self._tools[name]

        # --- Parse arguments ---
        try:
            args: dict[str, Any] = json.loads(arguments_str) if arguments_str.strip() else {}
        except json.JSONDecodeError as exc:
            logger.error("execute_tool '%s': invalid argument JSON: %s", name, exc)
            return _msg(
                {
                    "success": False,
                    "error": "invalid_args",
                    "message": f"Invalid JSON in tool arguments: {exc}",
                }
            )

        # --- Validate arguments ---
        try:
            self._validate(entry, args)
        except InvalidArgsError as exc:
            logger.error("execute_tool '%s': validation failed: %s", name, exc)
            content: dict[str, Any] = {
                "success": False,
                "error": "invalid_args",
                "message": str(exc),
            }
            if exc.allowed_values:
                content["valid_values"] = exc.allowed_values
            return _msg(content)

        # --- Execute ---
        # Strip parameters the function doesn't declare so extra LLM keys
        # don't cause a TypeError. Unknown keys are logged as a warning.
        known = set(entry.parameters.keys())
        extra = set(args.keys()) - known
        if extra:
            logger.warning(
                "execute_tool '%s': ignoring undeclared parameters: %s", name, sorted(extra)
            )
        filtered: dict[str, Any] = {k: v for k, v in args.items() if k in known}

        start = time.perf_counter()
        try:
            result = entry.fn(**filtered)
            elapsed = time.perf_counter() - start
            logger.info("execute_tool '%s' OK (%.3fs)", name, elapsed)
            return _msg({"success": True, "data": result})

        except NotFoundError as exc:
            elapsed = time.perf_counter() - start
            logger.warning("execute_tool '%s' NotFoundError (%.3fs): %s", name, elapsed, exc)
            body: dict[str, Any] = {
                "success": False,
                "error": "not_found",
                "message": str(exc),
            }
            if exc.suggestion:
                body["suggestion"] = exc.suggestion
            return _msg(body)

        except InvalidInputError as exc:
            elapsed = time.perf_counter() - start
            logger.warning("execute_tool '%s' InvalidInputError (%.3fs): %s", name, elapsed, exc)
            body = {
                "success": False,
                "error": "invalid_input",
                "message": str(exc),
            }
            if exc.allowed_values:
                body["valid_values"] = exc.allowed_values
            return _msg(body)

        except Exception as exc:  # noqa: BLE001
            elapsed = time.perf_counter() - start
            logger.error(
                "execute_tool '%s' unexpected exception (%.3fs): %s",
                name, elapsed, exc,
                exc_info=True,
            )
            return _msg(
                {
                    "success": False,
                    "error": "internal_error",
                    "message": str(exc),
                }
            )

    # ------------------------------------------------------------------
    # Internal validation
    # ------------------------------------------------------------------

    _TYPE_VALIDATORS: dict[str, type | tuple[type, ...]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    def _validate(self, entry: _Entry, args: dict[str, Any]) -> None:
        """Raise :class:`InvalidArgsError` on the first validation failure."""
        # Required presence
        for param_name, spec in entry.parameters.items():
            if spec.get("required", False) and param_name not in args:
                raise InvalidArgsError(
                    f"Missing required parameter: '{param_name}'"
                )

        # Type, enum, and range checks
        for param_name, spec in entry.parameters.items():
            if param_name not in args:
                continue
            value = args[param_name]
            expected_type = spec.get("type", "string")
            validator = self._TYPE_VALIDATORS.get(expected_type)

            if validator:
                # Reject bool as integer/number
                if expected_type in ("integer", "number") and isinstance(value, bool):
                    raise InvalidArgsError(
                        f"Parameter '{param_name}' must be a {expected_type}, "
                        f"got bool ({value!r})"
                    )
                if not isinstance(value, validator):
                    actual = type(value).__name__
                    raise InvalidArgsError(
                        f"Parameter '{param_name}' must be a {expected_type}, "
                        f"got {actual} ({value!r})"
                    )

            if "enum" in spec and value not in spec["enum"]:
                raise InvalidArgsError(
                    f"Parameter '{param_name}' must be one of {spec['enum']!r}, "
                    f"got {value!r}",
                    allowed_values=spec["enum"],
                )
