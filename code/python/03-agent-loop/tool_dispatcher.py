"""Tool execution dispatcher.

dispatch_tool() is the bridge between the LLM's tool_call decision and your
Python functions. It:

  - Looks up the tool by name in a registry dict
  - Parses the JSON arguments string from the tool_call
  - Calls the Python function with keyword arguments
  - Returns a properly formatted tool message for the messages array
  - Returns a descriptive error message (not a Python exception) if anything
    fails — the LLM receives the error and can explain it to the user

This module is intentionally decoupled from agent.py so you can swap the
registry without changing the orchestration loop.

See docs/02-the-agent-loop/01-anatomy-of-an-agent.md — "The Hands (Tools)"
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

ToolRegistry = dict[str, Callable[..., Any]]


def dispatch_tool(
    tool_call: Any,
    registry: ToolRegistry | None = None,
) -> dict:
    """Execute a tool call and return a formatted tool message.

    Args:
        tool_call: An OpenAI ToolCall object with .id, .function.name,
                   and .function.arguments (JSON string).
        registry:  Mapping of tool names to callables. Defaults to the
                   built-in tools (get_weather, get_stock_price).

    Returns:
        A tool message dict ready to append to the messages array::

            {"role": "tool", "content": "<json string>", "tool_call_id": "<id>"}

        On error, the content is ``{"error": "<description>"}`` — the LLM
        receives this and can explain the failure to the user.
    """
    if registry is None:
        from tools import get_weather, get_stock_price
        registry = {
            "get_weather": get_weather,
            "get_stock_price": get_stock_price,
        }

    name: str = tool_call.function.name
    tool_call_id: str = tool_call.id

    # Parse arguments — the SDK delivers them as a JSON string.
    raw_args = tool_call.function.arguments
    try:
        args: dict = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
    except json.JSONDecodeError as exc:
        logger.error("Tool %r: invalid argument JSON: %s", name, exc)
        return _error_message(tool_call_id, f"Invalid arguments JSON: {exc}")

    # Look up the tool.
    fn = registry.get(name)
    if fn is None:
        available = ", ".join(sorted(registry))
        logger.warning("Tool %r not found. Available: %s", name, available)
        return _error_message(
            tool_call_id,
            f"Tool '{name}' is not available. Available tools: {available}",
        )

    # Execute with timing.
    start = time.perf_counter()
    try:
        result = fn(**args)
    except TypeError as exc:
        elapsed = (time.perf_counter() - start) * 1000
        logger.error("Tool %r: argument mismatch (%.1f ms): %s", name, elapsed, exc)
        return _error_message(tool_call_id, f"Wrong arguments for '{name}': {exc}")
    except Exception as exc:  # noqa: BLE001
        elapsed = (time.perf_counter() - start) * 1000
        logger.error("Tool %r: execution failed (%.1f ms): %s", name, elapsed, exc)
        return _error_message(tool_call_id, f"Tool '{name}' failed: {exc}")

    elapsed_ms = (time.perf_counter() - start) * 1000
    result_json = json.dumps(result)
    preview = result_json[:200] + "…" if len(result_json) > 200 else result_json
    logger.debug(
        "Tool %r(%s) → %s  [%.1f ms]",
        name,
        json.dumps(args),
        preview,
        elapsed_ms,
    )

    return {
        "role": "tool",
        "content": result_json,
        "tool_call_id": tool_call_id,
    }


def _error_message(tool_call_id: str, error: str) -> dict:
    """Return a tool message that tells the LLM what went wrong."""
    return {
        "role": "tool",
        "content": json.dumps({"error": error}),
        "tool_call_id": tool_call_id,
    }
