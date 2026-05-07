"""Tool definitions and handler logic for the MCP weather server.

Separating schema definitions from server wiring keeps server.py thin
and makes it easy to unit-test handlers in isolation.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from typing import Any

from mcp.types import TextContent, Tool

from weather_data import CityNotFoundError, get_current_weather, get_forecast, list_supported_cities

# ---------------------------------------------------------------------------
# Tool schemas  (the source of truth — clients discover these at runtime)
# ---------------------------------------------------------------------------

GET_WEATHER_TOOL = Tool(
    name="get_weather",
    description=(
        "Get current weather conditions for a city. "
        "Returns temperature, humidity, wind speed, and sky conditions. "
        "Provide the city name with an optional country code for accuracy. "
        "Examples: 'Tokyo, JP', 'London, UK', 'New York, US'."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": (
                    "City name with optional ISO country code. "
                    "Examples: 'Tokyo, JP', 'London, UK', 'Sydney, AU'."
                ),
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit. Defaults to 'celsius'.",
                "default": "celsius",
            },
        },
        "required": ["city"],
        "additionalProperties": False,
    },
)

GET_FORECAST_TOOL = Tool(
    name="get_forecast",
    description=(
        "Get a multi-day weather forecast for a city. "
        "Returns daily high/low temperatures, sky conditions, and precipitation "
        "probability for 1–10 days. "
        "Examples: 'Paris, FR' for 3 days, 'Berlin, DE' for 7 days."
    ),
    inputSchema={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": (
                    "City name with optional ISO country code. "
                    "Examples: 'Berlin, DE', 'Toronto, CA', 'Dubai, AE'."
                ),
            },
            "days": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "description": "Number of forecast days (1–10). Defaults to 5.",
                "default": 5,
            },
        },
        "required": ["city"],
        "additionalProperties": False,
    },
)

ALL_TOOLS: list[Tool] = [GET_WEATHER_TOOL, GET_FORECAST_TOOL]


# ---------------------------------------------------------------------------
# Rate limiter  (token-bucket stub — replace with Redis in production)
# ---------------------------------------------------------------------------

class RateLimiter:
    """Per-client token-bucket rate limiter.

    In production use a distributed implementation backed by Redis
    (e.g. ``redis-py`` + ``lua`` scripts, or ``limits`` library).
    """

    def __init__(self, max_calls: int = 60, window_seconds: int = 60) -> None:
        self.max_calls = max_calls
        self.window_seconds = window_seconds
        self._calls: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, client_id: str = "default") -> bool:
        """Return ``True`` if this call is within rate limits."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        # Evict stale timestamps
        self._calls[client_id] = [t for t in self._calls[client_id] if t > cutoff]
        if len(self._calls[client_id]) >= self.max_calls:
            return False
        self._calls[client_id].append(now)
        return True


_rate_limiter = RateLimiter(max_calls=60, window_seconds=60)


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

def handle_get_weather(arguments: dict[str, Any]) -> list[TextContent]:
    """Execute the ``get_weather`` tool and return MCP content."""
    city: str | None = arguments.get("city")
    if not city:
        return [TextContent(type="text", text=json.dumps(
            {"error": "Missing required argument: 'city'."}
        ))]

    units: str = arguments.get("units", "celsius")
    if units not in ("celsius", "fahrenheit"):
        units = "celsius"

    print(f"[weather-server] get_weather: city={city!r} units={units}", file=sys.stderr)

    if not _rate_limiter.is_allowed():
        return [TextContent(type="text", text=json.dumps(
            {"error": "Rate limit exceeded. Please wait before retrying."}
        ))]

    try:
        data = get_current_weather(city, units)
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except CityNotFoundError as exc:
        return [TextContent(type="text", text=json.dumps({
            "error": str(exc),
            "supported_cities": list_supported_cities(),
        }))]
    except Exception as exc:  # noqa: BLE001
        print(f"[weather-server] Unexpected error in get_weather: {exc}", file=sys.stderr)
        return [TextContent(type="text", text=json.dumps(
            {"error": "Internal server error. Please try again."}
        ))]


def handle_get_forecast(arguments: dict[str, Any]) -> list[TextContent]:
    """Execute the ``get_forecast`` tool and return MCP content."""
    city: str | None = arguments.get("city")
    if not city:
        return [TextContent(type="text", text=json.dumps(
            {"error": "Missing required argument: 'city'."}
        ))]

    try:
        days = int(arguments.get("days", 5))
    except (TypeError, ValueError):
        days = 5

    print(f"[weather-server] get_forecast: city={city!r} days={days}", file=sys.stderr)

    if not _rate_limiter.is_allowed():
        return [TextContent(type="text", text=json.dumps(
            {"error": "Rate limit exceeded. Please wait before retrying."}
        ))]

    try:
        data = get_forecast(city, days)
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except CityNotFoundError as exc:
        return [TextContent(type="text", text=json.dumps({
            "error": str(exc),
            "supported_cities": list_supported_cities(),
        }))]
    except Exception as exc:  # noqa: BLE001
        print(f"[weather-server] Unexpected error in get_forecast: {exc}", file=sys.stderr)
        return [TextContent(type="text", text=json.dumps(
            {"error": "Internal server error. Please try again."}
        ))]
