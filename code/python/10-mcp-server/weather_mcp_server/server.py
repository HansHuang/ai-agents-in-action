"""MCP Weather Server — main entry point.

Exposes two tools and one resource over the stdio transport:

  Tools:
    get_weather(city, units)    — current conditions
    get_forecast(city, days)    — multi-day forecast

  Resources:
    weather://status            — server health and uptime

Usage:
    python server.py            # MCP client connects via stdio

Logs go to stderr; MCP protocol traffic uses stdin/stdout.
Replace weather_data.py with a real API client for production.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import datetime

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import AnyUrl, Resource, TextContent, Tool

from tools import ALL_TOOLS, handle_get_forecast, handle_get_weather
from weather_data import list_supported_cities

# ---------------------------------------------------------------------------
# Server instance
# ---------------------------------------------------------------------------

SERVER_START_TIME = time.time()
server = Server("weather-server")


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Advertise available tools to any connecting MCP client."""
    return ALL_TOOLS


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch incoming tool calls to the appropriate handler.

    Returns a JSON error payload for unknown tools rather than crashing,
    so misbehaving clients don't take down the server.
    """
    print(f"[weather-server] Tool call received: {name!r}", file=sys.stderr)

    if name == "get_weather":
        return handle_get_weather(arguments)
    if name == "get_forecast":
        return handle_get_forecast(arguments)

    return [TextContent(type="text", text=json.dumps({
        "error": f"Unknown tool: {name!r}",
        "available_tools": ["get_weather", "get_forecast"],
    }))]


# ---------------------------------------------------------------------------
# Resource handlers
# ---------------------------------------------------------------------------

@server.list_resources()
async def list_resources() -> list[Resource]:
    """Advertise readable resources."""
    return [
        Resource(
            uri=AnyUrl("weather://status"),
            name="Server Status",
            description=(
                "Health, uptime, and capability information for this "
                "weather MCP server."
            ),
            mimeType="application/json",
        )
    ]


@server.read_resource()
async def read_resource(uri: AnyUrl) -> str:
    """Return resource contents by URI."""
    if str(uri) == "weather://status":
        return json.dumps({
            "status": "healthy",
            "server": "weather-server",
            "version": "1.0.0",
            "uptime_seconds": int(time.time() - SERVER_START_TIME),
            "tools": ["get_weather", "get_forecast"],
            "supported_cities": list_supported_cities(),
            "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "transport": "stdio",
            "note": "Replace weather_data.py with a real API client for production.",
        }, indent=2)

    raise ValueError(f"Unknown resource URI: {uri!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    print("[weather-server] Starting on stdio transport ...", file=sys.stderr)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
