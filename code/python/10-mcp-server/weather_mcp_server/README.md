# MCP Weather Server

A production-ready MCP server that exposes weather tools over the stdio transport.

## Tools

| Tool | Description |
|------|-------------|
| `get_weather(city, units)` | Current temperature, humidity, wind speed, conditions |
| `get_forecast(city, days)` | Multi-day forecast (1–10 days) |

## Resource

| URI | Description |
|-----|-------------|
| `weather://status` | Server health, uptime, supported cities |

## Quick Start

```bash
pip install mcp
python server.py          # Starts listening on stdio
```

## Connect from an MCP Client

```python
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

params = StdioServerParameters(command="python", args=["server.py"])

async with stdio_client(params) as (read, write):
    async with ClientSession(read, write) as session:
        await session.initialize()
        tools = await session.list_tools()
        result = await session.call_tool("get_weather", {"city": "Tokyo, JP"})
        print(result.content[0].text)
```

## Production Notes

- Replace `weather_data.py` with a real API client (OpenWeatherMap, WeatherAPI, etc.)
- The `RateLimiter` in `tools.py` is a stub — replace with Redis-backed limiting
- All logs go to **stderr**; MCP protocol traffic uses stdin/stdout
