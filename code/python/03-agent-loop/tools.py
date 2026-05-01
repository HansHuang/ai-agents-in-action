"""Tool implementations and OpenAI function-calling definitions.

Each tool has two parts:
  1. A Python function the orchestrator calls after the LLM requests it.
  2. A JSON schema entry in TOOLS that the LLM sees and uses to decide
     when and how to call the tool.

The docstrings and descriptions are read by the LLM — every word matters.

See docs/02-the-agent-loop/01-anatomy-of-an-agent.md — "The Hands (Tools)"
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def get_weather(city: str) -> dict:
    """Return current weather conditions for a city.

    In production this would call a weather API (e.g. OpenWeatherMap).
    Mock data is used here so the agent can run without external API keys.

    Args:
        city: City name with optional country code, e.g. "Shanghai, CN".

    Returns:
        dict with keys: city, temperature_c, condition, humidity_percent, wind_kph
    """
    _mock: dict[str, dict] = {
        "Shanghai": {"temperature_c": 22, "condition": "light rain",     "humidity_percent": 85, "wind_kph": 15},
        "London":    {"temperature_c": 14, "condition": "overcast",        "humidity_percent": 78, "wind_kph": 20},
        "New York":  {"temperature_c": 18, "condition": "partly cloudy",   "humidity_percent": 60, "wind_kph": 25},
        "Paris":     {"temperature_c": 16, "condition": "sunny",           "humidity_percent": 55, "wind_kph": 12},
        "Sydney":    {"temperature_c": 28, "condition": "clear",           "humidity_percent": 45, "wind_kph": 18},
    }
    city_key = city.split(",")[0].strip()
    data = _mock.get(city_key, {"temperature_c": 20, "condition": "clear", "humidity_percent": 55, "wind_kph": 10})
    return {"city": city, **data}


def get_stock_price(ticker: str) -> dict:
    """Return current stock price and daily change for a ticker symbol.

    In production this would call a financial data API (e.g. Alpha Vantage).
    Mock data is used here so the agent can run without external API keys.

    Args:
        ticker: Stock ticker symbol in uppercase, e.g. "AAPL".

    Returns:
        dict with keys: ticker, price_usd, change_percent, currency, market_status
    """
    _mock: dict[str, dict] = {
        "AAPL":  {"price_usd": 192.35, "change_percent":  1.2, "currency": "USD"},
        "GOOGL": {"price_usd": 171.80, "change_percent": -0.5, "currency": "USD"},
        "MSFT":  {"price_usd": 415.10, "change_percent":  0.8, "currency": "USD"},
        "TSLA":  {"price_usd": 175.20, "change_percent": -2.3, "currency": "USD"},
        "AMZN":  {"price_usd": 188.40, "change_percent":  0.3, "currency": "USD"},
    }
    ticker_upper = ticker.upper()
    data = _mock.get(ticker_upper, {"price_usd": 100.00, "change_percent": 0.0, "currency": "USD"})
    return {"ticker": ticker_upper, "market_status": "open", **data}


# ---------------------------------------------------------------------------
# OpenAI function-calling definitions
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Get current weather conditions for a city. "
                "Use this when the user asks about weather, temperature, rain, "
                "humidity, wind, or whether to bring an umbrella or coat. "
                "Always call this tool rather than guessing — weather is dynamic."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": (
                            "City name with optional ISO country code, "
                            "e.g. 'Shanghai, CN', 'London, UK', 'New York, US'. "
                            "Include the country code when the city name is ambiguous."
                        ),
                    }
                },
                "required": ["city"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": (
                "Get the current stock price and daily percentage change for a publicly "
                "traded company. Use this when the user asks about stock price, share "
                "value, investment potential, or financial performance of a company. "
                "Always call this tool rather than using stale training data."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {
                        "type": "string",
                        "description": (
                            "Stock ticker symbol in uppercase, "
                            "e.g. 'AAPL' for Apple, 'GOOGL' for Google, "
                            "'MSFT' for Microsoft, 'TSLA' for Tesla."
                        ),
                    }
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
]
