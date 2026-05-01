"""Shared tool functions for stock-related skills.

Both stock_price and stock_analysis import from here so the validator
and mock data are not duplicated.
"""

from __future__ import annotations

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skill_base import SkillInputError


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_STOCKS: dict[str, dict] = {
    "AAPL": {
        "price_usd": 192.35,
        "change_pct": 1.2,
        "week_change_pct": 3.1,
        "high_52w": 220.20,
        "low_52w": 164.08,
        "currency": "USD",
    },
    "MSFT": {
        "price_usd": 415.10,
        "change_pct": 0.8,
        "week_change_pct": 2.4,
        "high_52w": 468.35,
        "low_52w": 309.45,
        "currency": "USD",
    },
    "GOOGL": {
        "price_usd": 171.80,
        "change_pct": -0.5,
        "week_change_pct": 1.5,
        "high_52w": 207.05,
        "low_52w": 130.67,
        "currency": "USD",
    },
    "TSLA": {
        "price_usd": 175.20,
        "change_pct": -2.3,
        "week_change_pct": -4.1,
        "high_52w": 299.29,
        "low_52w": 138.80,
        "currency": "USD",
    },
    "AMZN": {
        "price_usd": 188.40,
        "change_pct": 0.3,
        "week_change_pct": 0.8,
        "high_52w": 231.83,
        "low_52w": 151.61,
        "currency": "USD",
    },
}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def get_stock_price(ticker: str) -> dict:
    """Get current stock price data for a ticker symbol.

    In production: calls a financial data API.
    For the demo: returns mock data.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".

    Returns:
        Raw price dict with price_usd, change percentages, 52w range.

    Raises:
        ValueError: If the ticker is not in the mock database.
    """
    data = MOCK_STOCKS.get(ticker.upper())
    if data is None:
        raise ValueError(f"Ticker '{ticker}' not found in stock database")
    return {"ticker": ticker.upper(), **data}


# ---------------------------------------------------------------------------
# Validator (shared by stock_price and stock_analysis skills)
# ---------------------------------------------------------------------------


def validate_stock_input(params: dict) -> dict:
    """Validate ticker format: 1-5 uppercase letters only.

    Args:
        params: Parameters dict; must contain 'ticker' key.

    Returns:
        Params with ticker normalised to uppercase.

    Raises:
        SkillInputError: If ticker is missing or has invalid format.
    """
    ticker = params.get("ticker", "")
    if not ticker:
        raise SkillInputError(
            message="Ticker symbol is required.",
            suggestion="Provide a stock ticker like 'AAPL' or 'MSFT'.",
        )
    if not re.match(r"^[A-Za-z]{1,5}$", ticker):
        raise SkillInputError(
            message=f"Invalid ticker '{ticker}'. Tickers must be 1–5 letters only.",
            suggestion="Examples: AAPL, MSFT, GOOGL, TSLA.",
            fix_action="fix_ticker_format",
        )
    params["ticker"] = ticker.upper()
    return params


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


def normalize_stock_output(raw: dict) -> dict:
    """Normalise raw stock data into a consistent agent-facing format."""
    direction = "up" if raw.get("change_pct", 0) >= 0 else "down"
    price = raw.get("price_usd", 0)
    return {
        "ticker": raw.get("ticker"),
        "price": f"${price:.2f}",
        "price_usd": price,
        "currency": raw.get("currency", "USD"),
        "change": {
            "today_pct": raw.get("change_pct"),
            "week_pct": raw.get("week_change_pct"),
            "direction": direction,
        },
        "range_52w": {
            "high": raw.get("high_52w"),
            "low": raw.get("low_52w"),
        },
    }


# ---------------------------------------------------------------------------
# Fallback (shared)
# ---------------------------------------------------------------------------


def stock_fallback(params: dict, error: Exception) -> str:
    """Provide graceful degradation when stock data is unavailable."""
    ticker = params.get("ticker", "the requested ticker")
    return (
        f"Stock data for {ticker} is temporarily unavailable. "
        f"Please try again shortly or check a financial website for current prices."
    )
