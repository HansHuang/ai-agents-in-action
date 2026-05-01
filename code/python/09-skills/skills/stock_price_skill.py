"""Stock price skill — get current price and 52-week range for a ticker."""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skill_base import Skill, SkillTest
from skills.stock_tools import (
    get_stock_price,
    normalize_stock_output,
    stock_fallback,
    validate_stock_input,
)


def create_stock_price_skill() -> Skill:
    """Create and return the stock_price skill."""
    return Skill(
        name="stock_price",
        description=(
            "Get the current stock price, daily change, and 52-week range for a "
            "ticker symbol. Use when users ask about stock prices or market performance."
        ),
        tool=get_stock_price,
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": (
                        "Stock ticker symbol, e.g. 'AAPL', 'MSFT', 'GOOGL'. "
                        "Must be 1–5 letters."
                    ),
                }
            },
            "required": ["ticker"],
        },
        version="1.0.0",
        tags=["finance", "real-time", "stocks"],
        prompt_fragment="""
When using stock_price:
- Always show the current price and today's percentage change.
- Include the 52-week high/low for context.
- Use directional framing: "up 1.2%" or "down 0.5%".
- Do not give buy/sell/hold advice — report facts only.
        """,
        input_validator=validate_stock_input,
        output_normalizer=normalize_stock_output,
        fallback=stock_fallback,
        test_cases=[
            SkillTest(
                input={"ticker": "AAPL"},
                expect_success=True,
                expect_output_contains=["AAPL", "price"],
            ),
            SkillTest(
                input={"ticker": "aapl"},  # Lowercase — normalised to AAPL
                expect_success=True,
                expect_output_contains=["AAPL"],
            ),
            SkillTest(
                input={"ticker": "AAPL123"},  # Invalid: contains digits
                expect_success=False,
                expect_output_contains=["Invalid ticker"],
            ),
            SkillTest(
                input={"ticker": "ZZZZ"},  # Valid format, not in mock data
                expect_fallback=True,
            ),
        ],
    )
