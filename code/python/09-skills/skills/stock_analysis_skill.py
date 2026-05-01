"""Stock analysis skill — higher-level analysis built on top of stock_price.

Demonstrates skill composition: stock_analysis depends on stock_price.
When executed, it calls stock_price via the registry, then adds an
analysis layer (52-week position, valuation assessment).
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skill_base import Skill, SkillRegistry, SkillTest
from skills.stock_tools import stock_fallback, validate_stock_input


def create_stock_analysis_skill(registry: SkillRegistry) -> Skill:
    """Create and return the stock_analysis skill.

    Args:
        registry: A SkillRegistry with stock_price already registered.
                  The tool closure captures registry so it can delegate
                  the price lookup to the stock_price skill.
    """

    def analyze_stock(ticker: str) -> dict:
        """Analyse a stock: fetch price via stock_price skill, add context.

        The composition pattern:
          1. Delegate price lookup to stock_price skill via registry.
          2. Add 52-week position calculation and a plain-language assessment.
          3. Return a combined result.
        """
        price_result = registry.execute("stock_price", {"ticker": ticker})
        if not price_result.success:
            raise ValueError(price_result.error)

        data = price_result.data
        price_usd: float = data["price_usd"]
        high_52w: float = data["range_52w"]["high"]
        low_52w: float = data["range_52w"]["low"]

        pct_from_high = (price_usd - high_52w) / high_52w * 100
        pct_from_low = (price_usd - low_52w) / low_52w * 100

        if pct_from_high >= -5:
            assessment = "near 52-week high — strong recent momentum"
        elif pct_from_low <= 10:
            assessment = "near 52-week low — potential value or continued decline"
        else:
            assessment = "mid-range — watch for trend confirmation"

        return {
            "ticker": ticker,
            "current_price": data["price"],
            "price_usd": price_usd,
            "performance": {
                "today_pct": data["change"]["today_pct"],
                "week_pct": data["change"]["week_pct"],
                "direction": data["change"]["direction"],
            },
            "52w_position": {
                "pct_from_high": round(pct_from_high, 1),
                "pct_from_low": round(pct_from_low, 1),
            },
            "assessment": assessment,
            "note": "Informational data only — not financial advice.",
        }

    return Skill(
        name="stock_analysis",
        description=(
            "Get a stock analysis including price, performance, 52-week context, "
            "and a plain-language assessment. Use when users ask about investing in "
            "or researching a specific stock."
        ),
        tool=analyze_stock,
        parameters={
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol, e.g. 'AAPL', 'MSFT'.",
                }
            },
            "required": ["ticker"],
        },
        version="1.0.0",
        tags=["finance", "analysis", "stocks"],
        dependencies=["stock_price"],
        prompt_fragment="""
When using stock_analysis:
- Present current price and today's performance prominently.
- Explain the 52-week context: near high, near low, or mid-range.
- Include the assessment but clarify it is informational, not financial advice.
- Do not recommend buying, selling, or holding — report the data.
        """,
        input_validator=validate_stock_input,
        fallback=stock_fallback,
        test_cases=[
            SkillTest(
                input={"ticker": "AAPL"},
                expect_success=True,
                expect_output_contains=["AAPL", "assessment"],
            ),
            SkillTest(
                input={"ticker": "MSFT"},
                expect_success=True,
                expect_output_contains=["MSFT", "52w_position"],
            ),
            SkillTest(
                input={"ticker": "INVALID123"},  # Fails validation
                expect_success=False,
                expect_output_contains=["Invalid ticker"],
            ),
        ],
    )
