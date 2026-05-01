"""Skills package — reusable agent capability modules.

Each create_*() factory returns a configured Skill instance ready to be
registered with a SkillRegistry.
"""

from skills.weather_skill import create_weather_skill
from skills.stock_price_skill import create_stock_price_skill
from skills.stock_analysis_skill import create_stock_analysis_skill

__all__ = [
    "create_weather_skill",
    "create_stock_price_skill",
    "create_stock_analysis_skill",
]
