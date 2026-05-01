"""Weather reporting skill — demonstrates all six skill components.

Components used:
  - tool              get_weather (mock data, real API in production)
  - input_validator   validate_weather_input (requires country code)
  - output_normalizer normalize_weather_output (adds display field, °C/°F)
  - fallback          weather_fallback (graceful degradation)
  - prompt_fragment   teaches the agent how to report weather
  - test_cases        3 cases: happy path, missing country code, city not found
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skill_base import Skill, SkillTest
from skills.weather_tools import (
    get_weather,
    normalize_weather_output,
    validate_weather_input,
    weather_fallback,
)


def create_weather_skill() -> Skill:
    """Create and return the weather_reporting skill."""
    return Skill(
        name="weather_reporting",
        description=(
            "Get current weather conditions for a city. Use when users ask "
            "about weather, temperature, humidity, or climate conditions."
        ),
        tool=get_weather,
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": (
                        "City name with ISO country code, e.g. 'Tokyo, JP' "
                        "or 'London, GB'. The country code is required."
                    ),
                }
            },
            "required": ["city"],
        },
        version="1.0.0",
        tags=["weather", "real-time", "public-data"],
        prompt_fragment="""
When using weather_reporting:
- The city parameter MUST include a country code (e.g. "Tokyo, JP", "London, GB").
- Report temperature in both Celsius and Fahrenheit.
- If humidity > 80%, mention that it feels humid.
- If conditions include rain or snow, recommend appropriate gear.
- Keep reports concise: 2-3 sentences.
        """,
        input_validator=validate_weather_input,
        output_normalizer=normalize_weather_output,
        fallback=weather_fallback,
        test_cases=[
            SkillTest(
                input={"city": "Tokyo, JP"},
                expect_success=True,
                expect_output_contains=["Tokyo", "°C", "°F"],
            ),
            SkillTest(
                input={"city": "Tokyo"},  # Missing country code
                expect_success=False,
                expect_output_contains=["country code"],
            ),
            SkillTest(
                input={"city": "NonexistentCity, XX"},
                expect_fallback=True,  # Tool raises, fallback fires
            ),
        ],
    )
