"""Tool functions and helpers for the weather_reporting skill.

In production these would call a real weather API.
For the demo they return deterministic mock data.
"""

from __future__ import annotations

import sys
import os

# Allow standalone execution: python skills/weather_tools.py
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from skill_base import SkillInputError


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

MOCK_WEATHER: dict[str, dict] = {
    "Tokyo, JP": {"temp_c": 22, "humidity": 65, "condition": "partly cloudy"},
    "London, GB": {"temp_c": 14, "humidity": 78, "condition": "rain"},
    "New York, US": {"temp_c": 28, "humidity": 55, "condition": "sunny"},
    "Sydney, AU": {"temp_c": 19, "humidity": 70, "condition": "clear"},
    "Berlin, DE": {"temp_c": 16, "humidity": 60, "condition": "cloudy"},
}


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


def get_weather(city: str) -> dict:
    """Get current weather for a city.

    In production: calls a weather REST API.
    For the demo: returns mock data keyed by 'City, CC'.

    Args:
        city: City name with country code, e.g. "Tokyo, JP".

    Returns:
        Raw weather dict with temp_c, humidity, condition.

    Raises:
        ValueError: If the city is not found in the database.
    """
    data = MOCK_WEATHER.get(city)
    if data is None:
        raise ValueError(f"City '{city}' not found in weather database")
    return {"city": city, **data}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def validate_weather_input(params: dict) -> dict:
    """Validate that the city parameter includes a country code.

    Args:
        params: Parameters dict; must contain 'city' key.

    Returns:
        The (potentially corrected) params dict.

    Raises:
        SkillInputError: If the city is missing or has no country code.
    """
    city = params.get("city", "")
    if not city:
        raise SkillInputError(
            message="City parameter is required.",
            suggestion="Provide a city name with country code, e.g. 'Tokyo, JP'.",
        )
    if "," not in city:
        raise SkillInputError(
            message="City must include a country code for accurate results.",
            suggestion=f"Try '{city}, JP' or '{city}, US' instead of '{city}'.",
            fix_action="append_country_code",
        )
    return params


# ---------------------------------------------------------------------------
# Normaliser
# ---------------------------------------------------------------------------


def normalize_weather_output(raw: dict) -> dict:
    """Normalise raw API response into a consistent agent-facing format.

    Args:
        raw: Raw dict from get_weather().

    Returns:
        Normalised dict with typed temperature, humidity, conditions.
    """
    temp_c = raw.get("temp_c", 0)
    fahrenheit = round(temp_c * 9 / 5 + 32)
    return {
        "location": raw.get("city", "Unknown"),
        "temperature": {
            "celsius": temp_c,
            "fahrenheit": fahrenheit,
            "display": f"{temp_c}°C / {fahrenheit}°F",
        },
        "humidity_percent": raw.get("humidity"),
        "conditions": raw.get("condition", "Unknown"),
        "reported_at": "2026-05-01T12:00:00Z",
    }


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def weather_fallback(params: dict, error: Exception) -> str:
    """Provide graceful degradation when the weather service is unavailable.

    Args:
        params: The params that were passed to the tool.
        error:  The exception that was raised.

    Returns:
        A user-facing message explaining the failure and suggesting alternatives.
    """
    city = params.get("city", "the specified location")
    return (
        f"Weather data for {city} is temporarily unavailable due to a "
        f"service outage. Please try again in a few minutes, or check "
        f"a weather website directly for current conditions."
    )
