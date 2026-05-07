"""Mock weather data and API client.

In production, replace the functions below with real API calls
(e.g. OpenWeatherMap, WeatherAPI, Tomorrow.io).  The rest of the
server is API-agnostic and does not need to change.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta


class CityNotFoundError(Exception):
    """Raised when the requested city is not in the database."""

    def __init__(self, city: str) -> None:
        self.city = city
        super().__init__(
            f"City not found: {city!r}. "
            "Try 'London, UK', 'Tokyo, JP', or call list_supported_cities()."
        )


# ---------------------------------------------------------------------------
# Mock database  (key = normalized city name, lowercase, no country code)
# ---------------------------------------------------------------------------
_WEATHER_DB: dict[str, dict] = {
    "tokyo":     {"temp_c": 22, "humidity": 68, "condition": "partly cloudy",  "wind_kph": 14, "country": "JP"},
    "london":    {"temp_c": 12, "humidity": 80, "condition": "overcast",        "wind_kph": 20, "country": "UK"},
    "new york":  {"temp_c": 18, "humidity": 60, "condition": "sunny",           "wind_kph": 12, "country": "US"},
    "paris":     {"temp_c": 16, "humidity": 72, "condition": "light rain",      "wind_kph":  8, "country": "FR"},
    "sydney":    {"temp_c": 20, "humidity": 65, "condition": "clear",           "wind_kph": 18, "country": "AU"},
    "berlin":    {"temp_c": 10, "humidity": 75, "condition": "cloudy",          "wind_kph": 22, "country": "DE"},
    "dubai":     {"temp_c": 38, "humidity": 45, "condition": "sunny",           "wind_kph": 16, "country": "AE"},
    "moscow":    {"temp_c":  5, "humidity": 70, "condition": "snow",            "wind_kph": 10, "country": "RU"},
    "singapore": {"temp_c": 30, "humidity": 85, "condition": "thunderstorm",    "wind_kph": 24, "country": "SG"},
    "toronto":   {"temp_c":  8, "humidity": 62, "condition": "clear",           "wind_kph": 15, "country": "CA"},
    "shanghai":  {"temp_c": 24, "humidity": 73, "condition": "hazy",            "wind_kph": 11, "country": "CN"},
    "mumbai":    {"temp_c": 32, "humidity": 82, "condition": "humid",           "wind_kph":  9, "country": "IN"},
    "cairo":     {"temp_c": 35, "humidity": 30, "condition": "sunny",           "wind_kph": 13, "country": "EG"},
    "amsterdam": {"temp_c": 11, "humidity": 78, "condition": "rain",            "wind_kph": 25, "country": "NL"},
    "seoul":     {"temp_c": 19, "humidity": 66, "condition": "partly cloudy",   "wind_kph": 17, "country": "KR"},
}

_FORECAST_CONDITIONS = [
    "sunny", "partly cloudy", "cloudy", "light rain",
    "rain", "thunderstorm", "clear", "overcast", "snow",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(city: str) -> str:
    """'Tokyo, JP' → 'tokyo',  'New York, US' → 'new york'."""
    return city.split(",")[0].strip().lower()


def _c_to_f(temp_c: float) -> float:
    return round(temp_c * 9 / 5 + 32, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_current_weather(city: str, units: str = "celsius") -> dict:
    """Return current weather for *city*.

    Args:
        city:  City name with optional country code, e.g. ``"Tokyo, JP"``.
        units: ``"celsius"`` (default) or ``"fahrenheit"``.

    Returns:
        Dict with temperature, humidity, condition, wind_kph, timestamp.

    Raises:
        CityNotFoundError: city is not in the mock database.
    """
    key = _normalize(city)
    if key not in _WEATHER_DB:
        raise CityNotFoundError(city)

    data = _WEATHER_DB[key]
    temp = data["temp_c"] if units == "celsius" else _c_to_f(data["temp_c"])

    return {
        "city": city,
        "country": data["country"],
        "temperature": temp,
        "units": units,
        "humidity": data["humidity"],
        "condition": data["condition"],
        "wind_kph": data["wind_kph"],
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": "mock_data",  # → "openweathermap" in production
    }


def get_forecast(city: str, days: int = 5) -> dict:
    """Return a multi-day forecast for *city*.

    Args:
        city: City name with optional country code.
        days: Number of days (1–10, default 5).

    Returns:
        Dict with a ``forecast`` list of daily summaries.

    Raises:
        CityNotFoundError: city is not in the mock database.
    """
    key = _normalize(city)
    if key not in _WEATHER_DB:
        raise CityNotFoundError(city)

    days = max(1, min(days, 10))
    base = _WEATHER_DB[key]
    base_temp = base["temp_c"]

    # Deterministic per city for reproducible demos
    rng = random.Random(hash(key))

    forecast_days = []
    for i in range(days):
        date = (datetime.utcnow() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        day_name = (datetime.utcnow() + timedelta(days=i + 1)).strftime("%a")
        high_c = round(base_temp + rng.uniform(1.0, 5.0), 1)
        low_c = round(base_temp - rng.uniform(1.0, 5.0), 1)
        forecast_days.append({
            "date": date,
            "day": day_name,
            "high_c": high_c,
            "low_c": low_c,
            "condition": rng.choice(_FORECAST_CONDITIONS),
            "precipitation_chance_pct": rng.randint(0, 100),
        })

    return {
        "city": city,
        "days": days,
        "forecast": forecast_days,
        "source": "mock_data",
    }


def list_supported_cities() -> list[str]:
    """Return human-readable names of all cities in the mock database."""
    return [k.title() for k in sorted(_WEATHER_DB)]
