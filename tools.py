"""Tools the model can call: live weather (Open-Meteo), a budget check, and currency conversion (Frankfurter).

Every tool returns a plain dict and never raises. Failures come back as
{"error": "..."} so the model can read the message and react to it.
"""
from __future__ import annotations

import logging
import time
import unicodedata
from datetime import date, datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
RATES_URL = "https://api.frankfurter.dev/v1/latest"  # ECB reference rates, no API key
FORECAST_WINDOW_DAYS = 16  # Open-Meteo forecasts today plus the next 15 days
TIMEOUT_S = 10
CACHE_TTL_S = 20 * 60  # forecasts and rates barely move in 20 minutes; this spares the demo repeat calls

_cache: Dict[Any, Tuple[float, Any]] = {}


def clear_cache() -> None:
    _cache.clear()


def _cached(key: Any, fetch: Callable[[], Any]) -> Any:
    """Return a fresh cached value for `key`, else call `fetch`. Exceptions and empty results are not cached."""
    now = time.monotonic()
    hit = _cache.get(key)
    if hit and now - hit[0] < CACHE_TTL_S:
        return hit[1]
    value = fetch()
    if value:
        _cache[key] = (now, value)
    return value

# WMO weather interpretation codes used by Open-Meteo.
WEATHER_CODES = {
    0: "Clear sky", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Freezing fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle",
    56: "Freezing drizzle", 57: "Heavy freezing drizzle",
    61: "Light rain", 63: "Rain", 65: "Heavy rain",
    66: "Freezing rain", 67: "Heavy freezing rain",
    71: "Light snow", 73: "Snow", 75: "Heavy snow", 77: "Snow grains",
    80: "Light rain showers", 81: "Rain showers", 82: "Violent rain showers",
    85: "Snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Severe thunderstorm with hail",
}


def _place_name(hit: Dict[str, Any]) -> str:
    parts = [hit.get("name"), hit.get("admin1"), hit.get("country")]
    seen: List[str] = []
    for part in parts:
        if part and part not in seen:
            seen.append(part)
    return ", ".join(seen)


def _fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)).casefold().strip()


def _pick_place(hits: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
    """The geocoder is fuzzy ("Goa" can return Genoa first), so prefer exact name matches, largest first."""
    wanted = _fold(query)
    exact = [h for h in hits if _fold(h.get("name", "")) == wanted]
    pool = exact or hits
    return max(pool, key=lambda h: h.get("population") or 0) if exact else pool[0]


def _geocode(city: str, country_code: Optional[str]) -> List[Dict[str, Any]]:
    name = str(city).strip()
    country = str(country_code).strip().upper() if country_code and str(country_code).strip() else ""

    def fetch() -> List[Dict[str, Any]]:
        params = {"name": name, "count": 10, "language": "en", "format": "json"}
        if country:
            params["countryCode"] = country
        resp = requests.get(GEOCODE_URL, params=params, timeout=TIMEOUT_S)
        resp.raise_for_status()
        return resp.json().get("results") or []

    return _cached(("geocode", name.casefold(), country), fetch)


def _daily_forecast(latitude: float, longitude: float, start: date, end: date) -> Dict[str, Any]:
    def fetch() -> Dict[str, Any]:
        resp = requests.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                "timezone": "auto",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
            timeout=TIMEOUT_S,
        )
        resp.raise_for_status()
        return resp.json().get("daily") or {}

    return _cached(("forecast", latitude, longitude, start, end), fetch)


def get_weather(
    city: str, start_date: str, days: int = 3, country_code: Optional[str] = None, today: Optional[date] = None
) -> Dict[str, Any]:
    """Daily forecast for `city` from `start_date` (YYYY-MM-DD) for `days` days.

    `country_code` is an optional ISO 3166-1 alpha-2 code (e.g. "IN") to disambiguate the city name.
    """
    today = today or date.today()
    last_forecast_day = today + timedelta(days=FORECAST_WINDOW_DAYS - 1)

    if not city or not str(city).strip():
        return {"error": "No city given."}
    try:
        start = datetime.strptime(str(start_date).strip(), "%Y-%m-%d").date()
    except ValueError:
        return {"error": f"Could not read start_date {start_date!r}; use YYYY-MM-DD."}
    try:
        days = max(1, min(int(days), FORECAST_WINDOW_DAYS))
    except (TypeError, ValueError):
        return {"error": f"Could not read days {days!r}; use a whole number."}

    if start < today:
        return {"error": f"start_date {start} is in the past (today is {today})."}
    if start > last_forecast_day:
        return {
            "error": (
                f"No forecast available: trip starts {start}, but forecasts only reach {last_forecast_day}. "
                "Say this to the user and give typical seasonal advice, clearly labelled as such."
            )
        }
    end = min(start + timedelta(days=days - 1), last_forecast_day)

    try:
        hits = _geocode(city, country_code)
        if not hits:
            return {"error": f"Could not find a place called {city!r}. Try a specific city or town, or ask the user to check the spelling."}
        hit = _pick_place(hits, str(city))
        daily = _daily_forecast(hit["latitude"], hit["longitude"], start, end)
    except requests.RequestException as exc:
        return {"error": f"Weather service unreachable ({type(exc).__name__}). Say the forecast is unavailable right now."}
    except (KeyError, ValueError):
        return {"error": "Weather service returned data in an unexpected format."}

    rows = []
    for i, day in enumerate(daily.get("time", [])):
        code = (daily.get("weather_code") or [None] * (i + 1))[i]
        rows.append({
            "date": day,
            "temp_max_c": (daily.get("temperature_2m_max") or [None] * (i + 1))[i],
            "temp_min_c": (daily.get("temperature_2m_min") or [None] * (i + 1))[i],
            "rain_chance_pct": (daily.get("precipitation_probability_max") or [None] * (i + 1))[i],
            "conditions": WEATHER_CODES.get(code, "Unknown"),
        })

    result: Dict[str, Any] = {"place": _place_name(hit), "forecast": rows}
    if end < start + timedelta(days=days - 1):
        result["note"] = f"Forecast stops at {end}; later days have no forecast, so use typical seasonal advice for them."
    return result


def _to_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", "").strip())
        except ValueError:
            return None
    return None


def check_budget(activities: List[Dict[str, Any]], budget: float) -> Dict[str, Any]:
    """Total the activity costs and compare them with the budget."""
    limit = _to_number(budget)
    if limit is None or limit < 0:
        return {"error": f"Could not read budget {budget!r}; use a non-negative number."}

    total = 0.0
    for item in activities or []:
        cost = _to_number(item.get("estimated_cost")) if isinstance(item, dict) else None
        if cost is None:
            name = item.get("name") if isinstance(item, dict) else item
            return {"error": f"Activity {name!r} has no usable estimated_cost."}
        total += cost

    return {
        "total": round(total, 2),
        "budget": round(limit, 2),
        "remaining": round(limit - total, 2),
        "status": "within" if total <= limit else "over",
    }


class _UnsupportedCurrency(Exception):
    pass


def _currency_code(value: Any) -> Optional[str]:
    code = str(value or "").strip().upper()
    return code if len(code) == 3 and code.isalpha() else None


def _fetch_rate(src: str, dst: str) -> Dict[str, Any]:
    resp = requests.get(RATES_URL, params={"base": src, "symbols": dst}, timeout=TIMEOUT_S)
    if resp.status_code == 404:  # Frankfurter's answer for a code it does not carry
        raise _UnsupportedCurrency
    resp.raise_for_status()
    body = resp.json()
    return {"rate": float(body["rates"][dst]), "rate_date": body["date"]}


def convert_currency(amount: float, from_currency: str, to_currency: str) -> Dict[str, Any]:
    """Convert `amount` between two ISO 4217 currencies at the latest ECB reference rate."""
    value = _to_number(amount)
    if value is None or value < 0:
        return {"error": f"Could not read amount {amount!r}; use a non-negative number."}
    src, dst = _currency_code(from_currency), _currency_code(to_currency)
    if src is None or dst is None:
        return {"error": f"Use 3-letter currency codes such as INR, EUR or USD (got {from_currency!r} and {to_currency!r})."}

    result: Dict[str, Any] = {"amount": value, "from": src, "to": dst}
    if src == dst:
        return {**result, "rate": 1.0, "converted": round(value, 2)}
    try:
        quote = _cached(("rate", src, dst), lambda: _fetch_rate(src, dst))
    except _UnsupportedCurrency:
        return {"error": f"Exchange rates for {src} to {dst} are not available. Use your own rough estimate and say it is approximate."}
    except requests.RequestException as exc:
        return {"error": f"Exchange-rate service unreachable ({type(exc).__name__}). Use your own rough estimate and say it is approximate."}
    except (KeyError, ValueError, TypeError):
        return {"error": "Exchange-rate service returned data in an unexpected format."}
    return {**result, **quote, "converted": round(value * quote["rate"], 2)}


TOOL_FUNCTIONS = {"get_weather": get_weather, "check_budget": check_budget, "convert_currency": convert_currency}


def run_tool(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch a model-requested tool call; unknown names and bad arguments become error dicts."""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        log.warning("tool %s: unknown", name)
        return {"error": f"Unknown tool {name!r}."}
    started = time.monotonic()
    try:
        result = func(**args)
    except TypeError as exc:
        log.warning("tool %s: bad arguments", name)
        return {"error": f"Bad arguments for {name}: {exc}"}
    log.info("tool %s: %s in %.2fs", name, "error" if "error" in result else "ok", time.monotonic() - started)
    return result
