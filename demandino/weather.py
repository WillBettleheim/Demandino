"""Daily weather as an outside factor, from Open-Meteo (free, no key).

Only a place's coordinates and a date range are sent — never sales data. Past days come from the
historical archive, the next 16 days from the weather forecast, and later days from the place's
climate normal (the average of the same calendar day in past years), so the model always has a value.
Requests go through the connector rules in integrations.py (HTTPS, these hosts only, timeouts, size cap).
"""
from __future__ import annotations

import json
import urllib.parse

import numpy as np
import pandas as pd

from .integrations import ConnectorError, _check_url, _get

GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
FORECAST = "https://api.open-meteo.com/v1/forecast"
HOSTS = ("geocoding-api.open-meteo.com", "archive-api.open-meteo.com", "api.open-meteo.com")
VARIABLES = {"temperature_2m_mean": "temperature", "precipitation_sum": "rain"}
FUTURE_DAYS = 200


def _json(url: str, params: dict) -> dict:
    full = _check_url(f"{url}?{urllib.parse.urlencode(params)}", hosts=HOSTS)
    data, _ = _get(full)
    try:
        return json.loads(data)
    except ValueError:
        raise ConnectorError("The weather service sent an unreadable answer.") from None


def find_place(name: str) -> dict:
    """The best match for a town or city name: {name, country, latitude, longitude}."""
    name = name.strip()[:80]
    if len(name) < 2:
        raise ConnectorError("Type a town or city name.")
    found = _json(GEOCODE, {"name": name, "count": 1, "language": "en", "format": "json"}).get("results") or []
    if not found:
        raise ConnectorError(f"No place called '{name}' was found.")
    r = found[0]
    return {"name": r.get("name", name), "country": r.get("country", ""), "latitude": float(r["latitude"]),
            "longitude": float(r["longitude"])}


def _daily(payload: dict) -> pd.DataFrame:
    d = payload.get("daily") or {}
    if "time" not in d:
        raise ConnectorError("The weather service returned no daily data.")
    df = pd.DataFrame({VARIABLES[k]: d.get(k) for k in VARIABLES}, index=pd.to_datetime(d["time"]))
    return df.astype(float)


def climate_normal(history: pd.DataFrame, days: pd.DatetimeIndex) -> pd.DataFrame:
    """Average of the same calendar day across past years (smoothed over a week)."""
    by_day = history.groupby(history.index.day_of_year).mean()
    by_day = by_day.reindex(range(1, 367)).interpolate(limit_direction="both")
    smooth = pd.concat([by_day.iloc[-7:], by_day, by_day.iloc[:7]]).rolling(15, center=True).mean().iloc[7:-7]
    return pd.DataFrame(smooth.loc[days.day_of_year].to_numpy(), index=days, columns=history.columns)


def daily_weather(latitude: float, longitude: float, start: pd.Timestamp, end: pd.Timestamp,
                  today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Daily temperature (°C) and rain (mm) from `start` to `end` + FUTURE_DAYS, gaps filled sensibly."""
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    start = pd.Timestamp(start).normalize()
    past_end = min(pd.Timestamp(end), today - pd.Timedelta(days=6))  # the archive lags by about five days
    loc = {"latitude": round(latitude, 3), "longitude": round(longitude, 3), "timezone": "auto",
           "daily": ",".join(VARIABLES)}
    history = _daily(_json(ARCHIVE, {**loc, "start_date": f"{start:%Y-%m-%d}", "end_date": f"{past_end:%Y-%m-%d}"}))
    try:
        ahead = _daily(_json(FORECAST, {**loc, "past_days": 7, "forecast_days": 16}))
    except ConnectorError:
        ahead = history.iloc[:0]
    days = pd.date_range(start, pd.Timestamp(end) + pd.Timedelta(days=FUTURE_DAYS), freq="D")
    out = history.combine_first(ahead).reindex(days)
    normal = climate_normal(history.dropna(), days)
    out = out.fillna(normal)
    out["rain"] = out["rain"].clip(lower=0)
    return out.astype(float).round(2).rename_axis("date")


def describe(df: pd.DataFrame) -> str:
    t, r = df["temperature"], df["rain"]
    return (f"{len(df):,} days · {t.min():.0f}–{t.max():.0f} °C · average rain {r.mean():.1f} mm/day "
            f"({np.mean(r > 1):.0%} of days wet)")
