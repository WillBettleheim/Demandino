"""Product life stages: launching, growing, mature, declining or at end of life.

Each product is placed from its age, its growth against the same months last year (which removes
seasonality; the last 90 days against the 90 before when there's no year of history yet), and how far
recent sales sit below the product's best 90 days. The stage comes with what to do about stock: grow it,
keep it steady, or stop reordering before it turns into dead stock.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAGES = ["Launching", "Growing", "Mature", "Declining", "End of life"]
LAUNCH_DAYS = 120       # younger products are still launching
GROWTH = 0.15           # ±15% against last year counts as growing / shrinking
NEAR_PEAK = 0.7         # declining products sell under 70% of their best 90 days…
END_OF_LIFE = 0.3       # …and end-of-life ones under 30%

ADVICE = {
    "Launching": "Too new to judge. Order little and often, and watch the weekly sales.",
    "Growing": "Demand is rising: keep more safety stock and order earlier so you don't run out.",
    "Mature": "Steady seller: regular reorders, and protect the margin — it rarely needs discounts.",
    "Declining": "Sales are falling for real (not just seasonally): order smaller amounts and plan a successor.",
    "End of life": "Sales have mostly gone: stop reordering and clear the stock that's left.",
}


def _window(units: pd.Series, end: pd.Timestamp, days: int) -> float:
    w = units[(units.index > end - pd.Timedelta(days=days)) & (units.index <= end)]
    return float(w.mean()) if len(w) else np.nan


def classify(g: pd.DataFrame) -> dict:
    """Stage and evidence for one product's daily sales (columns: date, units_sold)."""
    u = g.set_index("date")["units_sold"].astype(float).sort_index()
    sold = u[u > 0]
    last = u.index.max()
    first = sold.index.min() if len(sold) else u.index.min()
    age = int((last - first).days) + 1
    recent = _window(u, last, 90)
    prior = _window(u, last - pd.Timedelta(days=90), 90)
    year_ago = _window(u, last - pd.Timedelta(days=364), 90) if age >= 364 + 90 else np.nan
    yoy = recent / year_ago - 1 if year_ago and year_ago > 0 else np.nan
    vs_prior = recent / prior - 1 if prior and prior > 0 else np.nan
    growth = yoy if not np.isnan(yoy) else vs_prior
    rolling = u.rolling(90, min_periods=60).mean()
    peak = float(rolling.max()) if rolling.notna().any() else recent
    of_peak = recent / peak if peak and peak > 0 else np.nan

    if age < LAUNCH_DAYS:
        stage = "Launching"
    elif not np.isnan(of_peak) and of_peak < END_OF_LIFE and (np.isnan(growth) or growth < -GROWTH):
        stage = "End of life"
    elif not np.isnan(growth) and growth < -GROWTH and (np.isnan(of_peak) or of_peak < NEAR_PEAK):
        stage = "Declining"
    elif not np.isnan(growth) and growth > GROWTH:
        stage = "Growing"
    else:
        stage = "Mature"
    return {"stage": stage, "age_days": age, "recent_daily": recent, "growth": growth,
            "growth_basis": "vs last year" if not np.isnan(yoy) else "vs previous 90 days",
            "of_peak": of_peak, "advice": ADVICE[stage]}


def stages(sales: pd.DataFrame) -> pd.DataFrame:
    """One row per product with its stage, the evidence and the advice, ordered by stage."""
    rows = []
    for pid, g in sales.groupby("product_id"):
        rows.append({"product_id": pid, "product_name": g["product_name"].iloc[0], "category": g["category"].iloc[0],
                     **classify(g[["date", "units_sold"]])})
    out = pd.DataFrame(rows)
    out["stage"] = pd.Categorical(out["stage"], STAGES, ordered=True)
    return out.sort_values(["stage", "recent_daily"], ascending=[True, False]).reset_index(drop=True)


def curve(g: pd.DataFrame) -> pd.Series:
    """Weekly sales as a share of the product's best (13-week smoothed, so seasons don't hide the trend)."""
    w = g.set_index("date")["units_sold"].resample("W").sum().rolling(13, min_periods=4).mean().dropna()
    return w / w.max() if w.max() > 0 else w
