"""What changed since last week: how much each product's forecast moved, and why, in plain words.

The forecast for the coming days is made twice with the same model: once from all sales, and once from
the sales known a week earlier. Then this week's information is swapped in one group at a time — last
week's sales, last year's pattern for these dates, being a week closer, price, outside factors — and each
swap's effect on the total is that group's share of the change. Swaps are done in a fixed order, so the
shares add up exactly to the change.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .model import MAX_GAP, DemandForecaster, SeriesContext, _extend

RECENT = ["log_level", "trend_ratio", "level_cv", "recent_7", "recent_14", "zero_share", "days_since_sale",
          "dow_ratio"]
GROUPS = [
    ("Last week's sales", RECENT),
    ("Last year's pattern", ["ly_ratio", "cat_ly_ratio"]),
    ("A week closer", ["gap"]),
    ("Price vs recent weeks", ["rel_price", "price_vs_hist", "price_index", "promo"]),
]


def _features(fc: DemandForecaster, pid: str, hist_end: pd.Timestamp, start: pd.Timestamp, horizon: int):
    full = fc.sales[fc.sales["product_id"] == pid]
    ctx = fc.context_for(pid, full[full["date"] <= hist_end])
    n = len(ctx.units)
    first = int((start - ctx.start).days)
    t = np.arange(first, first + horizon)
    span = t[-1] + 1
    reg = np.full(horizon, float(ctx.regular[-1]))
    ext = SeriesContext(units=ctx.units, price=_extend(ctx.price[:n], span, first, reg),
                        regular=_extend(ctx.regular[:n], span, first, reg),
                        promo=_extend(ctx.promo[:n], span, first, np.zeros(horizon)).astype(int), start=ctx.start,
                        category=ctx.category, price_index=ctx.price_index, cat_units=ctx.cat_units,
                        drivers=fc.driver_matrix(ctx.start, span), bulk_allowance=ctx.bulk_allowance)
    feats = fc._featurise(ext, t, t - np.minimum(t - (n - 1), MAX_GAP))
    return feats.reset_index(drop=True), ctx.bulk_allowance


def _total(fc: DemandForecaster, feats: pd.DataFrame, allowance: float) -> float:
    mean = fc.models["mean"].predict(feats[fc.features]) * (feats["_level"].to_numpy() + 0.5) * fc.bias_factor
    return float((np.clip(mean, 0, None) + allowance).sum())


def explain(fc: DemandForecaster, pid: str, horizon: int = 30, days_back: int = 7) -> dict:
    """The forecast for the next `horizon` days now vs `days_back` days ago, and each group's share of the change."""
    last = fc.sales["date"].max()
    then = last - pd.Timedelta(days=days_back)
    start = last + pd.Timedelta(days=1)
    new_f, new_a = _features(fc, pid, last, start, horizon)
    old_f, old_a = _features(fc, pid, then, start, horizon)
    old_total, new_total = _total(fc, old_f, old_a), _total(fc, new_f, new_a)
    groups = list(GROUPS) + [("Outside factors", [c for c in fc.features if c.startswith("drv_")])]
    parts, cur, cur_a, before = [], old_f.copy(), old_a, old_total
    for name, cols in groups:
        cols = [c for c in cols if c in cur]
        if not cols:
            continue
        cur[cols] = new_f[cols].to_numpy()
        if name == "Last week's sales":
            cur["_level"] = new_f["_level"].to_numpy()  # the recent level sets the scale
            cur_a = new_a
        after = _total(fc, cur, cur_a)
        parts.append((name, after - before))
        before = after
    parts.append(("Other", new_total - before))

    # How last week actually went against what the earlier forecast expected for it.
    week_f, week_a = _features(fc, pid, then, then + pd.Timedelta(days=1), days_back)
    expected = _total(fc, week_f, week_a)
    hist = fc.sales[(fc.sales["product_id"] == pid) & (fc.sales["date"] > then)]
    actual = float(hist["units_sold"].sum())
    return {"product_id": pid, "product_name": hist["product_name"].iloc[0] if len(hist) else pid,
            "before": old_total, "now": new_total, "change": new_total - old_total,
            "change_pct": (new_total / old_total - 1) if old_total > 0 else np.nan,
            "parts": [(n, v) for n, v in parts if abs(v) >= 0.005 * max(old_total, 1) or n != "Other"],
            "last_week_actual": actual, "last_week_expected": expected, "horizon": horizon}


def sentence(e: dict) -> str:
    """The change in one or two plain sentences."""
    pct = e["change_pct"]
    if np.isnan(pct) or abs(pct) < 0.02:
        head = f"About the same as a week ago ({e['now']:,.0f} units over the next {e['horizon']} days)."
    else:
        head = (f"{'Up' if pct > 0 else 'Down'} {abs(pct):.0%} to {e['now']:,.0f} units over the next "
                f"{e['horizon']} days ({e['change']:+,.0f}).")
    drivers = sorted((p for p in e["parts"] if p[0] != "Other" and abs(p[1]) >= 0.5), key=lambda p: -abs(p[1]))[:2]
    why = ""
    if drivers and abs(e["change"]) >= 0.5:
        why = " Mainly: " + "; ".join(f"{name.lower()} ({value:+,.0f})" for name, value in drivers) + "."
    exp, act = e["last_week_expected"], e["last_week_actual"]
    week = ""
    if exp > 0:
        rel = act / exp - 1
        word = "more than" if rel > 0.1 else "less than" if rel < -0.1 else "close to"
        week = f" Last week sold {act:,.0f} — {word} the {exp:,.0f} expected."
    return head + why + week


def all_products(fc: DemandForecaster, horizon: int = 30) -> pd.DataFrame:
    rows = []
    for pid in fc.sales["product_id"].unique():
        e = explain(fc, pid, horizon)
        rows.append({**{k: e[k] for k in ("product_id", "product_name", "before", "now", "change", "change_pct",
                                          "last_week_actual", "last_week_expected")}, "why": sentence(e)})
    out = pd.DataFrame(rows)
    return out.reindex(out["change"].abs().sort_values(ascending=False).index).reset_index(drop=True)
