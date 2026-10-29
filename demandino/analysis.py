"""Sales pattern analysis, price sensitivity, inventory planning and plain-language insights."""
from __future__ import annotations

from statistics import NormalDist
from typing import Callable

import numpy as np
import pandas as pd

from .model import RANGE_LABEL, ForecastResult

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------- patterns
def product_profile(g: pd.DataFrame) -> dict:
    """Descriptive statistics and seasonal patterns for one product's daily history."""
    g = g.sort_values("date").set_index("date")
    u = g["units_sold"]
    regular = g[g["promo"] == 0]["units_sold"]
    promo = g[g["promo"] == 1]["units_sold"]
    last90, prev90 = u.tail(90).sum(), u.iloc[-180:-90].sum() if len(u) >= 180 else np.nan
    ly90 = u.iloc[-455:-365].sum() if len(u) >= 455 else np.nan

    # Seasonal indices computed on regular-price days so promos don't distort them.
    base = regular if len(regular) > 60 else u
    dow_idx = base.groupby(base.index.day_of_week).mean()
    dow_idx = (dow_idx / dow_idx.mean()).reindex(range(7))
    month_idx = base.groupby(base.index.month).mean()
    month_idx = (month_idx / month_idx.mean()).reindex(range(1, 13))

    return {
        "days": len(u),
        "total_units": float(u.sum()),
        "avg_daily": float(u.mean()),
        "avg_daily_90": float(u.tail(90).mean()),
        "revenue": float((u * g["price"]).sum()),
        "growth_90_vs_prev": float(last90 / prev90 - 1) if prev90 and prev90 > 0 else np.nan,
        "growth_yoy_90": float(last90 / ly90 - 1) if ly90 and ly90 > 0 else np.nan,
        "cv": float(u.std() / u.mean()) if u.mean() > 0 else np.nan,
        "zero_days_pct": float((u == 0).mean()),
        "promo_share": float(g["promo"].mean()),
        "promo_lift": float(promo.mean() / regular.mean() - 1) if len(promo) > 5 and regular.mean() > 0 else np.nan,
        "dow_index": dow_idx,
        "month_index": month_idx,
        "best_dow": DOW_NAMES[int(dow_idx.idxmax())],
        "worst_dow": DOW_NAMES[int(dow_idx.idxmin())],
        "peak_month": MONTH_NAMES[int(month_idx.idxmax()) - 1],
        "low_month": MONTH_NAMES[int(month_idx.idxmin()) - 1],
        "elasticity": price_elasticity(g.reset_index()),
    }


def price_elasticity(g: pd.DataFrame) -> float:
    """Own-price elasticity from log-log regression, controlling for promotion, weekday, month and trend."""
    g = g[g["units_sold"] > 0]
    if len(g) < 60 or g["price"].nunique() < 4:
        return np.nan
    dates = pd.DatetimeIndex(g["date"])
    promo = g["promo"].to_numpy(float) if "promo" in g else np.zeros(len(g))
    X = [np.log(g["price"].to_numpy()), promo, np.arange(len(g)) / 365.0, np.ones(len(g))]
    X += [(dates.day_of_week == d).astype(float) for d in range(1, 7)]
    X += [(dates.month == m).astype(float) for m in range(2, 13)]
    X = np.column_stack(X)
    coef, *_ = np.linalg.lstsq(X, np.log(g["units_sold"].to_numpy()), rcond=None)
    return float(coef[0])


def decompose(g: pd.DataFrame) -> pd.DataFrame:
    """Weekly series split into trend (13-week centred average), seasonality and residual."""
    w = g.set_index("date")["units_sold"].resample("W").sum()
    trend = w.rolling(13, center=True, min_periods=4).mean()
    detr = w / trend
    seas = detr.groupby(detr.index.isocalendar().week.to_numpy()).transform("mean")
    seas = seas / seas.mean()
    return pd.DataFrame({"actual": w, "trend": trend, "seasonal": seas, "residual": w / (trend * seas)})


def abc_classes(sales: pd.DataFrame) -> pd.DataFrame:
    """ABC (revenue share) x XYZ (weekly demand variability) portfolio segmentation."""
    s = sales.assign(revenue=sales["units_sold"] * sales["price"])
    by = s.groupby(["product_id", "product_name", "category"]).agg(units=("units_sold", "sum"), revenue=("revenue", "sum"))
    by = by.sort_values("revenue", ascending=False)
    share = by["revenue"].cumsum() / by["revenue"].sum()
    by["ABC"] = np.where(share <= 0.7, "A", np.where(share <= 0.9, "B", "C"))
    weekly = s.set_index("date").groupby("product_id")["units_sold"].resample("W").sum()
    cv = weekly.groupby(level=0).agg(lambda x: x.std() / x.mean() if x.mean() else np.nan)
    by = by.reset_index()
    by["weekly_cv"] = by["product_id"].map(cv)
    by["XYZ"] = np.where(by["weekly_cv"] <= 0.25, "X", np.where(by["weekly_cv"] <= 0.5, "Y", "Z"))
    return by


def elasticities(sales: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """Per-product price elasticity shrunk toward its category median, plus the category medians.

    Elasticities are clipped to a plausible range so one noisy product can't produce absurd advice.
    """
    raw = {pid: price_elasticity(g) for pid, g in sales.groupby("product_id")}
    cat_of = sales.groupby("product_id")["category"].first()
    by_cat = pd.Series(raw).groupby(cat_of).median()
    overall = float(np.nanmedian(list(raw.values()))) if raw else -1.5
    cat_med = {c: float(np.clip(v if not np.isnan(v) else overall, -4, -0.3)) for c, v in by_cat.items()}
    prod = {}
    for pid, e in raw.items():
        prior = cat_med.get(cat_of[pid], overall)
        prod[pid] = float(np.clip(prior if np.isnan(e) else 0.7 * e + 0.3 * prior, -4, -0.3))
    return prod, cat_med


# ----------------------------------------------------------------- scenarios
def price_sensitivity(forecast_at: Callable[[float], ForecastResult], base_price: float, unit_cost: float,
                      factors: np.ndarray | None = None) -> pd.DataFrame:
    """Re-forecast with the regular price scaled up/down; returns demand, revenue and profit per price."""
    factors = np.linspace(0.7, 1.3, 13) if factors is None else factors
    rows = []
    for f in factors:
        res = forecast_at(base_price * f)
        units = float(res.mean.sum())
        revenue = float((res.mean * res.price).sum())
        rows.append({"price": base_price * f, "price_change_%": (f - 1) * 100, "units": units, "revenue": revenue,
                     "profit": revenue - unit_cost * units})
    return pd.DataFrame(rows)


def inventory_plan(res: ForecastResult, current_stock: float, lead_time: int, review_days: int,
                   service_level: float) -> dict:
    """Safety stock, reorder point, stock-out date and suggested order quantity.

    Order-up-to policy: when projected stock falls to the reorder point (lead-time demand
    plus safety stock), order enough to cover lead time + review period at the target
    service level.
    """
    z = NormalDist().inv_cdf(service_level)
    n = len(res.mean)
    demand_lt, _, hi_lt = res.total_interval(lead_time, z=1.0)
    safety = z * (hi_lt - demand_lt)
    cum = np.concatenate([[0.0], np.cumsum(res.mean)])
    stock_at = current_stock - cum[:n]  # projected stock at the start of each day
    out_idx = int(np.argmax(cum[1:] > current_stock)) if (cum[1:] > current_stock).any() else None
    stockout = res.dates[out_idx] if out_idx is not None else None

    # Reorder point from each day onward (demand over the following lead time + safety stock).
    lt_ahead = cum[np.minimum(np.arange(n) + lead_time, n)] - cum[:n]
    hit = stock_at <= lt_ahead + safety
    reorder_idx = int(np.argmax(hit)) if hit.any() else None

    order_qty = 0.0
    if reorder_idx is not None:
        i = reorder_idx
        cover = min(lead_time + review_days, n - i)
        window = res.with_values(res.mean[i:], res.lower[i:], res.upper[i:], sl=slice(i, None))
        demand_cover, _, hi_cover = window.total_interval(cover, z=1.0)
        order_qty = max(0.0, demand_cover + z * (hi_cover - demand_cover) - max(stock_at[i], 0))
    return {
        "z": z,
        "lead_time_demand": demand_lt,
        "safety_stock": safety,
        "reorder_point": demand_lt + safety,
        "order_qty": order_qty,
        "days_of_cover": (out_idx if out_idx is not None else n),
        "stockout_date": stockout,
        "reorder_by": res.dates[reorder_idx] if reorder_idx is not None else None,
        "needs_order_now": reorder_idx == 0,
    }


# ------------------------------------------------------------------ insights
def _pct(x: float) -> str:
    return f"{x * 100:+.0f}%"


def build_insights(res: ForecastResult, unit_cost: float, recent_daily: float | None,
                   promo_effect: dict | None, inventory: dict, sensitivity: pd.DataFrame | None,
                   current_price: float, profile: dict | None, method: str) -> list[str]:
    """Plain-language findings from the numbers on the forecast page."""
    out = []
    total, lo_total, hi_total = res.total_interval()
    days = len(res.mean)
    out.append(f"**Expected demand:** about **{total:,.0f} units** over the next {days} days "
               f"({RANGE_LABEL} range {lo_total:,.0f}–{hi_total:,.0f}), i.e. {total / days:,.1f} units/day on average. "
               f"Baseline built from {method}.")
    if recent_daily and recent_daily > 0:
        change = total / days / recent_daily - 1
        direction = "higher" if change > 0 else "lower"
        out.append(f"**Versus recent sales:** forecast daily demand is {abs(change) * 100:.0f}% {direction} than the "
                   f"last 56 days ({recent_daily:,.1f}/day), driven by season, price and promotion plans.")

    frame = res.frame()
    weekly = frame.groupby(frame["date"].dt.day_of_week)["forecast"].mean()
    if len(weekly) == 7:
        best, worst = int(weekly.idxmax()), int(weekly.idxmin())
        out.append(f"**Weekly rhythm:** {DOW_NAMES[best]} is the strongest day "
                   f"({_pct(weekly[best] / weekly.mean() - 1)} vs average), {DOW_NAMES[worst]} the weakest "
                   f"({_pct(weekly[worst] / weekly.mean() - 1)}). Schedule replenishment before {DOW_NAMES[best]}.")
    monthly = frame.set_index("date")["forecast"].resample("MS").mean()
    if len(monthly) > 1:
        peak = monthly.idxmax()
        out.append(f"**Seasonality:** daily demand peaks in **{peak:%B %Y}** at {monthly.max():,.1f}/day "
                   f"({_pct(monthly.max() / monthly.mean() - 1)} vs the horizon average).")
    peak_day = frame.loc[frame["forecast"].idxmax()]
    out.append(f"**Peak day:** {peak_day['date']:%a %d %b %Y} with ~{peak_day['forecast']:,.0f} units "
               f"(up to {peak_day['upper']:,.0f}).")

    if promo_effect and promo_effect.get("promo_days", 0) > 0:
        out.append(f"**Promotion impact:** the planned {promo_effect['promo_days']} promo days add "
                   f"~{promo_effect['extra_units']:,.0f} units ({_pct(promo_effect['lift'])} on promo days) and change "
                   f"profit by {promo_effect['extra_profit']:+,.0f}.")
    if sensitivity is not None and len(sensitivity):
        best = sensitivity.loc[sensitivity["profit"].idxmax()]
        cur = sensitivity.iloc[(sensitivity["price"] - current_price).abs().argmin()]
        if abs(best["price"] - current_price) / current_price > 0.04 and best["profit"] > cur["profit"] * 1.02:
            out.append(f"**Pricing:** within ±30%, profit is highest near **{best['price']:,.2f}** "
                       f"({best['price_change_%']:+.0f}%), projected profit {best['profit']:,.0f} vs {cur['profit']:,.0f} now. "
                       f"Test gradually: the model is least certain far from historical prices.")
        else:
            out.append(f"**Pricing:** the current price ({current_price:,.2f}) is close to the profit-maximising point "
                       f"in the ±30% range tested.")
    if profile and not np.isnan(profile.get("elasticity", np.nan)):
        e = profile["elasticity"]
        kind = "highly price sensitive" if e < -1.5 else "moderately price sensitive" if e < -0.8 else "not very price sensitive"
        out.append(f"**Price elasticity (history):** {e:.2f} — a 10% price cut has historically lifted units by "
                   f"~{(0.9 ** e - 1) * 100:.0f}%; this product is {kind}.")

    inv = inventory
    if inv["stockout_date"] is not None:
        msg = (f"**Inventory:** current stock lasts about **{inv['days_of_cover']} days** (runs out around "
               f"{inv['stockout_date']:%d %b %Y}).")
        if inv["needs_order_now"]:
            msg += f" Stock is at/below the reorder point ({inv['reorder_point']:,.0f}) — **order now**"
        else:
            msg += f" Place the next order by **{inv['reorder_by']:%d %b %Y}**"
        out.append(msg + f"; suggested quantity ≈ {inv['order_qty']:,.0f} units "
                         f"(safety stock {inv['safety_stock']:,.0f}).")
    else:
        msg = f"**Inventory:** current stock covers the whole {days}-day horizon."
        if inv["reorder_by"] is not None:
            msg += (f" Stock reaches the reorder point around **{inv['reorder_by']:%d %b %Y}** — order "
                    f"≈ {inv['order_qty']:,.0f} units then.")
        out.append(msg)
    width = (hi_total - lo_total) / max(total, 1e-9)
    if width > 1.1:
        out.append("**Uncertainty is high** for this product (wide range) — prefer smaller, more frequent orders.")
    return out


def detect_anomalies(g: pd.DataFrame, threshold: float = 4.0) -> pd.DataFrame:
    """Unusual days: robust z-score of sales against a centred 4-week median, adjusted for weekday.

    Promotion days are excluded because their spikes are explained.
    """
    g = g.sort_values("date").reset_index(drop=True)
    u = g["units_sold"].astype(float)
    dow = pd.DatetimeIndex(g["date"]).day_of_week
    dow_idx = u.groupby(dow).transform("mean") / max(u.mean(), 1e-9)
    adj = u / dow_idx.replace(0, np.nan)
    med = adj.rolling(29, center=True, min_periods=10).median()
    mad = (adj - med).abs().rolling(29, center=True, min_periods=10).median()
    scale = 1.4826 * mad + np.sqrt(med.clip(lower=1))  # Poisson floor so quiet products aren't over-flagged
    z = (adj - med) / scale
    expected = med * dow_idx
    out = g.assign(expected=expected, z=z)
    mask = (z.abs() > threshold) & (g.get("promo", 0) == 0)
    out = out[mask].copy()
    out["type"] = np.where(out["z"] > 0, "Spike", "Drop")
    return out[["date", "units_sold", "expected", "z", "type", "price"]]


def heatmap_month_dow(g: pd.DataFrame) -> pd.DataFrame:
    """Average units by month (rows) and weekday (columns)."""
    d = pd.DatetimeIndex(g["date"])
    t = g.assign(month=d.month, dow=d.day_of_week).pivot_table(index="month", columns="dow", values="units_sold",
                                                               aggfunc="mean")
    t.index = [MONTH_NAMES[m - 1] for m in t.index]
    t.columns = [DOW_NAMES[c] for c in t.columns]
    return t



def promo_spillover(sales: pd.DataFrame, pid: str, min_days: int = 15, min_effect: float = 0.03) -> pd.DataFrame:
    """How other products in the same category sell while `pid` is on promotion (cannibalization or halo).

    For each sibling, compares its sales on days `pid` was promoted (and the sibling was not) with its own
    typical level around those days. Effects smaller than `min_effect` or seen on fewer than `min_days` days
    are reported as 0 (not distinguishable from noise).
    """
    me = sales[sales["product_id"] == pid].set_index("date")
    if me.empty:
        return pd.DataFrame(columns=["product_id", "product_name", "effect", "days", "daily_level", "price"])
    cat = me["category"].iloc[0]
    my_promo = me["promo"] == 1
    rows = []
    for oid, g in sales[(sales["category"] == cat) & (sales["product_id"] != pid)].groupby("product_id"):
        g = g.set_index("date").sort_index()
        normal = g["units_sold"].where(g["promo"] == 0)
        typical = normal.rolling(29, center=True, min_periods=10).mean()
        days = my_promo.reindex(g.index, fill_value=False) & (g["promo"] == 0) & (typical > 0)
        n = int(days.sum())
        eff = float((g.loc[days, "units_sold"] / typical[days]).mean() - 1) if n else 0.0
        if n < min_days or abs(eff) < min_effect or not np.isfinite(eff):
            eff = 0.0
        rows.append({"product_id": oid, "product_name": g["product_name"].iloc[0], "effect": eff, "days": n,
                     "daily_level": float(g["units_sold"].tail(56).mean()), "price": float(g["price"].tail(28).mean())})
    return pd.DataFrame(rows)
