"""Sales targets: how a period is going, combining sales so far with the forecast for the days left.

Projected total = actual sales to date + forecast for the remaining days. The chance of reaching the target
uses the forecast's own uncertainty: daily noise adds up independently, and the level error is shared within
a category (as on the Store & category totals page). Levers say what it would take to close a gap, from the
products' price elasticity and past promotion lift; they are estimates to test on the Forecast page.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

METRICS = {"units": "Units", "revenue": "Revenue"}


def default_period(last: pd.Timestamp, kind: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """The month or quarter that contains the first day to forecast."""
    day = pd.Timestamp(last) + pd.Timedelta(days=1)
    if kind == "Quarter":
        start = pd.Timestamp(year=day.year, month=3 * ((day.month - 1) // 3) + 1, day=1)
        return start, start + pd.offsets.QuarterEnd(0)
    start = day.replace(day=1)
    return start, start + pd.offsets.MonthEnd(0)


def _value(units: np.ndarray, price: np.ndarray, metric: str) -> np.ndarray:
    return units * price if metric == "revenue" else units


def progress(fc, products: list[str], start: pd.Timestamp, end: pd.Timestamp, metric: str = "units") -> dict:
    """Actual so far, forecast for the rest, projected total and its uncertainty, per product and in total."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    sales = fc.sales[fc.sales["product_id"].isin(products)]
    last = fc.sales["date"].max()
    done = sales[(sales["date"] >= start) & (sales["date"] <= min(end, last))]
    first_left = max(start, last + pd.Timedelta(days=1))
    days_left = max(0, (end - first_left).days + 1)
    horizon = max(0, (end - last).days)
    rows, daily = [], []
    for pid in products:
        g = done[done["product_id"] == pid]
        act = float(_value(g["units_sold"].to_numpy(float), g["price"].to_numpy(float), metric).sum())
        meta = sales[sales["product_id"] == pid].iloc[-1]
        rest, var_noise = 0.0, 0.0
        if horizon:
            res = fc.forecast_product(pid, horizon)
            keep = res.dates >= first_left
            mean = _value(res.mean[keep], res.price[keep], metric)
            sd = res.daily_sd[keep] * (res.price[keep] if metric == "revenue" else 1)
            rest, var_noise = float(mean.sum()), float((sd ** 2).sum())
            daily.append(pd.Series(mean, index=res.dates[keep], name=pid))
        rows.append({"product_id": pid, "product_name": meta["product_name"], "category": meta["category"],
                     "actual": act, "forecast_rest": rest, "projected": act + rest, "var_noise": var_noise})
    table = pd.DataFrame(rows)
    lu = fc.level_uncertainty
    var = table["var_noise"].sum() + sum((lu * g["forecast_rest"].sum()) ** 2 for _, g in table.groupby("category"))
    actual, rest = float(table["actual"].sum()), float(table["forecast_rest"].sum())
    return {"start": start, "end": end, "last": last, "first_left": first_left, "days_left": days_left,
            "days_done": max(0, (min(end, last) - start).days + 1), "metric": metric, "actual": actual,
            "forecast_rest": rest, "projected": actual + rest, "sd": float(np.sqrt(var)),
            "products": table.drop(columns="var_noise").sort_values("projected", ascending=False),
            "daily_forecast": pd.concat(daily, axis=1).sum(axis=1) if daily else pd.Series(dtype=float),
            "daily_actual": done.assign(v=_value(done["units_sold"].to_numpy(float), done["price"].to_numpy(float),
                                                 metric)).groupby("date")["v"].sum()}


def chance(p: dict, target: float) -> float:
    """Probability that the period's total reaches the target."""
    if p["days_left"] == 0 or p["sd"] <= 0:
        return float(p["projected"] >= target)
    return float(1 - NormalDist(p["projected"], p["sd"]).cdf(target))


def same_period_last_year(fc, products: list[str], start: pd.Timestamp, end: pd.Timestamp, metric: str) -> float:
    s = fc.sales[fc.sales["product_id"].isin(products)]
    ly = s[(s["date"] >= start - pd.DateOffset(years=1)) & (s["date"] <= end - pd.DateOffset(years=1))]
    return float(_value(ly["units_sold"].to_numpy(float), ly["price"].to_numpy(float), metric).sum())


def levers(p: dict, target: float, elasticity: dict[str, float], promo_lift: float | None,
           currency: str = "") -> list[str]:
    """Plain-language ways to close a gap (or a note that the target is covered)."""
    gap = target - p["projected"]

    def amount(x: float, decimals: int = 0, noun: bool = True) -> str:
        if p["metric"] == "units":
            return f"{x:,.{decimals}f} units"
        return f"{currency}{x:,.{decimals}f}" + (" in sales" if noun else "")

    if gap <= 0:
        return [f"The forecast already covers the target with about {amount(abs(gap))} to spare — "
                "protect it by keeping best-sellers in stock."]
    if p["days_left"] == 0 or p["forecast_rest"] <= 0:
        return ["The period is over or nothing is forecast for the days left, so the gap can't be closed now."]
    out = [f"You need about {amount(gap)} more: {amount(gap / p['days_left'], 1 if gap / p['days_left'] < 100 else 0, noun=False)}"
           f" extra a day for the {p['days_left']} days left, {gap / p['forecast_rest']:+.0%} on top of the forecast."]
    prods = p["products"]
    w = prods["forecast_rest"].to_numpy()
    e = np.array([elasticity.get(pid, -1.5) for pid in prods["product_id"]])
    e_avg = float(np.average(e, weights=w)) if w.sum() > 0 else -1.5
    need = 1 + gap / p["forecast_rest"]
    power = e_avg if p["metric"] == "units" else 1 + e_avg
    if power < -0.05:
        cut = 1 - need ** (1 / power)
        if 0 < cut <= 0.4:
            out.append(f"**Price:** a cut of about {cut:.0%} for the rest of the period would close it "
                       f"(price sensitivity {e_avg:.1f}); check it doesn't cost more profit than it brings.")
        else:
            out.append("**Price:** a price cut alone would need to be deeper than 40% — combine levers instead.")
    elif p["metric"] == "revenue":
        out.append(f"**Price:** demand here isn't price sensitive enough (sensitivity {e_avg:.1f}) for a cut to "
                   "raise revenue — a small price rise may help revenue instead.")
    if promo_lift and promo_lift > 0.02:
        share = (need - 1) / promo_lift
        if share <= 1:
            out.append(f"**Promotions:** promotions have lifted sales {promo_lift:+.0%}; running them on about "
                       f"{share:.0%} of the days left would close the gap (before the discount's cost).")
        else:
            out.append(f"**Promotions:** even promoting every day left (past lift {promo_lift:+.0%}) wouldn't "
                       "close the gap on its own.")
    top = prods.head(3)["product_name"].tolist()
    if len(prods) > 1 and top:
        out.append(f"**Focus:** {', '.join(top)} carry the most of the forecast — pushing them moves the total "
                   "fastest. Use Grow demand to test ideas for each.")
    return out


def promo_lift(sales: pd.DataFrame, products: list[str]) -> float | None:
    """Average past lift on promotion days for these products, weighted by their regular-day sales."""
    lifts, weights = [], []
    for pid, g in sales[sales["product_id"].isin(products)].groupby("product_id"):
        promo, regular = g.loc[g["promo"] == 1, "units_sold"], g.loc[g["promo"] == 0, "units_sold"]
        if len(promo) > 5 and regular.mean() > 0:
            lifts.append(promo.mean() / regular.mean() - 1)
            weights.append(regular.mean())
    return float(np.average(lifts, weights=weights)) if lifts else None
