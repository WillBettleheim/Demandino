"""Store-wide what-if: one change applied to many products at once, and its effect on the whole shop.

Each product in scope is forecast twice with the same engine as the Forecast page — as things stand, and
with the change — so price changes use each product's price sensitivity and promotions use what the model
learned about promotion days. Results add up by product, category and in total.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .scenario import ForecastInputs, compute

DEFAULT_COST_RATIO = 0.55


@dataclass(frozen=True)
class Change:
    price_change: float = 0.0                    # e.g. 0.05 = all prices +5%
    discount: float = 0.0                        # promotion discount on promo days, e.g. 0.2
    promo_start: pd.Timestamp | None = None
    promo_end: pd.Timestamp | None = None

    def describe(self) -> str:
        parts = []
        if self.price_change:
            parts.append(f"prices {self.price_change:+.0%}")
        if self.discount and self.promo_start is not None:
            parts.append(f"{self.discount:.0%} off {self.promo_start:%d %b}–{self.promo_end:%d %b}")
        return ", ".join(parts) or "no change"


def _inputs(fc, pid: str, start: pd.Timestamp, horizon: int, price: float, cost: float, change: Change | None
            ) -> ForecastInputs:
    g = fc.sales[fc.sales["product_id"] == pid]
    base = ForecastInputs(False, pid, str(g["product_name"].iloc[0]), str(g["category"].iloc[0]), price, cost,
                          horizon, start)
    if change is None:
        return base
    promo = change.discount > 0 and change.promo_start is not None
    return ForecastInputs(
        False, pid, base.name, base.category, price * (1 + change.price_change), cost, horizon, start,
        promo_kind="Specific dates" if promo else "None", discount=change.discount,
        promo_dates=(change.promo_start.date(), change.promo_end.date()) if promo else None)


def run(fc, elasticities: tuple[dict, dict], level_unc: float, products: list[str], change: Change, horizon: int,
        unit_cost: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per product: units, revenue and profit as things stand and with the change."""
    start = fc.sales["date"].max() + pd.Timedelta(days=1)
    rows = []
    for pid in products:
        g = fc.sales[fc.sales["product_id"] == pid]
        price = float(g["regular_price"].iloc[-1])
        cost = float((unit_cost or {}).get(pid, price * DEFAULT_COST_RATIO))
        before = compute(fc, fc.sales, elasticities, level_unc, _inputs(fc, pid, start, horizon, price, cost, None),
                         detail=False)
        after = compute(fc, fc.sales, elasticities, level_unc, _inputs(fc, pid, start, horizon, price, cost, change),
                        detail=False)
        rows.append({"product_id": pid, "product_name": before.inputs.name, "category": before.inputs.category,
                     "units_before": before.total, "units_after": after.total,
                     "revenue_before": before.revenue, "revenue_after": after.revenue,
                     "profit_before": before.profit, "profit_after": after.profit})
    out = pd.DataFrame(rows)
    for m in ("units", "revenue", "profit"):
        out[f"{m}_change"] = out[f"{m}_after"] - out[f"{m}_before"]
    return out.sort_values("profit_change").reset_index(drop=True)


def totals(df: pd.DataFrame) -> dict:
    t = {k: float(df[k].sum()) for k in df.columns if k.endswith(("_before", "_after", "_change"))}
    for m in ("units", "revenue", "profit"):
        t[f"{m}_pct"] = t[f"{m}_change"] / t[f"{m}_before"] if t[f"{m}_before"] else np.nan
    return t


def by_category(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.endswith(("_before", "_after", "_change"))]
    return df.groupby("category", as_index=False)[cols].sum().sort_values("profit_change", ascending=False)


def verdict(t: dict, df: pd.DataFrame, currency: str = "") -> list[str]:
    """Plain-language read-out of the result."""
    def money(x: float) -> str:
        return f"{'-' if x < 0 else '+'}{currency}{abs(x):,.0f}"

    out = [f"**Overall:** units {t['units_pct']:+.1%}, revenue {t['revenue_pct']:+.1%}, profit {t['profit_pct']:+.1%} "
           f"({money(t['profit_change'])})."]
    losers = df[df["profit_change"] < 0]
    winners = df[df["profit_change"] > 0]
    if t["profit_change"] > 0 and len(losers):
        out.append(f"**Mostly worth it:** profit rises overall, but {len(losers)} product(s) lose — e.g. "
                   f"{losers.iloc[0]['product_name']} ({money(losers.iloc[0]['profit_change'])}). Consider "
                   "leaving those out.")
    elif t["profit_change"] < 0 and len(winners):
        best = winners.sort_values("profit_change", ascending=False).iloc[0]
        out.append(f"**Not worth it across the board:** profit falls overall, but {len(winners)} product(s) gain — "
                   f"e.g. {best['product_name']} ({money(best['profit_change'])}). Try it on those only.")
    elif t["profit_change"] < 0:
        out.append("**Not worth it:** every product makes less profit with this change.")
    if len(winners) and len(losers):
        out.append(f"**Best version:** only on the {len(winners)} product(s) that gain, profit would change by "
                   f"{money(winners['profit_change'].sum())}.")
    return out
