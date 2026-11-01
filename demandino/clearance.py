"""Slow sellers and clearance pricing.

A product is slow when the stock on hand would take far longer to sell than you'd like at the forecast
pace. For each one, a markdown is chosen from the product's price elasticity: the smallest discount that
sells the excess within the clearance window, capped at a maximum discount. Cash freed and the holding
cost avoided are compared with doing nothing.

The demand response to a discount uses constant elasticity: units × (1 − discount)^elasticity
(elasticity is negative, so a cut raises units). It is an estimate; start with the suggestion and adjust.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DISCOUNTS = np.round(np.arange(0.05, 0.701, 0.05), 2)


def cover_days(stock: float, daily: float) -> float:
    return float("inf") if daily <= 1e-9 else stock / daily


def clearance_plan(fc, stock: pd.DataFrame, elasticity: dict[str, float], window: int = 60, target_days: int = 90,
                   max_discount: float = 0.5, holding_rate: float = 0.02) -> pd.DataFrame:
    """One row per product. `target_days`: stock beyond this many days of sales counts as excess.
    `holding_rate`: monthly cost of keeping a unit, as a share of its unit cost (storage, capital, risk)."""
    per_product = stock.groupby("product_id").agg(stock=("stock", "sum"), unit_cost=("unit_cost", "mean"),
                                                  product_name=("product_name", "first"),
                                                  category=("category", "first"))
    rows = []
    for pid, s in per_product.iterrows():
        if pid not in fc.sales["product_id"].values:
            continue
        res = fc.forecast_product(pid, max(window, target_days))
        daily = float(res.mean.mean())
        price = float(np.median(res.price))
        units_window = float(res.mean[:window].sum())
        cover = cover_days(s["stock"], daily)
        excess = max(0.0, s["stock"] - daily * target_days)
        e = float(elasticity.get(pid, -1.5))
        row = {"product_id": pid, "product_name": s["product_name"], "category": s["category"],
               "stock": float(s["stock"]), "daily_forecast": daily, "cover_days": cover, "excess_units": excess,
               "price": price, "unit_cost": float(s["unit_cost"]), "elasticity": e, "slow": excess > 0,
               "discount": 0.0, "new_price": price, "units_cleared": 0.0, "cash_freed": 0.0,
               "holding_cost_avoided": 0.0, "margin_per_unit": price - float(s["unit_cost"])}
        if excess > 0:
            # Extra units sold in the window at each discount, beyond what would sell anyway.
            extra = units_window * ((1 - DISCOUNTS) ** e - 1)
            ok = DISCOUNTS[(extra >= excess) & (DISCOUNTS <= max_discount + 1e-9)]
            d = float(ok[0]) if len(ok) else float(max_discount)
            cleared = float(min(excess, units_window * ((1 - d) ** e - 1)))
            new_price = price * (1 - d)
            # Months the excess would otherwise sit before selling at the normal pace.
            months_held = (excess / daily if daily > 1e-9 else 365.0) / 30.4 / 2  # average unit waits half of it
            row.update({"discount": d, "new_price": new_price, "units_cleared": cleared,
                        "cash_freed": cleared * new_price,
                        "holding_cost_avoided": cleared * s["unit_cost"] * holding_rate * min(months_held, 12),
                        "margin_per_unit": new_price - float(s["unit_cost"]),
                        "fully_cleared": cleared >= excess - 0.5})
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["stock_value"] = out["stock"] * out["unit_cost"]
    out["excess_value"] = out["excess_units"] * out["unit_cost"]
    return out.sort_values(["slow", "excess_value"], ascending=False).reset_index(drop=True)


def advice(row, currency: str = "") -> str:
    if not row["slow"]:
        return "Stock is in line with demand."
    days = "no sales expected" if np.isinf(row["cover_days"]) else f"{row['cover_days']:.0f} days of stock"
    tail = ("" if row.get("fully_cleared", True) else
            " Even at the maximum discount not all of it sells in time — consider bundling, returning it to the "
            "supplier or a marketplace sale.")
    loss = " (below cost — only worth it to free cash or space)" if row["margin_per_unit"] < 0 else ""
    return (f"{days}. Mark down {row['discount']:.0%} to about {currency}{row['new_price']:,.2f}{loss} to sell "
            f"~{row['units_cleared']:,.0f} extra units.{tail}")
