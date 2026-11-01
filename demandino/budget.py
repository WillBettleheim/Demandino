"""Monthly buying budget (open-to-buy): how much stock each category needs to buy each month to meet the
forecast, after using up the stock already on hand, compared with the budget you set.

Per product, current stock covers the first forecast sales; every unit beyond that must be bought, valued
at unit cost, in the month it's needed. Months over budget are flagged, as are months with room to spare.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COST_RATIO = 0.55


def needs(fc, stock: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    """Per month × category: forecast units, cost of goods sold, and money needed for new stock."""
    start = fc.sales["date"].max() + pd.Timedelta(days=1)
    end = (start + pd.DateOffset(months=months)).replace(day=1) - pd.Timedelta(days=1)
    horizon = (end - start).days + 1
    per_product = stock.groupby("product_id").agg(stock=("stock", "sum"), unit_cost=("unit_cost", "mean"))
    rows = []
    for pid, g in fc.sales.groupby("product_id"):
        res = fc.forecast_product(pid, horizon)
        price = float(g["regular_price"].iloc[-1])
        on_hand = float(per_product["stock"].get(pid, 0.0))
        cost = per_product["unit_cost"].get(pid, np.nan)
        cost = float(cost) if pd.notna(cost) and cost > 0 else price * DEFAULT_COST_RATIO
        sold = np.cumsum(res.mean)
        to_buy = np.diff(np.concatenate([[0.0], np.maximum(sold - on_hand, 0.0)]))  # units beyond stock, per day
        rows.append(pd.DataFrame({"date": res.dates, "category": g["category"].iloc[0], "units": res.mean,
                                  "cogs": res.mean * cost, "buy": to_buy * cost}))
    d = pd.concat(rows)
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    return d.groupby(["month", "category"], as_index=False)[["units", "cogs", "buy"]].sum()


def against_budget(n: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Per month: money needed vs a total monthly budget, and what's left (negative = over budget)."""
    m = n.groupby("month", as_index=False)[["units", "cogs", "buy"]].sum()
    m["budget"] = float(budget)
    m["left"] = m["budget"] - m["buy"]
    m["status"] = np.where(m["left"] < 0, "Over budget", np.where(m["left"] > 0.25 * m["budget"], "Room to spare",
                                                                  "On budget"))
    return m


def summary(m: pd.DataFrame, currency: str = "") -> list[str]:
    over = m[m["status"] == "Over budget"]
    lines = [f"**Total to buy:** {currency}{m['buy'].sum():,.0f} over {len(m)} months against "
             f"{currency}{m['budget'].sum():,.0f} of budget."]
    if len(over):
        worst = over.sort_values("left").iloc[0]
        lines.append(f"**Over budget:** {len(over)} month(s); worst is {worst['month']:%B %Y}, short by "
                     f"{currency}{-worst['left']:,.0f}. Bring orders forward into earlier months with room, or "
                     "trim slow sellers.")
    else:
        lines.append("**Within budget:** every month's buying fits.")
    return lines
