"""Sales lost to stock-outs, in money.

Demandino already flags likely stock-out days (runs of zero sales for products that normally sell, or days
your stock data says you had none) and estimates the demand on those days from the days around them.
Lost sales are that demand minus what sold, valued at the day's price; lost profit uses the unit cost.
Consecutive stock-out days are grouped into episodes, so repeat offenders stand out.
"""
from __future__ import annotations

import pandas as pd

DEFAULT_COST_RATIO = 0.55   # unit cost as a share of price when the real cost isn't known


def daily(sales: pd.DataFrame, unit_cost: dict[str, float] | None = None) -> pd.DataFrame:
    """One row per stock-out day: estimated demand, units sold, lost units, lost sales and lost profit."""
    if "stockout" not in sales or "demand" not in sales:
        return pd.DataFrame(columns=["date", "product_id", "product_name", "category", "demand", "sold", "lost_units",
                                     "lost_sales", "lost_profit"])
    so = sales[sales["stockout"].astype(bool)].copy()
    lost = (so["demand"] - so["units_sold"]).clip(lower=0)
    cost = so["product_id"].map(unit_cost or {}).fillna(so["price"] * DEFAULT_COST_RATIO)
    out = pd.DataFrame({"date": so["date"], "product_id": so["product_id"], "product_name": so["product_name"],
                        "category": so["category"], "demand": so["demand"], "sold": so["units_sold"],
                        "lost_units": lost, "lost_sales": lost * so["price"],
                        "lost_profit": lost * (so["price"] - cost).clip(lower=0)})
    return out[out["lost_units"] > 0].reset_index(drop=True)


def episodes(rows: pd.DataFrame) -> pd.DataFrame:
    """Runs of consecutive stock-out days per product, costliest first."""
    if rows.empty:
        return pd.DataFrame(columns=["product_id", "product_name", "start", "end", "days", "lost_units", "lost_sales"])
    r = rows.sort_values(["product_id", "date"])
    new_run = (r["product_id"] != r["product_id"].shift()) | (r["date"].diff() != pd.Timedelta(days=1))
    r = r.assign(run=new_run.cumsum())
    ep = r.groupby("run").agg(product_id=("product_id", "first"), product_name=("product_name", "first"),
                              start=("date", "min"), end=("date", "max"), days=("date", "size"),
                              lost_units=("lost_units", "sum"), lost_sales=("lost_sales", "sum"))
    return ep.sort_values("lost_sales", ascending=False).reset_index(drop=True)


def by_product(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["product_id", "product_name", "category", "days", "episodes", "lost_units",
                                     "lost_sales", "lost_profit"])
    ep = episodes(rows).groupby("product_id").size().rename("episodes")
    out = rows.groupby(["product_id", "product_name", "category"]).agg(
        days=("date", "size"), lost_units=("lost_units", "sum"), lost_sales=("lost_sales", "sum"),
        lost_profit=("lost_profit", "sum")).reset_index()
    out = out.merge(ep, left_on="product_id", right_index=True)
    return out.sort_values("lost_sales", ascending=False).reset_index(drop=True)


def by_month(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame(columns=["month", "category", "lost_sales"])
    m = rows.assign(month=rows["date"].dt.to_period("M").dt.to_timestamp())
    return m.groupby(["month", "category"], as_index=False)["lost_sales"].sum()


def ideas(prod: pd.DataFrame, total_sales: float, currency: str = "") -> list[str]:
    """What the losses say, in plain words."""
    if prod.empty:
        return ["**No stock-outs found:** nothing was lost to empty shelves in this period."]
    out = []
    lost = prod["lost_sales"].sum()
    top = prod.iloc[0]
    share = top["lost_sales"] / lost if lost else 0
    out.append(f"**Biggest loss:** {top['product_name']} — about {currency}{top['lost_sales']:,.0f} of sales over "
               f"{top['days']} stock-out days ({share:.0%} of all lost sales).")
    if total_sales > 0:
        out.append(f"**Overall:** stock-outs cost about {lost / (total_sales + lost):.1%} of potential sales.")
    repeat = prod[prod["episodes"] >= 3]
    if len(repeat):
        names = ", ".join(repeat["product_name"].head(3))
        out.append(f"**Keeps running out:** {names} ran out three or more times. Raise their safety stock (a higher "
                   "service level on Demand planning) or order earlier.")
    return out


def window(rows: pd.DataFrame, last: pd.Timestamp, days: int | None) -> pd.DataFrame:
    return rows if days is None else rows[rows["date"] > last - pd.Timedelta(days=days)]


def potential(sales: pd.DataFrame, last: pd.Timestamp, days: int | None) -> float:
    s = sales if days is None else sales[sales["date"] > last - pd.Timedelta(days=days)]
    return float((s["units_sold"] * s["price"]).sum())

