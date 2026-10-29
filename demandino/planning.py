"""Portfolio demand plan: forecast every product and flag what needs ordering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import inventory_plan
from .model import LEVEL_WINDOW, DemandForecaster

STATUS_ORDER = ["Order now", "Order this week", "Plan order", "Healthy"]


def default_stock(sales: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Illustrative starting stock (5–60 days of recent demand) so the plan has something to show."""
    rng = np.random.default_rng(seed)
    recent = sales.groupby("product_id").apply(lambda g: g.sort_values("date")["units_sold"].tail(LEVEL_WINDOW).mean(),
                                               include_groups=False)
    meta = sales.groupby("product_id")[["product_name", "category"]].first()
    days = rng.integers(5, 61, len(recent))
    return pd.DataFrame({"product_id": recent.index, "product_name": meta.loc[recent.index, "product_name"].to_numpy(),
                         "category": meta.loc[recent.index, "category"].to_numpy(),
                         "stock": np.round(recent.to_numpy() * days).astype(int), "lead_time": 14})


def portfolio_plan(fc: DemandForecaster, level_unc: float, stock: pd.DataFrame, horizon: int, review: int,
                   service: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns one row per product (forecast, risk, order advice) and the daily forecast in long format."""
    rows, daily = [], []
    today = fc.sales["date"].max() + pd.Timedelta(days=1)
    for r in stock.itertuples(index=False):
        res = fc.forecast_product(r.product_id, horizon)
        res.level_uncertainty = level_unc
        inv = inventory_plan(res, float(r.stock), int(r.lead_time), review, service)
        total, low, high = res.total_interval()
        hist = fc.sales[fc.sales["product_id"] == r.product_id]["units_sold"].tail(LEVEL_WINDOW)
        reorder = inv["reorder_by"]
        if inv["needs_order_now"]:
            status = "Order now"
        elif reorder is not None and (reorder - today).days <= 7:
            status = "Order this week"
        elif reorder is not None:
            status = "Plan order"
        else:
            status = "Healthy"
        weekly = pd.Series(res.mean, index=res.dates).resample("W").sum()
        rows.append({
            "status": status, "product_id": r.product_id, "product_name": r.product_name, "category": r.category,
            "recent_daily": float(hist.mean()), "forecast_daily": total / horizon, "forecast_units": total,
            "low": low, "high": high, "revenue": float((res.mean * res.price).sum()), "stock": float(r.stock),
            "days_of_cover": inv["days_of_cover"], "stockout_date": inv["stockout_date"],
            "reorder_date": reorder, "order_qty": float(np.ceil(inv["order_qty"])),
            "lead_time": int(r.lead_time), "safety_stock": inv["safety_stock"],
            "trend": weekly.round(1).tolist(),
        })
        daily.append(pd.DataFrame({"date": res.dates, "product_id": r.product_id, "category": r.category,
                                   "forecast": res.mean, "revenue": res.mean * res.price}))
    plan = pd.DataFrame(rows)
    plan["status"] = pd.Categorical(plan["status"], STATUS_ORDER, ordered=True)
    plan = plan.sort_values(["status", "days_of_cover"]).reset_index(drop=True)
    return plan, pd.concat(daily, ignore_index=True)
