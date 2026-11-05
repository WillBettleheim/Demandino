"""Portfolio demand plan: forecast every product, apply real ordering rules and flag what to buy.

Ordering rules: minimum order quantity (MOQ), case packs, unit cost, a purchasing budget and several
warehouses (each holding stock for a share of a product's demand). Perishable products have a shelf life:
orders are capped so a delivery can sell before it expires, and stock likely to go to waste is flagged.
A cash-flow view compares money going out for orders with money coming in from forecast sales.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .analysis import inventory_plan
from .model import LEVEL_WINDOW, DemandForecaster

STATUS_ORDER = ["Order now", "Order this week", "Plan order", "Healthy", "Deferred (budget)"]
# Typical shelf lives (days) for the sample shop's perishable products; everything else doesn't expire.
ILLUSTRATIVE_SHELF_LIFE = {"Orange Juice 1L": 21, "Cold Brew Coffee": 45, "Iced Tea Lemon": 120,
                           "Energy Drink": 270, "Sparkling Water 6pk": 365}


def default_stock(sales: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    """Illustrative starting stock (5–60 days of recent demand, a few overstocked at 150–300 days) so the plan and
    the clearance advice have something to show."""
    rng = np.random.default_rng(seed)
    recent = sales.groupby("product_id").apply(lambda g: g.sort_values("date")["units_sold"].tail(LEVEL_WINDOW).mean(),
                                               include_groups=False)
    meta = sales.groupby("product_id")[["product_name", "category"]].first()
    price = sales.sort_values("date").groupby("product_id")["regular_price"].last()
    days = rng.integers(5, 61, len(recent))
    over = rng.random(len(recent)) < 0.15
    days[over] = rng.integers(150, 301, over.sum())
    return pd.DataFrame({
        "product_id": recent.index, "product_name": meta.loc[recent.index, "product_name"].to_numpy(),
        "category": meta.loc[recent.index, "category"].to_numpy(), "warehouse": "Main",
        "demand_share": 100.0, "stock": np.round(recent.to_numpy() * days).astype(int), "lead_time": 14,
        "moq": 0, "case_pack": 1, "unit_cost": np.round(price.loc[recent.index].to_numpy() * 0.55, 2),
        "shelf_life": [ILLUSTRATIVE_SHELF_LIFE.get(n, 0) for n in meta.loc[recent.index, "product_name"]],
        "expires_in": 0, "supplier": "Main supplier",
    })


def _demand(mean: np.ndarray, start: int, end: int) -> float:
    """Forecast units sold on days [start, end), continuing at the last four weeks' pace beyond the horizon."""
    start, end = max(0, start), max(0, end)
    inside = float(mean[start:min(end, len(mean))].sum())
    beyond = max(0, end - max(start, len(mean)))
    return inside + beyond * float(mean[-28:].mean() if len(mean) else 0.0)


def expiry_check(mean: np.ndarray, stock: float, shelf_life: float, expires_in: float | None, arrival: int | None,
                 order_qty: float) -> dict:
    """What expires before it sells (first in, first out), and the largest order that can sell in time.

    `shelf_life`: days a new delivery lasts (0 = doesn't expire). `expires_in`: days left on the current stock
    (default: a full shelf life). `arrival`: day the next order arrives (None = no order).
    """
    out = {"expiring_units": 0.0, "order_cap": None, "order_waste": 0.0}
    if not shelf_life or shelf_life <= 0:
        return out
    life = int(round(shelf_life))
    left = int(round(expires_in)) if expires_in and expires_in > 0 else life
    out["expiring_units"] = max(0.0, stock - _demand(mean, 0, left))
    if arrival is not None:
        old_left = max(0.0, stock - _demand(mean, 0, arrival) - out["expiring_units"]) if arrival < left else 0.0
        cap = max(0.0, _demand(mean, arrival, arrival + life) - old_left)
        out["order_cap"] = float(np.floor(cap))
        out["order_waste"] = max(0.0, order_qty - cap)
    return out


def apply_order_rules(qty: float, moq: float, case_pack: float) -> float:
    """Round an order up to whole case packs and lift it to the minimum order quantity."""
    if qty <= 0:
        return 0.0
    case = max(1.0, float(case_pack or 1))
    q = np.ceil(qty / case) * case
    if moq and q < moq:
        q = np.ceil(moq / case) * case
    return float(q)


def portfolio_plan(fc: DemandForecaster, level_unc: float, stock: pd.DataFrame, horizon: int, review: int,
                   service: float, budget: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per product × warehouse (forecast, risk, order advice) and the daily forecast (long format)."""
    stock = stock[stock["product_id"].astype(str).isin(fc.sales["product_id"].unique())]  # not removed products
    stock = stock.copy()
    for col, default in (("warehouse", "Main"), ("demand_share", 100.0), ("moq", 0), ("case_pack", 1),
                         ("unit_cost", 0.0), ("shelf_life", 0), ("expires_in", 0), ("supplier", "Main supplier")):
        if col not in stock:
            stock[col] = default
    stock["demand_share"] = pd.to_numeric(stock["demand_share"], errors="coerce").fillna(100.0).clip(0, 100)
    shares = stock.groupby("product_id")["demand_share"].transform(lambda s: s / s.sum() if s.sum() else 1 / len(s))
    rows, daily = [], []
    today = fc.sales["date"].max() + pd.Timedelta(days=1)
    forecasts = {}
    for pid in stock["product_id"].unique():
        res = fc.forecast_product(pid, horizon)
        res.level_uncertainty = level_unc
        forecasts[pid] = res
    for r, share in zip(stock.itertuples(index=False), shares):
        full = forecasts[r.product_id]
        res = full.with_values(full.mean * share, full.lower * share, full.upper * share)
        shelf = float(pd.to_numeric(r.shelf_life, errors="coerce") or 0)
        expires_in = float(pd.to_numeric(r.expires_in, errors="coerce") or 0)
        expiring = expiry_check(res.mean, float(r.stock), shelf, expires_in, None, 0.0)["expiring_units"]
        # Stock that will expire before it sells can't cover demand, so reorder timing ignores it.
        inv = inventory_plan(res, float(r.stock) - expiring, int(r.lead_time), review, service)
        total, low, high = res.total_interval()
        hist = fc.sales[fc.sales["product_id"] == r.product_id]["units_sold"].tail(LEVEL_WINDOW) * share
        reorder = inv["reorder_by"]
        if inv["needs_order_now"]:
            status = "Order now"
        elif reorder is not None and (reorder - today).days <= 7:
            status = "Order this week"
        elif reorder is not None:
            status = "Plan order"
        else:
            status = "Healthy"
        need = inv["order_qty"]
        arrival = (reorder - today).days + int(r.lead_time) if reorder is not None else None
        exp = expiry_check(res.mean, float(r.stock), shelf, expires_in, arrival, 0.0)
        if exp["order_cap"] is not None and need > exp["order_cap"]:
            need = exp["order_cap"]  # never order more than can sell before it expires
        qty = apply_order_rules(need, float(r.moq or 0), float(r.case_pack or 1)) if reorder is not None and need > 0 else 0.0
        if exp["order_cap"] is not None:
            exp["order_waste"] = max(0.0, qty - exp["order_cap"])  # e.g. a minimum order larger than can sell in time
        weekly = pd.Series(res.mean, index=res.dates).resample("W").sum()
        rows.append({
            "status": status, "product_id": r.product_id, "product_name": r.product_name, "category": r.category,
            "supplier": str(r.supplier or "Main supplier"),
            "warehouse": str(r.warehouse), "recent_daily": float(hist.mean()), "forecast_daily": total / horizon,
            "forecast_units": total, "low": low, "high": high, "revenue": float((res.mean * res.price).sum()),
            "stock": float(r.stock), "days_of_cover": inv["days_of_cover"], "stockout_date": inv["stockout_date"],
            "reorder_date": reorder, "suggested_qty": float(np.ceil(inv["order_qty"])), "order_qty": qty,
            "order_cost": qty * float(r.unit_cost or 0), "unit_cost": float(r.unit_cost or 0),
            "moq": float(r.moq or 0), "case_pack": float(r.case_pack or 1), "lead_time": int(r.lead_time),
            "safety_stock": inv["safety_stock"], "trend": weekly.round(1).tolist(),
            "shelf_life": shelf, "expiring_units": exp["expiring_units"],
            "expiring_value": exp["expiring_units"] * float(r.unit_cost or 0),
            "order_capped": exp["order_cap"] is not None and inv["order_qty"] > exp["order_cap"] + 0.5,
            "order_waste": exp["order_waste"],
        })
        daily.append(pd.DataFrame({"date": res.dates, "product_id": r.product_id, "category": r.category,
                                   "warehouse": str(r.warehouse), "forecast": res.mean, "revenue": res.mean * res.price}))
    plan = pd.DataFrame(rows)
    plan["status"] = pd.Categorical(plan["status"], STATUS_ORDER, ordered=True)
    plan = plan.sort_values(["status", "days_of_cover"]).reset_index(drop=True)
    if budget is not None and budget > 0:
        spent = 0.0
        for i, row in plan.iterrows():
            if row["order_qty"] <= 0:
                continue
            if spent + row["order_cost"] <= budget:
                spent += row["order_cost"]
            else:
                plan.loc[i, "status"] = "Deferred (budget)"
        plan = plan.sort_values(["status", "days_of_cover"]).reset_index(drop=True)
    plan.attrs["today"] = f"{today:%Y-%m-%d}"  # text, so tables carrying it can still be displayed
    return plan, pd.concat(daily, ignore_index=True)


def suggest_transfers(plan: pd.DataFrame, review: int, min_units: float = 1.0) -> pd.DataFrame:
    """Move stock from warehouses with more than they need to ones about to run short, before buying more.

    A warehouse can give what it holds beyond its own next cycle: forecast sales until its next delivery
    could arrive (lead time + days between orders) plus its safety stock. A warehouse takes up to what it
    would otherwise order now or this week. Largest moves first; each unit is only promised once.
    """
    cols = ["product_id", "product_name", "from_warehouse", "to_warehouse", "units", "order_before", "order_after",
            "money_saved"]
    if plan.empty or plan["warehouse"].nunique() < 2:
        return pd.DataFrame(columns=cols)
    rows = []
    for pid, g in plan.groupby("product_id"):
        if len(g) < 2:
            continue
        spare = {r.warehouse: max(0.0, r.stock - r.forecast_daily * (r.lead_time + review) - r.safety_stock)
                 for r in g.itertuples()}
        needy = g[g["status"].astype(str).isin(["Order now", "Order this week"]) & (g["order_qty"] > 0)]
        for r in needy.sort_values("days_of_cover").itertuples():
            want = float(r.order_qty)
            for donor in sorted((w for w in spare if w != r.warehouse), key=lambda w: -spare[w]):
                units = float(np.floor(min(want, spare[donor])))
                if units < min_units:
                    continue
                spare[donor] -= units
                rows.append({"product_id": pid, "product_name": r.product_name, "from_warehouse": donor,
                             "to_warehouse": r.warehouse, "units": units, "order_before": float(r.order_qty),
                             "order_after": max(0.0, want - units), "money_saved": units * float(r.unit_cost)})
                want -= units
                if want < min_units:
                    break
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values("money_saved", ascending=False).reset_index(drop=True)


def cash_flow(plan: pd.DataFrame, daily: pd.DataFrame, cost_ratio: pd.Series | None = None) -> pd.DataFrame:
    """Weekly money out (orders paid when placed) vs money in (forecast sales revenue)."""
    orders = plan[(plan["order_qty"] > 0) & (plan["status"] != "Deferred (budget)")]
    start = daily["date"].min()
    out = pd.Series(orders["order_cost"].to_numpy(),
                    index=pd.to_datetime(orders["reorder_date"]).fillna(start).clip(lower=start)).groupby(level=0).sum()
    inflow = daily.groupby("date")["revenue"].sum()
    idx = pd.date_range(start, daily["date"].max(), freq="D")
    df = pd.DataFrame({"money_in": inflow.reindex(idx, fill_value=0.0), "money_out": out.reindex(idx, fill_value=0.0)})
    weekly = df.resample("W").sum()
    weekly["net"] = weekly["money_in"] - weekly["money_out"]
    weekly["cumulative"] = weekly["net"].cumsum()
    return weekly
