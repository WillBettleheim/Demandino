"""Inbox: everything that needs attention, from every part of Demandino, in one list.

Each item has a priority (act now / this week / for your information), one plain sentence and the page
where it's handled. Nothing new is calculated here — it reads the same plan, alerts and checks the pages use.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from . import db, lifecycle, lostsales, targets
from .planning import default_stock, portfolio_plan

PRIORITY = {"now": 0, "week": 1, "info": 2}


def item(priority: str, title: str, text: str, page: str) -> dict:
    return {"priority": priority, "title": title, "text": text, "page": page}


def collect(plan: pd.DataFrame, drift: pd.DataFrame, stages: dict[str, str], lost: pd.DataFrame,
            targets: list[dict], currency: str = "") -> list[dict]:
    out = []
    now = plan[plan["status"].astype(str) == "Order now"]
    if len(now):
        out.append(item("now", "Order now", f"{len(now)} product(s) are at or below their reorder point: "
                        + ", ".join(now["product_name"].head(4)) + ".", "views/planning.py"))
    week = plan[plan["status"].astype(str) == "Order this week"]
    if len(week):
        out.append(item("week", "Order this week", f"{len(week)} product(s) reach their reorder point within 7 "
                        "days: " + ", ".join(week["product_name"].head(4)) + ".", "views/planning.py"))
    if "expiring_units" in plan:
        exp = plan[plan["expiring_units"] > 0.5]
        if len(exp):
            out.append(item("week", "Stock about to expire", f"About {exp['expiring_units'].sum():,.0f} units may "
                            "expire before they sell: " + ", ".join(exp["product_name"].head(3)) + ".",
                            "views/clearance.py"))
    fading = plan[plan["product_id"].map(stages).eq("End of life") & (plan["order_qty"] > 0)]
    if len(fading):
        out.append(item("week", "End-of-life products on order", "Still on the order list: "
                        + ", ".join(fading["product_name"].unique()[:3]) + ". Consider not reordering.",
                        "views/lifecycle.py"))
    if len(drift):
        flagged = drift[drift["flagged"]]
        for direction, word in (("Above", "faster"), ("Below", "slower")):
            g = flagged[flagged["direction"] == direction]
            if len(g):
                out.append(item("week", f"Selling {word} than forecast", ", ".join(g["product_name"].head(3))
                                + f" — {len(g)} product(s) outside the forecast range for weeks.", "views/alerts.py"))
    if len(lost):
        out.append(item("info", "Recent stock-outs", f"{lost['product_id'].nunique()} product(s) ran out in the last "
                        f"two weeks, losing about {currency}{lost['lost_sales'].sum():,.0f} of sales.",
                        "views/lostsales.py"))
    for t in targets:
        if t.get("chance", 1) < 0.35:
            out.append(item("info", "Target at risk", f"{t['label']} ({t['start']} → {t['end']}): only "
                            f"{t['chance']:.0%} chance of making it.", "views/targets.py"))
    return sorted(out, key=lambda i: PRIORITY[i["priority"]])


@st.cache_data(show_spinner="Gathering everything that needs attention…", max_entries=8)
def _plan(data_key: str, stock: pd.DataFrame, _fc, level_uncertainty: float) -> pd.DataFrame:
    return portfolio_plan(_fc, level_uncertainty, stock, 30, 14, 0.95)[0]


def gather(app, currency: str = "") -> list[dict]:
    """Everything needing attention for the active dataset (used by the Dashboard and Inbox pages)."""
    saved = db.get_setting(app.user_id, f"stock:{app.data_key}") if app.user_id else None
    stock = st.session_state.get(f"pl_stock_{app.data_key}")
    stock = stock if stock is not None else (pd.DataFrame(saved) if saved else default_stock(app.sales))
    plan = _plan(app.data_key, stock, app.fc, app.level_uncertainty)
    stages = lifecycle.stages(app.sales).set_index("product_id")["stage"].astype(str).to_dict()
    lost = lostsales.window(lostsales.daily(app.fc.sales), app.fc.sales["date"].max(), 14)
    saved_targets = []
    for t in (db.get_setting(app.user_id, f"targets:{app.base_key}", []) if app.user_id else []):
        p = targets.progress(app.fc, t["products"], pd.Timestamp(t["start"]), pd.Timestamp(t["end"]), t["metric"])
        saved_targets.append({**t, "chance": targets.chance(p, t["target"])})
    return collect(plan, app.drift()[0], stages, lost, saved_targets, currency)
