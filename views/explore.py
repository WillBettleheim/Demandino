import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import ui
from demandino.analysis import product_profile
from demandino.components import findings, show_patterns
from demandino.state import app

A = app()

ui.page_header("Explore sales patterns",
               "See how a product sells over time: trend, seasonality, weekday rhythm, promotions, price "
               "sensitivity and unusual days.")

c1, c2 = st.columns([2, 1])
pid = c1.selectbox("Product", list(A.label_of), format_func=A.label_of.get, key="ex_pid")
full = A.history(pid)
span = c2.segmented_control("Period", ["6M", "1Y", "2Y", "All"], default="All", required=True, key="ex_span")
days = {"6M": 182, "1Y": 365, "2Y": 730, "All": None}[span]
g = full if days is None else full[full["date"] > A.last_date - pd.Timedelta(days=days)]
prof = product_profile(g)
name = g["product_name"].iloc[0]

lines = [f"Sells **{prof['avg_daily']:,.1f} units/day** on average ({prof['total_units']:,.0f} units, revenue "
         f"{ui.money(prof['revenue'])} over {prof['days']} days).",
         f"Strongest weekday **{prof['best_dow']}**, weakest **{prof['worst_dow']}**; seasonal peak in "
         f"**{prof['peak_month']}**, low in **{prof['low_month']}**."]
if not np.isnan(prof["growth_90_vs_prev"]):
    trend = "growing" if prof["growth_90_vs_prev"] > 0.05 else "declining" if prof["growth_90_vs_prev"] < -0.05 else "flat"
    lines.append(f"Recent momentum is **{trend}** ({prof['growth_90_vs_prev'] * 100:+.0f}% last 90 days vs the 90 "
                 f"before; part of this can be seasonal).")
if not np.isnan(prof["promo_lift"]):
    lines.append(f"Promotions ran on {prof['promo_share'] * 100:.0f}% of days and lifted sales by "
                 f"**{prof['promo_lift'] * 100:+.0f}%** on average.")
if not np.isnan(prof["elasticity"]):
    lines.append(f"Price elasticity **{prof['elasticity']:.2f}**: a 10% price cut has lifted units by about "
                 f"{(0.9 ** prof['elasticity'] - 1) * 100:.0f}%.")
if not np.isnan(prof["cv"]):
    vol = "very volatile" if prof["cv"] > 0.8 else "moderately volatile" if prof["cv"] > 0.4 else "stable"
    lines.append(f"Daily demand is **{vol}** (coefficient of variation {prof['cv']:.2f}).")
findings(lines, f"What the history says about {name}")

show_patterns(g, prof, name)

st.divider()
st.markdown("#### Compare products")
others = st.multiselect("Products to compare", list(A.label_of), default=[pid], format_func=A.label_of.get,
                        max_selections=6, key="ex_cmp")
if others:
    norm = st.toggle("Index to 100 (compare shapes, not volumes)", value=len(others) > 1, key="ex_norm")
    fig = go.Figure()
    for p in others:
        h = A.history(p).set_index("date")["units_sold"].resample("W").sum()
        if days:
            h = h[h.index > A.last_date - pd.Timedelta(days=days)]
        y = h / h.mean() * 100 if norm else h
        fig.add_scatter(x=y.index, y=y.rolling(4, min_periods=1).mean(), name=A.label_of[p], mode="lines")
    ui.style(fig, "Weekly units (4-week average)" + (" · indexed, average = 100" if norm else ""), height=380)
    ui.show(fig)
