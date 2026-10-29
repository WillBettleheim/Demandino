import html
import pickle

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import ui
from demandino.advisor import growth_plan
from demandino.scenario import ForecastInputs
from demandino.state import app

A = app()


@st.cache_data(show_spinner="Testing price, promotion and timing options with the model…", max_entries=32,
               hash_funcs={ForecastInputs: lambda i: pickle.dumps(i)})
def plan_for(data_key: str, inp: ForecastInputs, target: float | None) -> dict:
    return growth_plan(A.fc, A.sales, A.elasticities, A.level_uncertainty, inp, target)


ui.page_header("Grow demand",
               "Find out what would actually move sales for a product. Every number below is the model re-run with "
               "one change — price, promotion depth and frequency, timing, or staying in stock — so you can compare "
               "options before spending money.", eyebrow="Growth advisor")

with ui.card("adv-inputs"):
    last: ForecastInputs | None = st.session_state.get("last_inputs")
    sources = ["Pick a product"] + (["Product from the Forecast page"] if last is not None else [])
    src = st.segmented_control("Product", sources, default=sources[-1], required=True, key="adv_src")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    if src == "Product from the Forecast page":
        inp = last
        c1.markdown(f"**{inp.name}** · {inp.category}  \n{ui.money(inp.price, 2)} · cost {ui.money(inp.cost, 2)} · "
                    f"{inp.horizon} days from {inp.start:%d %b %Y}")
        target = c2.number_input("Sales goal (units, optional)", 0, 10_000_000, 0, 10, key="adv_goal_fc") or None
    else:
        pid = c1.selectbox("Product", list(A.label_of), format_func=A.label_of.get, key="adv_pid")
        h = A.history(pid)
        price = float(h["regular_price"].iloc[-1])
        cost = c2.number_input(f"Unit cost ({ui.currency()})", 0.0, price * 10, round(price * 0.55, 2), 0.5,
                               key=f"adv_cost_{pid}", help="Needed to judge profit. Defaults to 55% of the price.")
        horizon = c3.select_slider("Horizon (days)", [30, 60, 90, 120, 180], 90, key="adv_h")
        stock = c4.number_input("Units in stock", 0, 10_000_000, 500, 10, key=f"adv_stock_{pid}")
        target = st.number_input("Sales goal over the horizon (units, optional)", 0, 10_000_000, 0, 10,
                                 key=f"adv_goal_{pid}",
                                 help="We'll find the most profitable tested plan that reaches it.") or None
        inp = ForecastInputs(new=False, pid=pid, name=h["product_name"].iloc[0], category=h["category"].iloc[0],
                             price=price, cost=float(cost), horizon=int(horizon),
                             start=A.last_date + pd.Timedelta(days=1), stock=float(stock))

g = plan_for(A.data_key, inp, float(target) if target else None)
base, recs = g["base"], g["recommendations"]

measured = recs[recs["lever"] != "Availability"] if len(recs) else recs
best = measured.iloc[0] if len(measured) and measured.iloc[0]["profit_delta"] > 0 else None
if best is not None:
    ui.spotlight(f"Best measured opportunity · {inp.name}", "+" + ui.money(best["profit_delta"]),
                 f"extra profit over {inp.horizon} days",
                 f"<b>{html.escape(best['action'])}</b> → {best['units_delta']:+,.0f} units vs today's plan "
                 f"({ui.num(base.total)} units, {ui.money(base.profit)} profit).",
                 [(best["lever"], "acc"), (f"{len(recs)} options tested", ""),
                  ("Measured with the model", "good")],
                 count=(best["profit_delta"], f"adv_{inp.name}", 0, "+" + ui.currency()))
else:
    ui.spotlight(f"Current plan · {inp.name}", ui.num(base.total), f"units over {inp.horizon} days",
                 f"Your current price and plan are already near the best of the options tested "
                 f"(profit {ui.money(base.profit)}). Volume options below trade profit for units.",
                 [("Already well optimised", "good"), (f"{len(recs)} options tested", "")])

if g["target"]:
    t = g["target"]
    if t["reached"]:
        st.success(f"**Goal reachable:** {t['plan']} → {t['units']:,.0f} units, profit {ui.money(t['profit'])} "
                   f"({t['profit_delta']:+,.0f} vs today). This is the most profitable tested plan that hits the goal.",
                   icon=":material/flag:")
    else:
        st.warning(f"**Goal not reached with price and promotions alone.** The strongest tested plan ({t['plan']}) "
                   f"gives {t['units']:,.0f} units; you'd need about {t['extra_uplift_needed']:+.0%} more demand from "
                   "marketing, new channels or distribution.", icon=":material/flag:")

st.markdown("#### :material/trending_up: Ranked actions")
if len(recs):
    show = recs[["lever", "action", "units_delta", "profit_delta", "revenue_delta", "note"]]
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "lever": "Lever", "action": st.column_config.TextColumn("What to do", width="large"),
        "units_delta": st.column_config.NumberColumn("Extra units", format="%+,.0f"),
        "profit_delta": st.column_config.NumberColumn(f"Profit change ({ui.currency()})", format="%+,.0f"),
        "revenue_delta": st.column_config.NumberColumn(f"Revenue change ({ui.currency()})", format="%+,.0f"),
        "note": st.column_config.TextColumn("Why", width="medium")})
    st.caption("All changes compare against your current price with no promotions. Availability shows forecast sales "
               "you would lose after a stock-out if nothing is reordered.")

c1, c2 = st.columns(2)
grid = g["promo_grid"]
if len(grid):
    pv = grid.pivot(index="discount", columns="every_weeks", values="profit_delta")
    fig = go.Figure(go.Heatmap(z=pv.values, x=[f"every {c} wks" for c in pv.columns],
                               y=[f"{d:.0%} off" for d in pv.index], colorscale="RdYlGn", zmid=0,
                               text=np.vectorize(lambda v: f"{v:+,.0f}")(pv.values), texttemplate="%{text}",
                               hovertemplate="%{y}, %{x}: %{z:+,.0f} profit<extra></extra>", colorbar=dict(thickness=10)))
    ui.style(fig, "Promotion plans · profit change", height=340, legend=False, hover="closest")
    ui.show(fig, c1)
price = g["price"]
if len(price):
    fig = go.Figure()
    fig.add_scatter(x=price["price"], y=price["units"], name="Units", line=dict(color=ui.HIST), line_shape="spline")
    ui.glow_line(fig, price["price"], price["profit"], ui.GOOD, "Profit", yaxis="y2")
    fig.add_vline(x=inp.price, line_dash="dot", line_color="rgba(20,184,166,.8)", annotation_text="now")
    ui.style(fig, "Price · units and profit", height=340,
             yaxis2=dict(title="Profit", overlaying="y", side="right", showgrid=False))
    fig.update_xaxes(title_text=f"Regular price ({ui.currency()})")
    ui.show(fig, c2)

timing = g["timing"]
if len(timing):
    fig = go.Figure(go.Bar(x=timing["week_start"], y=timing["units_delta"], marker_line_width=0,
                           marker_color=[ui.FC if v == timing["units_delta"].max() else ui.BRAND
                                         for v in timing["units_delta"]],
                           hovertemplate="Week of %{x|%d %b}: %{y:+,.0f} units<extra></extra>"))
    ui.style(fig, "Best timing · extra units from one 7-day promotion starting that week", height=300, legend=False,
             hover="closest")
    ui.show(fig)

spill = g.get("spillover")
if spill is not None and len(spill):
    with ui.card("spill"):
        st.markdown("##### :material/compare_arrows: Effect on other products in the category")
        st.caption("While this product was on promotion in the past, its siblings sold this much more (halo) or less "
                   "(cannibalization) than usual. Promotion numbers above are net of these effects. Effects under 3% "
                   "or seen on fewer than 15 days count as zero.")
        st.dataframe(spill[["product_name", "effect", "days", "daily_level"]], hide_index=True, width="stretch",
                     column_config={"product_name": "Product",
                                    "effect": st.column_config.NumberColumn("Effect on its sales", format="percent"),
                                    "days": "Promo days observed",
                                    "daily_level": st.column_config.NumberColumn("Its units/day", format="%.1f")})

if g["lost_past"] > 0:
    st.info(f"**Possible past stock-outs:** unusually low-sales days in the history add up to about "
            f"{g['lost_past']:,.0f} units below normal. If those were stock-outs, keeping stock available is the "
            "cheapest growth lever. See *Explore sales patterns → unusual days*.", icon=":material/inventory_2:")

with ui.card("good-practice"):
    st.markdown("##### :material/tips_and_updates: More ways to grow demand")
    st.caption("General good practice. These are not measured by the model because your sales data does not record "
               "them, so test them on a small scale first.")
    for title, text in g["good_practice"]:
        st.markdown(f"- **{title}:** {text}")
