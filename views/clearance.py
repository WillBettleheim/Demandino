import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import clearance, db, ui
from demandino.planning import default_stock
from demandino.security import safe_csv
from demandino.state import app

A = app()


@st.cache_data(show_spinner="Looking for slow stock…", max_entries=16)
def plan_for(data_key: str, stock: pd.DataFrame, window: int, target: int, max_disc: float, holding: float):
    return clearance.clearance_plan(A.fc, stock, A.elasticities[0], window, target, max_disc, holding)


ui.page_header("Clear slow stock",
               "Find products with more stock than they'll sell, and the smallest markdown that clears the excess "
               "in time — so cash isn't sitting on the shelf.", eyebrow="Forecast")

saved = db.get_setting(A.user_id, f"stock:{A.data_key}") if A.user_id else None
stock = st.session_state.get(f"pl_stock_{A.data_key}")
stock = stock if stock is not None else (pd.DataFrame(saved) if saved else default_stock(A.sales))
if saved is None and f"pl_stock_{A.data_key}" not in st.session_state:
    st.caption("Using illustrative stock levels. Enter your real stock on **Demand planning** (and save it) for "
               "advice about your shop.")

with st.container(border=True, key="card-clear-settings"):
    c1, c2, c3, c4 = st.columns(4)
    target = c1.number_input("Keep up to (days of stock)", 14, 365, 90, 7, key="cl_target",
                             help="Stock beyond this many days of forecast sales counts as excess.")
    window = c2.number_input("Clear within (days)", 14, 180, 60, 7, key="cl_window",
                             help="How long the markdown can run.")
    max_disc = c3.slider("Deepest discount", 10, 70, 50, 5, format="%d%%", key="cl_max") / 100
    holding = c4.number_input("Holding cost (% of cost / month)", 0.0, 10.0, 2.0, 0.5, key="cl_hold",
                              help="Storage, insurance and the cash tied up — typically 1.5–3% a month.") / 100

plan = plan_for(A.data_key, stock, int(window), int(target), float(max_disc), float(holding))
slow = plan[plan["slow"]]
ui.spotlight(f"Stock beyond {target} days of sales", ui.money(slow["excess_value"].sum()), "tied up in excess stock",
             f"<b>{len(slow)}</b> of {len(plan)} products are slow · suggested markdowns free about "
             f"<b>{ui.money(slow['cash_freed'].sum())}</b> and avoid <b>{ui.money(slow['holding_cost_avoided'].sum())}</b> "
             f"of holding cost",
             [("No slow stock", "good")] if slow.empty else
             [(f"{len(slow)} slow products", "warn"),
              (f"{int((slow['margin_per_unit'] < 0).sum())} would sell below cost", "bad")])

if slow.empty:
    st.success("Every product's stock is within your limit. Nothing to clear.", icon=":material/check_circle:")
else:
    c1, c2 = st.columns([1.15, 1])
    top = slow.head(12).iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=top["product_name"], x=top["excess_value"], orientation="h", name="Excess stock value",
                marker_color=ui.WARN, hovertemplate="%{y}: %{x:,.0f}<extra>excess</extra>")
    fig.add_bar(y=top["product_name"], x=top["cash_freed"], orientation="h", name="Cash freed by markdown",
                marker_color=ui.BRAND, hovertemplate="%{y}: %{x:,.0f}<extra>cash freed</extra>")
    ui.show(ui.style(fig, f"Excess stock vs cash freed ({ui.currency()})", height=380, hover="closest"), c1)
    cover = plan.assign(cover=np.minimum(plan["cover_days"], 720)).nlargest(12, "cover").iloc[::-1]
    fig = go.Figure(go.Bar(y=cover["product_name"], x=cover["cover"], orientation="h",
                           marker_color=np.where(cover["slow"], ui.WARN, ui.BRAND),
                           hovertemplate="%{y}: %{x:.0f} days of stock<extra></extra>"))
    fig.add_vline(x=target, line_dash="dot", annotation_text=f"your limit · {target} days")
    ui.show(ui.style(fig, "Days of stock · 12 longest", height=380, legend=False, hover="closest"), c2)

    st.subheader("Suggested markdowns")
    for r in slow.head(10).itertuples():
        with ui.card(f"clear-{r.product_id}"):
            a, b = st.columns([1.6, 1])
            a.markdown(f"**{ui.md_safe(r.product_name)}** · {ui.md_safe(r.category)}")
            a.caption(ui.md_safe(clearance.advice(r._asdict(), ui.currency())))
            b.html(f'<div class="dm-stat"><span>Suggested markdown</span><b>{r.discount:.0%} off → '
                   f'{html.escape(ui.money(r.new_price, 2))}</b><small>frees {html.escape(ui.money(r.cash_freed))} · '
                   f'saves {html.escape(ui.money(r.holding_cost_avoided))} holding</small></div>')

with st.expander(f"All products ({len(plan)})", icon=":material/table_rows:"):
    st.dataframe(plan[["product_name", "stock", "daily_forecast", "cover_days", "excess_units", "discount", "new_price",
                       "units_cleared", "cash_freed", "holding_cost_avoided", "elasticity"]],
                 hide_index=True, width="stretch", column_config={
                     "product_name": "Product", "stock": st.column_config.NumberColumn("Stock", format="%,.0f"),
                     "daily_forecast": st.column_config.NumberColumn("Sells / day", format="%.1f"),
                     "cover_days": st.column_config.NumberColumn("Days of stock", format="%.0f"),
                     "excess_units": st.column_config.NumberColumn("Excess units", format="%,.0f"),
                     "discount": st.column_config.NumberColumn("Markdown", format="percent"),
                     "new_price": st.column_config.NumberColumn("Sale price", format="%.2f"),
                     "units_cleared": st.column_config.NumberColumn("Extra units sold", format="%,.0f"),
                     "cash_freed": st.column_config.NumberColumn("Cash freed", format="%,.0f"),
                     "holding_cost_avoided": st.column_config.NumberColumn("Holding saved", format="%,.0f"),
                     "elasticity": st.column_config.NumberColumn("Price sensitivity", format="%.2f",
                                                                 help="−2 means a 10% cut sells ~20% more.")})
st.caption("Markdown effects are estimated from each product's past price changes. Test a markdown on the "
           "**Forecast a product** page (set the price) before committing to it.")
st.download_button("Download clearance plan (CSV)", safe_csv(plan), "clearance_plan.csv", "text/csv", icon=":material/download:",
                   key="dl_clearance_plan")
