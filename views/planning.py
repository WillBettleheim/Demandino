import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import db, ui
from demandino.planning import STATUS_ORDER, cash_flow, default_stock, portfolio_plan
from demandino.security import safe_csv
from demandino.state import app

A = app()


@st.cache_data(show_spinner="Forecasting every product…", max_entries=16)
def plan_for(data_key: str, stock: pd.DataFrame, horizon: int, review: int, service: float, budget: float | None):
    return portfolio_plan(A.fc, A.level_uncertainty, stock, horizon, review, service, budget)


ui.page_header("Demand planning",
               "Forecast every product at once, see which ones will run out, and get a ready-made purchase list.")

with ui.card("plan-controls"):
    c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1, 1.1, 1.3])
    horizon = c1.segmented_control("Planning horizon", [30, 60, 90], default=60, required=True,
                                   format_func=lambda d: f"{d} days", key="pl_h")
    review = c2.number_input("Days between orders", 1, 60, 14, key="pl_review")
    service = c3.number_input("Service level %", 80.0, 99.9, 95.0, 0.5, key="pl_service") / 100
    default_lead = c4.number_input("Default lead time (days)", 1, 180, 14, key="pl_lead")
    budget = c5.number_input(f"Purchase budget ({ui.currency()})", 0, 1_000_000_000, 0, 1000, key="pl_budget",
                             help="0 = no limit. With a budget, the most urgent orders are kept and the rest are "
                                  "marked Deferred.") or None

stock_key = f"pl_stock_{A.data_key}"
uid = A.user_id
if stock_key not in st.session_state:
    saved = db.get_setting(uid, f"stock:{A.data_key}") if uid else None
    st.session_state[stock_key] = pd.DataFrame(saved) if saved else default_stock(A.sales)
base = st.session_state[stock_key].copy()
if st.session_state.get("pl_lead_applied") != default_lead:
    base["lead_time"] = default_lead
    st.session_state[stock_key] = base
    st.session_state["pl_lead_applied"] = default_lead

with st.expander("Stock, warehouses & ordering rules — edit to match your business", icon=":material/edit_note:"):
    st.caption("Starting values are illustrative. Add a row per warehouse for products stocked in several places "
               "(their demand shares are split proportionally). MOQ = minimum order quantity; orders are rounded up to "
               "whole case packs.")
    edited = st.data_editor(base, key=f"pl_editor_{A.data_key}", hide_index=True, width="stretch", num_rows="dynamic",
                            disabled=["product_name", "category"],
                            column_config={
                                "product_id": st.column_config.SelectboxColumn("Product", options=list(A.label_of),
                                                                               required=True),
                                "product_name": "Name", "category": "Category",
                                "warehouse": st.column_config.TextColumn("Warehouse", max_chars=40, default="Main"),
                                "demand_share": st.column_config.NumberColumn("Demand share %", min_value=0,
                                                                              max_value=100, default=100),
                                "stock": st.column_config.NumberColumn("Stock on hand", min_value=0, step=1),
                                "lead_time": st.column_config.NumberColumn("Lead time (days)", min_value=1,
                                                                           max_value=180, step=1, default=14),
                                "moq": st.column_config.NumberColumn("MOQ", min_value=0, step=1, default=0),
                                "case_pack": st.column_config.NumberColumn("Case pack", min_value=1, step=1, default=1),
                                "unit_cost": st.column_config.NumberColumn(f"Unit cost ({ui.currency()})",
                                                                           min_value=0.0, format="%.2f")})
    names = A.products.set_index("product_id")
    edited = edited.dropna(subset=["product_id"]).copy()
    edited["product_name"] = edited["product_id"].map(names["product_name"])
    edited["category"] = edited["product_id"].map(names["category"])
    edited = edited.fillna({"warehouse": "Main", "demand_share": 100, "stock": 0, "lead_time": default_lead, "moq": 0,
                            "case_pack": 1, "unit_cost": 0.0})
    b1, b2 = st.columns(2)
    if b1.button("Reset to illustrative values", icon=":material/restart_alt:"):
        st.session_state[stock_key] = default_stock(A.sales)
        st.session_state.pop(f"pl_editor_{A.data_key}", None)
        st.rerun()
    if uid and b2.button("Save stock table to my workspace", icon=":material/save:"):
        db.set_setting(uid, f"stock:{A.data_key}", edited.to_dict(orient="list"))
        st.toast("Stock table saved.", icon=":material/check:")

if edited.empty:
    st.info("Add at least one product row to plan.", icon=":material/add:")
    st.stop()
plan, daily = plan_for(A.data_key, edited, int(horizon), int(review), float(service), float(budget) if budget else None)

counts = plan["status"].value_counts().reindex(STATUS_ORDER, fill_value=0)
with st.container(horizontal=True):
    st.metric("Forecast units", ui.num(plan["forecast_units"].sum()), border=True,
              chart_data=daily.groupby(pd.Grouper(key="date", freq="W"))["forecast"].sum().round().tolist(),
              chart_type="bar", help=f"All products, next {horizon} days.")
    st.metric("Forecast revenue", ui.money(plan["revenue"].sum()), border=True)
    st.metric("Order now", int(counts["Order now"]), border=True, delta="below reorder point" if counts["Order now"] else None,
              delta_color="inverse", delta_arrow="off")
    st.metric("Order this week", int(counts["Order this week"]), border=True)
    st.metric("Stock-outs in horizon", int(plan["stockout_date"].notna().sum()), border=True,
              help="Products whose current stock runs out before the horizon ends if nothing is ordered.")
    spend = plan.loc[plan["status"] != "Deferred (budget)", "order_cost"].sum()
    st.metric("Order spend", ui.money(spend), border=True,
              delta=f"{int(counts['Deferred (budget)'])} deferred" if counts["Deferred (budget)"] else None,
              delta_color="off", delta_arrow="off")

filt = st.pills("Show", STATUS_ORDER, selection_mode="multi", default=STATUS_ORDER, key="pl_filter")
view = plan[plan["status"].isin(filt or STATUS_ORDER)].copy()
icons = {"Order now": "🔴 Order now", "Order this week": "🟠 Order this week", "Plan order": "🟡 Plan order",
         "Healthy": "🟢 Healthy", "Deferred (budget)": "⚪ Deferred (budget)"}
view["status"] = view["status"].astype(str).map(icons)
st.dataframe(
    view[["status", "product_name", "warehouse", "trend", "recent_daily", "forecast_daily", "forecast_units", "stock",
          "days_of_cover", "stockout_date", "reorder_date", "suggested_qty", "order_qty", "order_cost", "revenue"]],
    hide_index=True, width="stretch", height=min(42 + 35 * len(view), 640),
    column_config={
        "status": "Status", "product_name": "Product", "warehouse": "Warehouse",
        "suggested_qty": st.column_config.NumberColumn("Needed", format="%,.0f",
                                                       help="Units needed before MOQ and case-pack rounding."),
        "order_cost": st.column_config.NumberColumn(f"Cost ({ui.currency()})", format="%,.0f"),
        "trend": st.column_config.AreaChartColumn("Weekly forecast", y_min=0),
        "recent_daily": st.column_config.NumberColumn("Recent / day", format="%.1f",
                                                      help="Average daily sales over the last 56 days."),
        "forecast_daily": st.column_config.NumberColumn("Forecast / day", format="%.1f"),
        "forecast_units": st.column_config.NumberColumn("Forecast units", format="%,.0f"),
        "stock": st.column_config.NumberColumn("Stock", format="%,.0f"),
        "days_of_cover": st.column_config.ProgressColumn("Days of cover", format="%d", min_value=0,
                                                         max_value=int(horizon)),
        "stockout_date": st.column_config.DateColumn("Runs out", format="D MMM"),
        "reorder_date": st.column_config.DateColumn("Order by", format="D MMM"),
        "order_qty": st.column_config.NumberColumn("Order qty", format="%,.0f", help="After MOQ and case packs."),
        "revenue": st.column_config.NumberColumn(f"Revenue ({ui.currency()})", format="%,.0f"),
    })

to_order = plan[plan["status"].isin(["Order now", "Order this week"])]
c1, c2, c3 = st.columns(3)
po = to_order[["product_id", "product_name", "warehouse", "order_qty", "unit_cost", "order_cost", "reorder_date",
               "stock", "days_of_cover"]].copy()
c1.download_button(f"Purchase list · {len(po)} lines (CSV)", safe_csv(po), "purchase_list.csv",
                   "text/csv", icon=":material/shopping_cart:", type="primary", width="stretch", disabled=po.empty)
c2.download_button("Full plan (CSV)", safe_csv(plan.drop(columns=["trend"])), "demand_plan.csv",
                   "text/csv", icon=":material/download:", width="stretch")
if uid:
    if c3.button("Send purchase list for approval", icon=":material/approval:", width="stretch", disabled=po.empty):
        db.create_po(uid, po.to_dict(orient="records"), note=f"{len(po)} lines · {ui.money(po['order_cost'].sum())}")
        st.toast("Draft purchase order created — approve it on the Purchase orders page.", icon=":material/check:")
else:
    c3.button("Sign in to send for approval", disabled=True, width="stretch", icon=":material/lock:")

cf = cash_flow(plan, daily)
fig = go.Figure()
fig.add_bar(x=cf.index, y=cf["money_in"], name="Money in (forecast sales)", marker_color=ui.GOOD, marker_line_width=0)
fig.add_bar(x=cf.index, y=-cf["money_out"], name="Money out (orders)", marker_color=ui.BAD, marker_line_width=0)
ui.glow_line(fig, cf.index, cf["cumulative"], ui.BRAND, "Cumulative net")
ui.style(fig, f"Cash flow · weekly ({ui.currency()})", height=340, barmode="relative")
ui.show(fig)

st.space("small")
c1, c2 = st.columns([1, 2])
status_colors = {"Order now": ui.BAD, "Order this week": ui.WARN, "Plan order": ui.LAST_YEAR, "Healthy": ui.GOOD,
                 "Deferred (budget)": ui.HIST}
fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.62, sort=False,
                       marker=dict(colors=[status_colors[s] for s in counts.index], line=dict(width=0)),
                       textinfo="value", hovertemplate="%{label}: %{value} products<extra></extra>"))
fig.add_annotation(text=f"<b>{len(plan)}</b><br>products", showarrow=False, font=dict(size=16))
ui.show(ui.style(fig, "Stock status", height=360, hover="closest"), c1)
fig = go.Figure()
for status, g in plan.groupby("status", observed=True):
    fig.add_scatter(x=g["days_of_cover"], y=g["revenue"], mode="markers", name=str(status),
                    marker=dict(size=np.clip(np.sqrt(g["order_qty"] + 1) * 1.6, 8, 46), color=status_colors[str(status)],
                                opacity=0.75, line=dict(width=1, color="rgba(255,255,255,0.5)")),
                    customdata=np.stack([g["product_name"], g["order_qty"]], axis=1),
                    hovertemplate="<b>%{customdata[0]}</b><br>%{x} days of cover · revenue %{y:,.0f}"
                                  "<br>order %{customdata[1]:,.0f} units<extra></extra>")
fig.add_vline(x=float(review), line_dash="dot", line_color=ui.HIST, annotation_text="next order cycle")
ui.style(fig, "Risk map · days of cover vs forecast revenue (bubble = order size)", height=360, hover="closest")
fig.update_xaxes(title_text="Days of cover", rangemode="tozero")
fig.update_yaxes(title_text=f"Forecast revenue ({ui.currency()})")
ui.show(fig, c2)

c1, c2 = st.columns(2)
cat = daily.groupby([pd.Grouper(key="date", freq="W"), "category"])["forecast"].sum().reset_index()
fig = go.Figure()
for name, g in cat.groupby("category"):
    fig.add_scatter(x=g["date"], y=g["forecast"], name=name, stackgroup="one", mode="lines")
ui.show(ui.style(fig, "Forecast by category · weekly units", height=380), c1)
top = plan.nlargest(10, "revenue")[::-1]
fig = go.Figure(go.Bar(x=top["revenue"], y=top["product_name"], orientation="h", marker_color=ui.BRAND,
                       hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
ui.show(ui.style(fig, f"Top 10 products by forecast revenue ({ui.currency()})", height=380, legend=False,
                 hover="closest"), c2)
