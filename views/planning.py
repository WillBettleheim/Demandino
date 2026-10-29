import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import ui
from demandino.planning import STATUS_ORDER, default_stock, portfolio_plan
from demandino.security import safe_csv
from demandino.state import app

A = app()


@st.cache_data(show_spinner="Forecasting every product…", max_entries=16)
def plan_for(data_key: str, stock: pd.DataFrame, horizon: int, review: int, service: float):
    return portfolio_plan(A.fc, A.level_uncertainty, stock, horizon, review, service)


ui.page_header("Demand planning",
               "Forecast every product at once, see which ones will run out, and get a ready-made purchase list.")

with ui.card("plan-controls"):
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1.2])
    horizon = c1.segmented_control("Planning horizon", [30, 60, 90], default=60, required=True,
                                   format_func=lambda d: f"{d} days", key="pl_h")
    review = c2.number_input("Days between orders", 1, 60, 14, key="pl_review")
    service = c3.number_input("Service level %", 80.0, 99.9, 95.0, 0.5, key="pl_service") / 100
    default_lead = c4.number_input("Default lead time (days)", 1, 180, 14, key="pl_lead")

stock_key = f"pl_stock_{A.data_key}"
if stock_key not in st.session_state:
    st.session_state[stock_key] = default_stock(A.sales)
base = st.session_state[stock_key].copy()
if st.session_state.get("pl_lead_applied") != default_lead:
    base["lead_time"] = default_lead
    st.session_state[stock_key] = base
    st.session_state["pl_lead_applied"] = default_lead

with st.expander("Stock on hand & lead times — edit to match your warehouse", icon=":material/edit_note:"):
    st.caption("Starting values are illustrative (5–60 days of recent sales). Type your real stock levels, or paste a "
               "column from a spreadsheet.")
    edited = st.data_editor(base, key=f"pl_editor_{A.data_key}", hide_index=True, width="stretch",
                            disabled=["product_id", "product_name", "category"],
                            column_config={"product_id": "ID", "product_name": "Product", "category": "Category",
                                           "stock": st.column_config.NumberColumn("Stock on hand", min_value=0, step=1),
                                           "lead_time": st.column_config.NumberColumn("Lead time (days)", min_value=1,
                                                                                      max_value=180, step=1)})
    if st.button("Reset to illustrative values", icon=":material/restart_alt:"):
        st.session_state[stock_key] = default_stock(A.sales)
        st.session_state.pop(f"pl_editor_{A.data_key}", None)
        st.rerun()

plan, daily = plan_for(A.data_key, edited, int(horizon), int(review), float(service))

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

filt = st.pills("Show", STATUS_ORDER, selection_mode="multi", default=STATUS_ORDER, key="pl_filter")
view = plan[plan["status"].isin(filt or STATUS_ORDER)].copy()
icons = {"Order now": "🔴 Order now", "Order this week": "🟠 Order this week", "Plan order": "🟡 Plan order",
         "Healthy": "🟢 Healthy"}
view["status"] = view["status"].astype(str).map(icons)
st.dataframe(
    view[["status", "product_name", "category", "trend", "recent_daily", "forecast_daily", "forecast_units", "stock",
          "days_of_cover", "stockout_date", "reorder_date", "order_qty", "revenue"]],
    hide_index=True, width="stretch", height=min(42 + 35 * len(view), 640),
    column_config={
        "status": "Status", "product_name": "Product", "category": "Category",
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
        "order_qty": st.column_config.NumberColumn("Order qty", format="%,.0f"),
        "revenue": st.column_config.NumberColumn(f"Revenue ({ui.currency()})", format="%,.0f"),
    })

to_order = plan[plan["status"].isin(["Order now", "Order this week"])]
c1, c2 = st.columns([1, 1])
po = to_order[["product_id", "product_name", "category", "order_qty", "reorder_date", "stock", "days_of_cover"]].copy()
c1.download_button(f"Purchase list · {len(po)} products (CSV)", safe_csv(po), "purchase_list.csv",
                   "text/csv", icon=":material/shopping_cart:", type="primary", width="stretch", disabled=po.empty)
c2.download_button("Full plan (CSV)", safe_csv(plan.drop(columns=["trend"])), "demand_plan.csv",
                   "text/csv", icon=":material/download:", width="stretch")

st.space("small")
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
