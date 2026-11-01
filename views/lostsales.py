import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import db, lostsales, ui
from demandino.security import safe_csv
from demandino.state import app

A = app()
ui.page_header("Lost sales",
               "What empty shelves cost you: demand on days a product had run out, valued in money — by product, "
               "by month and by stock-out.", eyebrow="Analyze")

saved = db.get_setting(A.user_id, f"stock:{A.data_key}") if A.user_id else None
unit_cost = (pd.DataFrame(saved).groupby("product_id")["unit_cost"].mean().to_dict()
             if saved and "unit_cost" in pd.DataFrame(saved) else None)
period = st.segmented_control("Period", ["Last 90 days", "Last year", "All history"], default="Last year",
                              required=True, key="ls_period")
days = {"Last 90 days": 90, "Last year": 365, "All history": None}[period]
last = A.fc.sales["date"].max()
rows = lostsales.window(lostsales.daily(A.fc.sales, unit_cost), last, days)
prod = lostsales.by_product(rows)
total, profit = float(rows["lost_sales"].sum()), float(rows["lost_profit"].sum())
ui.spotlight(f"{period} · sales lost to stock-outs", ui.money(total), "of sales missed",
             f"About <b>{ui.money(profit)}</b> of profit · <b>{ui.num(rows['lost_units'].sum())}</b> units · "
             f"<b>{len(rows)}</b> stock-out days across <b>{len(prod)}</b> products",
             [("Using your unit costs" if unit_cost else f"Profit assumes cost = {lostsales.DEFAULT_COST_RATIO:.0%} of "
               "price", "acc")])
ui.finding_cards(lostsales.ideas(prod, lostsales.potential(A.fc.sales, last, days), ui.currency()))
if rows.empty:
    st.stop()

c1, c2 = st.columns([1.2, 1])
monthly = lostsales.by_month(rows)
fig = go.Figure()
for cat, g in monthly.groupby("category"):
    fig.add_bar(x=g["month"], y=g["lost_sales"], name=cat, hovertemplate="%{x|%b %Y}: %{y:,.0f}<extra>" + cat + "</extra>")
ui.show(ui.style(fig, f"Lost sales by month ({ui.currency()})", height=360, barmode="stack"), c1)
top = prod.head(10).iloc[::-1]
fig = go.Figure(go.Bar(x=top["lost_sales"], y=top["product_name"], orientation="h", marker_color=ui.BAD,
                       customdata=top[["days", "episodes"]].to_numpy(),
                       hovertemplate="%{y}: %{x:,.0f}<br>%{customdata[0]} days · %{customdata[1]} stock-outs"
                                     "<extra></extra>"))
ui.show(ui.style(fig, "Products losing the most", height=360, legend=False, hover="closest"), c2)

st.markdown("#### Costliest stock-outs")
st.dataframe(lostsales.episodes(rows).head(15).drop(columns=["product_id"]), hide_index=True, width="stretch",
             column_config={"product_name": "Product", "start": st.column_config.DateColumn("From", format="D MMM YYYY"),
                            "end": st.column_config.DateColumn("To", format="D MMM YYYY"), "days": "Days",
                            "lost_units": st.column_config.NumberColumn("Units lost", format="%,.0f"),
                            "lost_sales": st.column_config.NumberColumn(f"Sales lost ({ui.currency()})", format="%,.0f")})
with st.expander(f"All products ({len(prod)})", icon=":material/table_rows:"):
    st.dataframe(prod.drop(columns=["product_id"]), hide_index=True, width="stretch", column_config={
        "product_name": "Product", "category": "Category", "days": "Stock-out days", "episodes": "Times run out",
        "lost_units": st.column_config.NumberColumn("Units lost", format="%,.0f"),
        "lost_sales": st.column_config.NumberColumn(f"Sales lost ({ui.currency()})", format="%,.0f"),
        "lost_profit": st.column_config.NumberColumn(f"Profit lost ({ui.currency()})", format="%,.0f")})
st.caption("Stock-out days are runs of zero sales for products that normally sell (or days your stock data shows "
           "none). Demand on those days is estimated from the days around them, so treat figures as estimates.")
st.page_link("views/planning.py", label="Prevent the next one on Demand planning", icon=":material/inventory:")
st.download_button("Download lost sales by product (CSV)", safe_csv(prod), "lost_sales.csv", "text/csv", icon=":material/download:",
                   key="dl_lost_sales")
