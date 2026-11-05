import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import db, ui, whatif
from demandino.security import safe_csv
from demandino.state import app

A = app()


@st.cache_data(show_spinner="Forecasting every product with and without the change…", max_entries=16)
def run_for(data_key: str, products: tuple, change: whatif.Change, horizon: int, costs: tuple) -> pd.DataFrame:
    return whatif.run(A.fc, A.elasticities, A.level_uncertainty, list(products), change, horizon, dict(costs) or None)


ui.page_header("Store what-if",
               "Try one change across many products at once — a price rise, a sale on a category — and see what it "
               "does to units, revenue and profit for the whole shop.", eyebrow="Forecast")

with st.container(border=True, key="card-whatif"):
    c1, c2 = st.columns([1, 1.4])
    scope = c1.segmented_control("Apply to", ["Whole shop", "Categories", "Products"], default="Whole shop",
                                 required=True, key="wi_scope")
    if scope == "Categories":
        cats = c2.multiselect("Categories", A.categories, default=A.categories[:1], key="wi_cats")
        products = A.products.loc[A.products["category"].isin(cats), "product_id"].tolist()
    elif scope == "Products":
        products = c2.multiselect("Products", list(A.label_of), format_func=A.label_of.get,
                                  default=list(A.label_of)[:3], key="wi_pids")
    else:
        products = A.products["product_id"].tolist()
        c2.caption(f"All {len(products)} products")
    c1, c2, c3 = st.columns(3)
    price_change = c1.slider("Change regular prices by", -30, 30, 5, 1, format="%+d%%", key="wi_price") / 100
    horizon = c2.select_slider("Over the next", [30, 60, 90, 180], value=60, format_func=lambda d: f"{d} days",
                               key="wi_h")
    first = A.last_date + pd.Timedelta(days=1)
    discount, start, end = 0.0, None, None
    if c3.toggle("Add a sale", key="wi_sale"):
        discount = c3.slider("Discount", 5, 60, 20, 5, format="%d%%", key="wi_disc") / 100
        dates = c3.date_input("Sale dates", (first.date() + pd.Timedelta(days=7), first.date() + pd.Timedelta(days=13)),
                              min_value=first.date(), max_value=(first + pd.Timedelta(days=horizon - 1)).date(),
                              key="wi_dates")
        if len(dates) == 2:
            start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])

if not products:
    st.info("Pick at least one category or product.", icon=":material/touch_app:")
    st.stop()
change = whatif.Change(price_change, discount if start is not None else 0.0, start, end)
saved = db.get_setting(A.user_id, f"stock:{A.settings_key}") if A.user_id else None
costs = tuple(sorted(pd.DataFrame(saved).groupby("product_id")["unit_cost"].mean().items())) if saved else ()
df = run_for(A.data_key, tuple(products), change, int(horizon), costs)
t = whatif.totals(df)

tone = "good" if t["profit_change"] > 0 else "bad"
ui.spotlight(f"{change.describe()} · {len(products)} products · next {horizon} days",
             ("+" if t["profit_change"] > 0 else "") + ui.money(t["profit_change"]), "profit change",
             f"Units <b>{t['units_pct']:+.1%}</b> ({ui.num(t['units_before'])} → {ui.num(t['units_after'])}) · revenue "
             f"<b>{t['revenue_pct']:+.1%}</b> · profit <b>{t['profit_pct']:+.1%}</b>",
             [("Worth it" if t["profit_change"] > 0 else "Costs profit", tone),
              (f"{int((df['profit_change'] > 0).sum())} products gain", "good"),
              (f"{int((df['profit_change'] < 0).sum())} lose", "warn")])
ui.finding_cards(whatif.verdict(t, df, ui.currency()))

c1, c2 = st.columns(2)
cat = whatif.by_category(df)
fig = go.Figure(go.Bar(x=cat["category"], y=cat["profit_change"],
                       marker_color=[ui.GOOD if v >= 0 else ui.BAD for v in cat["profit_change"]],
                       hovertemplate="%{x}: %{y:+,.0f}<extra></extra>"))
ui.show(ui.style(fig, f"Profit change by category ({ui.currency()})", height=340, legend=False, hover="closest"), c1)
top = pd.concat([df.head(6), df.tail(6)]).drop_duplicates("product_id").sort_values("profit_change")
fig = go.Figure(go.Bar(x=top["profit_change"], y=top["product_name"], orientation="h",
                       marker_color=[ui.GOOD if v >= 0 else ui.BAD for v in top["profit_change"]],
                       hovertemplate="%{y}: %{x:+,.0f}<extra></extra>"))
ui.show(ui.style(fig, "Biggest winners and losers", height=340, legend=False, hover="closest"), c2)

with st.expander(f"All products ({len(df)})", icon=":material/table_rows:"):
    st.dataframe(df[["product_name", "category", "units_before", "units_after", "revenue_change", "profit_before",
                     "profit_after", "profit_change"]], hide_index=True, width="stretch", column_config={
        "product_name": "Product", "category": "Category",
        "units_before": st.column_config.NumberColumn("Units now", format="%,.0f"),
        "units_after": st.column_config.NumberColumn("Units with change", format="%,.0f"),
        "revenue_change": st.column_config.NumberColumn("Revenue change", format="%+,.0f"),
        "profit_before": st.column_config.NumberColumn("Profit now", format="%,.0f"),
        "profit_after": st.column_config.NumberColumn("Profit with change", format="%,.0f"),
        "profit_change": st.column_config.NumberColumn("Profit change", format="%+,.0f")})
st.caption("Price changes use each product's price sensitivity; sales use what the model learned about promotion "
           "days. Profit uses your saved unit costs, or 55% of the price where none is saved. Treat it as an estimate "
           "and test individual products on the Forecast page.")
st.download_button("Download what-if results (CSV)", safe_csv(df), "store_what_if.csv", "text/csv", icon=":material/download:",
                   key="dl_store_what_if")
