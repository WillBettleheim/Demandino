import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import lifecycle, ui
from demandino.security import safe_csv
from demandino.state import app

A = app()


@st.cache_data(show_spinner="Forecasting…", max_entries=16)
def compare_for(data_key: str, products: tuple, horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stages = lifecycle.stages(A.sales).set_index("product_id")
    weekly, rows = [], []
    for pid in products:
        res = A.fc.forecast_product(pid, horizon)
        hist = A.history(pid)
        name = hist["product_name"].iloc[0]
        s = pd.Series(res.mean, index=res.dates).resample("W").sum()
        weekly.append(pd.DataFrame({"week": s.index, "product": name, "units": s.values}))
        total, low, high = res.total_interval()
        recent = float(hist["units_sold"].tail(56).mean())
        rows.append({"Product": name, "Life stage": str(stages.loc[pid, "stage"]), "Forecast units": total,
                     "Likely low": low, "Likely high": high, "Per day": total / horizon, "Recent per day": recent,
                     "Change": (total / horizon) / recent - 1 if recent > 0 else None,
                     "Revenue": float((res.mean * res.price).sum()), "Price": float(res.price[0])})
    return pd.concat(weekly, ignore_index=True), pd.DataFrame(rows)


ui.page_header("Compare products", "Put up to four products side by side: their forecasts, life stage, revenue and "
               "how each is moving against its recent sales.", eyebrow="Analyze")
ids = list(A.label_of)
c1, c2 = st.columns([2, 1])
picked = c1.multiselect("Products", ids, default=ids[:2], max_selections=4, format_func=A.label_of.get, key="cmp_pids")
horizon = c2.segmented_control("Next", [30, 60, 90], default=60, required=True, format_func=lambda d: f"{d} days",
                               key="cmp_h")
if len(picked) < 2:
    st.info("Pick at least two products to compare.", icon=":material/compare_arrows:")
    st.stop()
weekly, table = compare_for(A.data_key, tuple(picked), int(horizon))
best = table.loc[table["Revenue"].idxmax()]
fastest = table.loc[table["Change"].fillna(-9).idxmax()]
ui.finding_cards([f"**Most revenue:** {best['Product']}, about {ui.money(best['Revenue'])} over the next {horizon} days.",
                  f"**Rising fastest:** {fastest['Product']}, {fastest['Change']:+.0%} against its last 8 weeks."])
index = st.toggle("Compare shapes (each as a share of its own peak)", value=True, key="cmp_index",
                  help="Products selling very different amounts are hard to compare on one scale.")
fig = go.Figure()
for name, g in weekly.groupby("product", sort=False):
    y = g["units"] / g["units"].max() * 100 if index and g["units"].max() > 0 else g["units"]
    fig.add_scatter(x=g["week"], y=y, name=name, mode="lines+markers", line_shape="spline",
                    customdata=g["units"], hovertemplate="%{customdata:,.0f} units<extra>" + "%{fullData.name}</extra>")
ui.style(fig, "Weekly forecast" + (" · % of each product's peak" if index else " · units"), height=380)
fig.update_yaxes(rangemode="tozero")
ui.show(fig)
st.dataframe(table, hide_index=True, width="stretch", column_config={
    **{c: st.column_config.NumberColumn(format="%,.0f") for c in ("Forecast units", "Likely low", "Likely high")},
    "Per day": st.column_config.NumberColumn(format="%.1f"), "Recent per day": st.column_config.NumberColumn(format="%.1f"),
    "Change": st.column_config.NumberColumn("Vs recent", format="percent"),
    "Revenue": st.column_config.NumberColumn(f"Revenue ({ui.currency()})", format="%,.0f"),
    "Price": st.column_config.NumberColumn(f"Price ({ui.currency()})", format="%.2f")})
st.download_button("Download comparison (CSV)", safe_csv(table), "product_comparison.csv", "text/csv",
                   icon=":material/download:", key="dl_compare")
