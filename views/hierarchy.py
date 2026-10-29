import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import ui
from demandino.model import RANGE_LABEL
from demandino.state import app

A = app()


@st.cache_data(show_spinner="Forecasting every product…", max_entries=8)
def all_forecasts(data_key: str, horizon: int) -> pd.DataFrame:
    rows = []
    for pid, g in A.sales.groupby("product_id"):
        res = A.fc.forecast_product(pid, horizon)
        rows.append(pd.DataFrame({"date": res.dates, "product_id": pid, "product_name": g["product_name"].iloc[0],
                                  "category": g["category"].iloc[0], "forecast": res.mean, "sd": res.daily_sd,
                                  "revenue": res.mean * res.price}))
    return pd.concat(rows, ignore_index=True)


def interval(df: pd.DataFrame, z: float, lu: float) -> tuple[float, float, float]:
    """Range of a total: daily noise adds up independently; the level error is shared within a category and
    independent between categories."""
    total = df["forecast"].sum()
    var = (df["sd"] ** 2).sum() + sum((lu * g["forecast"].sum()) ** 2 for _, g in df.groupby("category"))
    sd = float(np.sqrt(var))
    return float(total), max(0.0, total - z * sd), total + z * sd


ui.page_header("Store & category totals",
               "Product forecasts rolled up into categories and the whole store. Totals always add up: every level is "
               "the sum of the products below it, with ranges that account for shared uncertainty.", eyebrow="Analyze")
horizon = st.segmented_control("Horizon", [30, 60, 90, 180], default=90, required=True,
                               format_func=lambda d: f"{d} days", key="hi_h")
fx = all_forecasts(A.data_key, int(horizon))
z = 1.96
lu = A.level_uncertainty

store_t, store_lo, store_hi = interval(fx, z, lu)
ui.spotlight(f"Whole store · next {horizon} days", ui.num(store_t), "units expected",
             f"{RANGE_LABEL} range <b>{ui.num(store_lo)} – {ui.num(store_hi)}</b> · revenue "
             f"<b>{ui.money(fx['revenue'].sum())}</b> · {fx['product_id'].nunique()} products in "
             f"{fx['category'].nunique()} categories",
             [("Totals reconcile exactly", "good"), (f"{RANGE_LABEL} ranges", "acc")],
             count=(store_t, f"store_{horizon}", 0))

cats = []
for cat, g in fx.groupby("category"):
    t, lo, hi = interval(g, z, lu)
    cats.append({"category": cat, "units": t, "low": lo, "high": hi, "revenue": g["revenue"].sum(),
                 "share": t / store_t, "products": g["product_id"].nunique(),
                 "weekly": g.groupby(pd.Grouper(key="date", freq="W"))["forecast"].sum().round(1).tolist()})
ct = pd.DataFrame(cats).sort_values("units", ascending=False)

c1, c2 = st.columns([1, 1])
prod = fx.groupby(["category", "product_name"], as_index=False)[["forecast", "revenue"]].sum()
labels = ["Store"] + ct["category"].tolist() + prod["product_name"].tolist()
ids = ["Store"] + ct["category"].tolist() + (prod["category"] + "/" + prod["product_name"]).tolist()
parents = [""] + ["Store"] * len(ct) + prod["category"].tolist()
values = [store_t] + ct["units"].tolist() + prod["forecast"].tolist()
fig = go.Figure(go.Sunburst(ids=ids, labels=labels, parents=parents, values=values, branchvalues="total",
                            marker=dict(line=dict(width=1)),
                            hovertemplate="<b>%{label}</b><br>%{value:,.0f} units<extra></extra>", insidetextorientation="radial"))
ui.style(fig, "Store → category → product (units)", height=460, legend=False, hover="closest")
fig.update_layout(margin=dict(l=4, r=4, t=48, b=4))
ui.show(fig, c1)
wk = fx.groupby([pd.Grouper(key="date", freq="W"), "category"])["forecast"].sum().reset_index()
fig = go.Figure()
for cat, g in wk.groupby("category"):
    fig.add_scatter(x=g["date"], y=g["forecast"], name=cat, stackgroup="one", mode="lines", line_shape="spline")
ui.show(ui.style(fig, "Weekly units by category", height=460), c2)

fig = go.Figure(go.Bar(x=ct["category"], y=ct["units"], marker_color=ui.BRAND, marker_line_width=0,
                       error_y=dict(type="data", symmetric=False, array=ct["high"] - ct["units"],
                                    arrayminus=ct["units"] - ct["low"], color=ui.HIST),
                       hovertemplate="%{x}: %{y:,.0f} units<extra></extra>"))
ui.show(ui.style(fig, f"Category totals with {RANGE_LABEL} ranges", height=320, legend=False, hover="closest"))
st.dataframe(ct, hide_index=True, width="stretch", column_config={
    "category": "Category", "units": st.column_config.NumberColumn("Units", format="%,.0f"),
    "low": st.column_config.NumberColumn("Low", format="%,.0f"), "high": st.column_config.NumberColumn("High", format="%,.0f"),
    "revenue": st.column_config.NumberColumn(f"Revenue ({ui.currency()})", format="%,.0f"),
    "share": st.column_config.ProgressColumn("Share of store", format="percent", min_value=0, max_value=1),
    "products": "Products", "weekly": st.column_config.AreaChartColumn("Weekly", y_min=0)})
st.caption("Why the store range is narrower than adding category ranges: random ups and downs in different "
           "categories partly cancel out.")
