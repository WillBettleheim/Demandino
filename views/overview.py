import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import ui
from demandino.analysis import MONTH_NAMES as MONTHS, abc_classes
from demandino.state import app

A = app()
s = A.sales.assign(revenue=A.sales["units_sold"] * A.sales["price"])
last = A.last_date

ui.page_header("Portfolio overview", "How the whole range is performing — categories, growth and which products "
                                     "matter most.")

cur = s[s["date"] > last - pd.Timedelta(days=365)]
prev = s[(s["date"] <= last - pd.Timedelta(days=365)) & (s["date"] > last - pd.Timedelta(days=730))]
monthly_units = s.groupby(pd.Grouper(key="date", freq="MS"))["units_sold"].sum()
monthly_rev = s.groupby(pd.Grouper(key="date", freq="MS"))["revenue"].sum()


def yoy(a: float, b: float) -> str | None:
    return f"{(a / b - 1) * 100:+.1f}% YoY" if b else None


with st.container(horizontal=True):
    st.metric("Products", s["product_id"].nunique(), border=True, help=f"{len(A.categories)} categories")
    st.metric("Units · last 12 months", ui.num(cur["units_sold"].sum()), border=True,
              delta=yoy(cur["units_sold"].sum(), prev["units_sold"].sum()),
              chart_data=monthly_units.tail(24).tolist(), chart_type="area")
    st.metric("Revenue · last 12 months", ui.money(cur["revenue"].sum()), border=True,
              delta=yoy(cur["revenue"].sum(), prev["revenue"].sum()), chart_data=monthly_rev.tail(24).tolist(),
              chart_type="area")
    st.metric("Avg selling price", ui.money(cur["revenue"].sum() / max(cur["units_sold"].sum(), 1), 2), border=True)
    st.metric("Days on promotion", ui.pct(cur["promo"].mean(), signed=False), border=True)

c1, c2 = st.columns([2, 1])
monthly = s.groupby([pd.Grouper(key="date", freq="MS"), "category"])["revenue"].sum().reset_index()
fig = go.Figure()
for cat, g in monthly.groupby("category"):
    fig.add_scatter(x=g["date"], y=g["revenue"], name=cat, stackgroup="one", mode="lines")
ui.show(ui.style(fig, f"Monthly revenue by category ({ui.currency()})", height=400), c1)
share = cur.groupby("category")["revenue"].sum().sort_values(ascending=False)
fig = go.Figure(go.Pie(labels=share.index, values=share.values, hole=0.55, sort=False,
                       hovertemplate="%{label}: %{percent}<extra></extra>", textinfo="percent"))
ui.show(ui.style(fig, "Revenue share · last 12 months", height=400, hover="closest"), c2)

c1, c2 = st.columns([3, 2])
tm = cur.groupby(["category", "product_name"], as_index=False)["revenue"].sum()
growth_p = []
for (c, n), g in tm.groupby(["category", "product_name"]):
    p_rev = prev[(prev["category"] == c) & (prev["product_name"] == n)]["revenue"].sum()
    growth_p.append((g["revenue"].sum() / p_rev - 1) if p_rev else 0.0)
tm["growth"] = growth_p
labels = list(tm["category"].unique()) + tm["product_name"].tolist()
parents = [""] * tm["category"].nunique() + tm["category"].tolist()
values = list(tm.groupby("category")["revenue"].sum().reindex(tm["category"].unique())) + tm["revenue"].tolist()
colors = [0.0] * tm["category"].nunique() + tm["growth"].clip(-0.5, 0.5).tolist()
fig = go.Figure(go.Treemap(labels=labels, parents=parents, values=values, branchvalues="total",
                           marker=dict(colors=colors, colorscale=[[0, ui.BAD], [0.5, "rgba(124,139,154,0.35)"],
                                                                  [1, ui.GOOD]], cmid=0, line=dict(width=1)),
                           hovertemplate="<b>%{label}</b><br>revenue %{value:,.0f}<extra></extra>",
                           textinfo="label+percent root"))
ui.style(fig, "Revenue map · last 12 months (colour = growth vs prior year)", height=430, legend=False,
         hover="closest")
fig.update_layout(margin=dict(l=4, r=4, t=48, b=4))
ui.show(fig, c1)
ym = s.assign(y=s["date"].dt.year, m=s["date"].dt.month).pivot_table(index="y", columns="m", values="units_sold",
                                                                      aggfunc="sum")
fig = go.Figure(go.Heatmap(z=ym.values, x=[MONTHS[m - 1] for m in ym.columns], y=[str(y) for y in ym.index],
                           colorscale=ui.SEQ, xgap=3, ygap=3, colorbar=dict(thickness=8),
                           hovertemplate="%{x} %{y}: %{z:,.0f} units<extra></extra>"))
ui.style(fig, "Seasonal heatmap · units by month", height=430, legend=False, hover="closest")
fig.update_yaxes(type="category", autorange="reversed")
fig.update_xaxes(type="category")
ui.show(fig, c2)

cat_tbl = pd.DataFrame({
    "units": cur.groupby("category")["units_sold"].sum(),
    "revenue": cur.groupby("category")["revenue"].sum(),
    "prev": prev.groupby("category")["revenue"].sum(),
})
cat_tbl["growth"] = cat_tbl["revenue"] / cat_tbl["prev"] - 1 if len(prev) else np.nan
cat_tbl["trend"] = [s[s["category"] == c].groupby(pd.Grouper(key="date", freq="MS"))["revenue"].sum().tail(24).tolist()
                    for c in cat_tbl.index]
st.dataframe(cat_tbl.drop(columns="prev").reset_index().sort_values("revenue", ascending=False), hide_index=True,
             width="stretch", column_config={
                 "category": "Category", "units": st.column_config.NumberColumn("Units (12m)", format="%,.0f"),
                 "revenue": st.column_config.NumberColumn(f"Revenue (12m, {ui.currency()})", format="%,.0f"),
                 "growth": st.column_config.NumberColumn("YoY", format="percent"),
                 "trend": st.column_config.LineChartColumn("Monthly revenue · 24 months", y_min=0)})

st.markdown("#### Product segmentation")
st.caption("**ABC** by revenue share — A: top 70% of revenue, B: next 20%, C: last 10%. **XYZ** by weekly demand "
           "variability — X: steady, Y: variable, Z: erratic. **AX** items deserve tight stock control; **CZ** items "
           "are candidates for lower service levels or make-to-order.")
abc = abc_classes(A.sales)
c1, c2 = st.columns([1, 2])
matrix = pd.crosstab(abc["ABC"], abc["XYZ"]).reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"], fill_value=0)
fig = go.Figure(go.Heatmap(z=matrix.values, x=["X · steady", "Y · variable", "Z · erratic"],
                           y=["A · top revenue", "B · middle", "C · tail"], colorscale=ui.SEQ, showscale=False,
                           text=matrix.values, texttemplate="%{text}", textfont=dict(size=18),
                           hovertemplate="%{y} / %{x}: %{z} products<extra></extra>"))
ui.style(fig, "Products per segment", height=330, legend=False, hover="closest")
fig.update_yaxes(autorange="reversed")
ui.show(fig, c1)
weekly = s.set_index("date").groupby("product_id")["units_sold"].resample("W").sum()
abc["trend"] = abc["product_id"].map(lambda p: weekly.loc[p].tail(26).tolist())
c2.dataframe(abc[["ABC", "XYZ", "product_name", "category", "revenue", "units", "weekly_cv", "trend"]], hide_index=True,
             width="stretch", height=330, column_config={
                 "product_name": "Product", "category": "Category",
                 "revenue": st.column_config.ProgressColumn(f"Revenue ({ui.currency()})", format="%,.0f", min_value=0,
                                                            max_value=float(abc["revenue"].max())),
                 "units": st.column_config.NumberColumn("Units", format="%,.0f"),
                 "weekly_cv": st.column_config.NumberColumn("Variability", format="%.2f"),
                 "trend": st.column_config.LineChartColumn("Weekly units · 26 weeks", y_min=0)})

growth = []
for (pid, name), g in A.sales.groupby(["product_id", "product_name"]):
    u = g.sort_values("date")["units_sold"]
    if len(u) >= 455:
        growth.append({"product": name, "yoy": u.tail(90).sum() / max(u.iloc[-455:-365].sum(), 1) - 1})
if growth:
    gdf = pd.DataFrame(growth).sort_values("yoy")
    fig = go.Figure(go.Bar(x=gdf["yoy"] * 100, y=gdf["product"], orientation="h",
                           marker_color=[ui.GOOD if v >= 0 else ui.BAD for v in gdf["yoy"]],
                           hovertemplate="%{y}: %{x:+.0f}%<extra></extra>"))
    ui.style(fig, "Momentum · last 90 days vs the same 90 days a year ago", height=max(320, 22 * len(gdf)),
             legend=False, hover="closest")
    fig.update_xaxes(ticksuffix="%")
    ui.show(fig)
