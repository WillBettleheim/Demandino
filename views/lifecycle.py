import plotly.graph_objects as go
import streamlit as st

from demandino import lifecycle, ui
from demandino.state import app

A = app()
COLORS = {"Launching": ui.LAST_YEAR, "Growing": ui.GOOD, "Mature": ui.BRAND, "Declining": ui.WARN,
          "End of life": ui.BAD}


@st.cache_data(show_spinner=False, max_entries=8)
def stages_for(data_key: str):
    return lifecycle.stages(A.sales)


ui.page_header("Product life stages",
               "Where each product is in its life — launching, growing, mature, declining or at the end — judged "
               "against the same months last year so seasons don't fool it, with what to do about stock.",
               eyebrow="Analyze")
st_df = stages_for(A.data_key)
counts = st_df["stage"].value_counts().reindex(lifecycle.STAGES, fill_value=0)
ending = st_df[st_df["stage"].isin(["Declining", "End of life"])]
ui.spotlight(f"{len(st_df)} products", str(len(ending)), "need a stock rethink",
             f"<b>{counts['Growing']}</b> growing · <b>{counts['Mature']}</b> mature · <b>{counts['Declining']}</b> "
             f"declining · <b>{counts['End of life']}</b> at end of life · <b>{counts['Launching']}</b> launching",
             [(f"{n} {s.lower()}", {"Growing": "good", "Declining": "warn", "End of life": "bad"}.get(s, ""))
              for s, n in counts.items() if n])

c1, c2 = st.columns([1.2, 1])
fig = go.Figure()
for stage, g in st_df.groupby("stage", observed=True):
    fig.add_scatter(x=g["growth"] * 100, y=g["of_peak"] * 100, mode="markers", name=str(stage),
                    marker=dict(size=12, color=COLORS[str(stage)], line=dict(width=1, color="rgba(255,255,255,.5)")),
                    customdata=g[["product_name", "growth_basis"]].to_numpy(),
                    hovertemplate="<b>%{customdata[0]}</b><br>%{x:+.0f}% %{customdata[1]}"
                                  "<br>%{y:.0f}% of its best 90 days<extra></extra>")
fig.add_vline(x=0, line_dash="dot", line_color=ui.HIST)
fig.add_hline(y=lifecycle.NEAR_PEAK * 100, line_dash="dot", line_color=ui.HIST, annotation_text="near its best")
ui.style(fig, "Growth vs how close to its best", height=380, hover="closest")
fig.update_xaxes(title_text="Growth (%)")
fig.update_yaxes(title_text="Recent sales, % of best 90 days", rangemode="tozero")
ui.show(fig, c1)

default = ending["product_id"].head(3).tolist() or st_df["product_id"].head(3).tolist()
pick = c2.multiselect("Sales curve for", st_df["product_id"].tolist(), default=default, max_selections=5,
                      format_func=dict(zip(st_df["product_id"], st_df["product_name"])).get, key="lc_pick")
fig = go.Figure()
for pid in pick:
    row = st_df.set_index("product_id").loc[pid]
    c = lifecycle.curve(A.history(pid))
    fig.add_scatter(x=c.index, y=c.values * 100, name=row["product_name"], mode="lines", line_shape="spline")
ui.style(fig, "Sales trend (13-week average), % of its best", height=340)
fig.update_yaxes(rangemode="tozero")
ui.show(fig, c2)

for stage in lifecycle.STAGES:
    g = st_df[st_df["stage"] == stage]
    if g.empty:
        continue
    with st.expander(f"{stage} · {len(g)}", expanded=stage in ("Declining", "End of life"),
                     icon={"Launching": ":material/rocket_launch:", "Growing": ":material/trending_up:",
                           "Mature": ":material/balance:", "Declining": ":material/trending_down:",
                           "End of life": ":material/block:"}[stage]):
        st.caption(lifecycle.ADVICE[stage])
        st.dataframe(g[["product_name", "category", "age_days", "recent_daily", "growth", "growth_basis", "of_peak"]],
                     hide_index=True, width="stretch", column_config={
                         "product_name": "Product", "category": "Category",
                         "age_days": st.column_config.NumberColumn("Selling for (days)", format="%d"),
                         "recent_daily": st.column_config.NumberColumn("Sells / day (90d)", format="%.1f"),
                         "growth": st.column_config.NumberColumn("Growth", format="percent"),
                         "growth_basis": "Compared with",
                         "of_peak": st.column_config.ProgressColumn("Of its best 90 days", format="percent",
                                                                    min_value=0, max_value=1)})
if len(ending):
    st.page_link("views/clearance.py", label="Clear the stock of declining products", icon=":material/sell:")
