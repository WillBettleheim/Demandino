import numpy as np
import plotly.graph_objects as go
import streamlit as st

from demandino import changes, ui
from demandino.security import safe_csv
from demandino.state import app

A = app()


@st.cache_data(show_spinner="Comparing with last week's forecasts…", max_entries=8)
def table_for(data_key: str, horizon: int):
    return changes.all_products(A.fc, horizon)


ui.page_header("What changed",
               "How each product's forecast moved since a week ago, and why — last week's sales, last year's "
               "pattern, being a week closer, or price.", eyebrow="Forecast")
horizon = st.segmented_control("Forecast period", [14, 30, 60], default=30, required=True,
                               format_func=lambda d: f"next {d} days", key="ch_h")
df = table_for(A.data_key, int(horizon))
up, down = df[df["change_pct"] >= 0.05], df[df["change_pct"] <= -0.05]
ui.spotlight(f"Next {horizon} days · now vs a week ago", f"{(df['now'].sum() / df['before'].sum() - 1):+.0%}",
             "total change",
             f"<b>{ui.num(df['before'].sum())}</b> → <b>{ui.num(df['now'].sum())}</b> units · "
             f"<b>{len(up)}</b> products up 5%+ · <b>{len(down)}</b> down 5%+ · "
             f"{len(df) - len(up) - len(down)} about the same",
             [(f"{len(up)} up", "good")] * bool(len(up)) + [(f"{len(down)} down", "warn")] * bool(len(down)))

ui.finding_cards([f"**{r.product_name}:** {r.why}" for r in df.head(6).itertuples()])

c1, c2 = st.columns([1, 1.2])
pick = c1.selectbox("Why did this one change?", df["product_id"].tolist(),
                    format_func=dict(zip(df["product_id"], df["product_name"])).get, key="ch_pick")
e = changes.explain(A.fc, pick, int(horizon))
c1.markdown(changes.sentence(e).replace("$", "\\$"))
c1.caption("The same model forecasts these days twice: from the sales known a week ago, and from all sales now. "
           "This week's information is then added one piece at a time to see what each piece did.")
names = ["A week ago"] + [n for n, _ in e["parts"]] + ["Now"]
vals = [e["before"]] + [v for _, v in e["parts"]] + [e["now"]]
fig = go.Figure(go.Waterfall(x=names, y=vals, measure=["absolute"] + ["relative"] * len(e["parts"]) + ["total"],
                             increasing=dict(marker_color=ui.GOOD), decreasing=dict(marker_color=ui.BAD),
                             totals=dict(marker_color=ui.BRAND), connector=dict(line=dict(color=ui.HIST, width=1)),
                             hovertemplate="%{x}: %{y:+,.0f}<extra></extra>"))
steps = np.cumsum(vals[:-1])
low, high = float(min(steps.min(), e["now"])), float(max(steps.max(), e["now"]))
pad = max((high - low) * 0.6, high * 0.02, 1.0)
ui.style(fig, f"Forecast for the next {horizon} days: a week ago → now (units)", height=360, legend=False,
         hover="closest")
fig.update_yaxes(range=[max(0.0, low - pad), high + pad])  # zoom in so the changes are visible
ui.show(fig, c2)

with st.expander(f"All products ({len(df)})", icon=":material/table_rows:"):
    st.dataframe(df.drop(columns=["product_id"]), hide_index=True, width="stretch", column_config={
        "product_name": "Product", "before": st.column_config.NumberColumn("A week ago", format="%,.0f"),
        "now": st.column_config.NumberColumn("Now", format="%,.0f"),
        "change": st.column_config.NumberColumn("Change", format="%+,.0f"),
        "change_pct": st.column_config.NumberColumn("Change %", format="percent"),
        "last_week_actual": st.column_config.NumberColumn("Sold last week", format="%,.0f"),
        "last_week_expected": st.column_config.NumberColumn("Expected last week", format="%,.0f"),
        "why": st.column_config.TextColumn("Why", width="large")})
st.download_button("Download forecast changes (CSV)", safe_csv(df), "forecast_changes.csv", "text/csv", icon=":material/download:",
                   key="dl_forecast_changes")
