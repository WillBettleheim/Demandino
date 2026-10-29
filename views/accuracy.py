import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import ui
from demandino.model import RANGE_LABEL, TARGET_COVERAGE, ForecastResult
from demandino.state import ROOT, app

A = app()
bt = A.backtest
m = bt["metrics"]
preds = bt["predictions"]
cal = bt.get("calibration", {})
open_preds = preds[preds["open"]] if "open" in preds else preds
target_miss = 1 - TARGET_COVERAGE

ui.page_header(
    "Model accuracy",
    f"An honest test: the model was trained and calibrated only on sales up to <b>{bt['cutoff']:%d %b %Y}</b>, then "
    f"asked to forecast the next 90 days for all {preds['product_id'].nunique()} products. Nothing from those 90 "
    "days was seen in advance. Below is how often reality landed outside its predicted range.",
    eyebrow="Reliability report")

daily_hit = 1 - m["Daily miss rate"]
ui.spotlight(
    "Range hit rate · held-out days", f"{daily_hit * 100:.1f}%", "of real daily sales inside the range",
    f"Wrong-range rate <b>{m['Daily miss rate'] * 100:.1f}%</b> per day and <b>{m['Weekly miss rate'] * 100:.1f}%</b> "
    f"per weekly total — target ≤ 7–8%.",
    [("Daily ✓ under target" if m["Daily miss rate"] <= 0.08 else "Daily above target",
      "good" if m["Daily miss rate"] <= 0.08 else "bad"),
     ("Weekly ✓ under target" if m["Weekly miss rate"] <= 0.08 else "Weekly above target",
      "good" if m["Weekly miss rate"] <= 0.08 else "bad"),
     (f"{len(open_preds):,} product-days tested", ""), (f"Calibrated to a {RANGE_LABEL} range", "violet")])

with st.container(horizontal=True):
    st.metric("Daily misses", f"{m['Daily miss rate'] * 100:.1f}%", border=True,
              help="Share of held-out trading days where actual sales fell outside the predicted range.")
    st.metric("Weekly misses", f"{m['Weekly miss rate'] * 100:.1f}%", border=True,
              help="Share of product-weeks where the actual weekly total fell outside the predicted weekly range.")
    st.metric("Daily error (WAPE)", f"{m['WAPE (daily)'] * 100:.0f}%", border=True,
              delta=f"{(m['WAPE (daily)'] - m['Baseline WAPE: recent average']) * 100:+.1f} pts vs simple average",
              delta_color="inverse", help="Total absolute error ÷ total actual units. Lower is better.")
    st.metric("Weekly error (WAPE)", f"{m['WAPE (weekly)'] * 100:.0f}%", border=True)
    st.metric("Bias", f"{m['Bias'] * 100:+.1f}%", border=True, help="Positive = forecasts too high overall.")

with ui.card("method"):
    st.markdown("##### :material/science: How this was tested")
    st.markdown(
        f"1. **Train** on everything before the last {90 + 180} days.\n"
        f"2. **Calibrate**: forecast the following 180 days from several start dates, compare with what really sold, "
        f"and widen or narrow the ranges until {RANGE_LABEL} of outcomes fall inside "
        f"(range widening {cal.get('conformal_q', 0):.2f} × level, multi-day level error "
        f"{cal.get('level_uncertainty', 0) * 100:.0f}%).\n"
        f"3. **Retrain** on everything before {bt['cutoff']:%d %b %Y} and **test** on the 90 unseen days after it.\n\n"
        "Days the whole shop was closed are excluded — a 0–0 range trivially “covers” them and would flatter the "
        "score. **Point accuracy (WAPE) and range reliability are different things:** day-to-day sales of a single "
        "product are noisy, so the ranges are honest about that rather than pretending to be precise.")

c1, c2 = st.columns(2)
p = open_preds.assign(week=((open_preds["date"] - open_preds["date"].min()).dt.days // 7) + 1,
                      miss=lambda d: ~((d["actual"] >= d["lower"]) & (d["actual"] <= d["upper"])))
by_week = p.groupby("week")["miss"].mean()
fig = go.Figure()
fig.add_hrect(y0=0, y1=8, fillcolor="rgba(52,211,153,0.07)", line_width=0)
fig.add_hline(y=8, line_dash="dash", line_color=ui.GOOD, annotation_text="8% target", annotation_position="top left")
fig.add_bar(x=by_week.index, y=by_week.values * 100, marker_color=[ui.BRAND if v <= 0.08 else ui.BAD for v in by_week],
            marker_line_width=0, hovertemplate="Week %{x}: %{y:.1f}% missed<extra></extra>", name="Missed")
ui.style(fig, "Misses by weeks ahead", height=320, legend=False, hover="closest")
fig.update_xaxes(title_text="Weeks ahead", dtick=1)
fig.update_yaxes(ticksuffix="%", rangemode="tozero")
ui.show(fig, c1)
by_cat = p.groupby("category")["miss"].mean().sort_values()
fig = go.Figure(go.Bar(x=by_cat.values * 100, y=by_cat.index, orientation="h", marker_line_width=0,
                       marker_color=[ui.BRAND if v <= 0.08 else ui.BAD for v in by_cat],
                       hovertemplate="%{y}: %{x:.1f}% missed<extra></extra>"))
fig.add_vline(x=8, line_dash="dash", line_color=ui.GOOD)
ui.style(fig, "Misses by category", height=320, legend=False, hover="closest")
fig.update_xaxes(ticksuffix="%")
ui.show(fig, c2)

c1, c2 = st.columns(2)
comp = pd.DataFrame({"Method": ["Demandino ML", "Recent 56-day average", "Same day last year"],
                     "WAPE": [m["WAPE (daily)"], m["Baseline WAPE: recent average"],
                              m["Baseline WAPE: same day last year"]]})[::-1]
fig = go.Figure(go.Bar(x=comp["WAPE"] * 100, y=comp["Method"], orientation="h", marker_line_width=0,
                       marker_color=[ui.HIST, ui.HIST, ui.FC], text=(comp["WAPE"] * 100).round(1).astype(str) + "%",
                       textposition="outside", hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
ui.style(fig, "Point error vs simple baselines (lower is better)", height=300, legend=False, hover="closest")
fig.update_xaxes(range=[0, comp["WAPE"].max() * 120], ticksuffix="%")
ui.show(fig, c1)
imp = bt["importance"].head(10)[::-1]
fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ui.FC, marker_line_width=0,
                       hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
ui.show(ui.style(fig, "What the model relies on (permutation importance)", height=300, legend=False,
                 hover="closest"), c2)

bench = ROOT / "data" / "benchmark_results.json"
if bench.exists():
    st.markdown("#### :material/verified: Proven across datasets and time windows")
    data = json.loads(bench.read_text())
    t = pd.DataFrame(data["results"])
    t["window"] = t["test_start"] + " → " + t["test_end"]
    st.dataframe(t[["dataset", "window", "products", "Daily miss rate", "Weekly miss rate", "WAPE (daily)",
                    "Baseline WAPE: recent average", "WAPE (weekly)"]], hide_index=True, width="stretch",
                 column_config={
                     "dataset": "Dataset", "window": "Unseen test window", "products": "Products",
                     "Daily miss rate": st.column_config.ProgressColumn("Daily misses", format="percent", min_value=0,
                                                                        max_value=0.10),
                     "Weekly miss rate": st.column_config.ProgressColumn("Weekly misses", format="percent",
                                                                         min_value=0, max_value=0.10),
                     "WAPE (daily)": st.column_config.NumberColumn("Daily WAPE", format="percent"),
                     "Baseline WAPE: recent average": st.column_config.NumberColumn("Baseline WAPE", format="percent"),
                     "WAPE (weekly)": st.column_config.NumberColumn("Weekly WAPE", format="percent")})
    st.caption(f"Generated {data['generated']} by `python scripts/benchmark.py`. Real data: UCI Online Retail II "
               "(a UK online gift wholesaler, CC BY 4.0). Progress bars run to 10%; the target is ≤ 8%.")

st.markdown("#### Inspect a product")
pid = st.selectbox("Product", list(A.label_of), format_func=A.label_of.get, key="acc_pid")
pp = preds[preds["product_id"] == pid]
hist = A.sales[(A.sales["product_id"] == pid) & (A.sales["date"] <= bt["cutoff"])]
res = ForecastResult(pd.DatetimeIndex(pp["date"]), pp["forecast"].to_numpy(), pp["lower"].to_numpy(),
                     pp["upper"].to_numpy(), np.zeros(len(pp)), np.zeros(len(pp), int), 0.0)
fig = ui.forecast_chart(res, hist, "Back-test: forecast vs what actually happened")
inside = (pp["actual"] >= pp["lower"]) & (pp["actual"] <= pp["upper"])
fig.add_scatter(x=pp.loc[inside, "date"], y=pp.loc[inside, "actual"], mode="markers", name="Actual · inside range",
                marker=dict(color=ui.LAST_YEAR, size=5))
fig.add_scatter(x=pp.loc[~inside, "date"], y=pp.loc[~inside, "actual"], mode="markers", name="Actual · missed",
                marker=dict(color=ui.BAD, size=8, symbol="x"))
ui.show(fig)
row = bt["per_product"].set_index("product_id").loc[pid]
st.caption(f"Over the 90 days this product sold **{row['actual_units']:,.0f}** units; the model predicted "
           f"**{row['forecast_units']:,.0f}** ({row['total_error_%']:+.1f}%). Daily range missed on "
           f"**{row['miss_rate'] * 100:.1f}%** of days.")

with st.expander("All products", icon=":material/table:"):
    st.dataframe(bt["per_product"].sort_values("miss_rate"), hide_index=True, width="stretch", column_config={
        "product_id": "ID", "product_name": "Product", "category": "Category",
        "actual_units": st.column_config.NumberColumn("Actual", format="%,.0f"),
        "forecast_units": st.column_config.NumberColumn("Forecast", format="%,.0f"),
        "WAPE_daily": st.column_config.NumberColumn("Daily WAPE", format="percent"),
        "total_error_%": st.column_config.NumberColumn("Total error", format="%+.1f%%"),
        "miss_rate": st.column_config.ProgressColumn("Daily misses", format="percent", min_value=0, max_value=0.2)})
