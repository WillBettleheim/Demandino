import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import db, targets, ui
from demandino.security import safe_csv
from demandino.state import app

A = app()
ui.page_header("Sales targets",
               "Set a target for a month, a quarter or any period. Demandino adds the forecast for the days left to "
               "what you've already sold, tells you how likely you are to make it, and what would close the gap.",
               eyebrow="Forecast")

with st.container(border=True, key="card-target-setup"):
    c1, c2, c3 = st.columns([1.4, 1.2, 1])
    scope = c1.segmented_control("Target for", ["Whole shop", "Category", "Product"], default="Whole shop",
                                 required=True, key="tg_scope")
    if scope == "Category":
        item = c2.selectbox("Category", A.categories, key="tg_cat")
        products = A.products.loc[A.products["category"] == item, "product_id"].tolist()
        label = item
    elif scope == "Product":
        item = c2.selectbox("Product", list(A.label_of), format_func=A.label_of.get, key="tg_pid")
        products, label = [item], A.label_of[item]
    else:
        products, label = A.products["product_id"].tolist(), "Whole shop"
        c2.caption(f"All {len(products)} products")
    metric = c3.segmented_control("Measure", list(targets.METRICS), default="revenue", required=True,
                                  format_func=targets.METRICS.get, key="tg_metric")

    c1, c2, c3 = st.columns([1.4, 1.2, 1])
    kind = c1.segmented_control("Period", ["Month", "Quarter", "Custom"], default="Month", required=True, key="tg_kind")
    start, end = targets.default_period(A.last_date, "Quarter" if kind == "Quarter" else "Month")
    if kind == "Custom":
        picked = c2.date_input("Dates", (start.date(), end.date()), key="tg_dates",
                               min_value=(A.last_date - pd.Timedelta(days=365)).date(),
                               max_value=(A.last_date + pd.Timedelta(days=180)).date())
        if len(picked) == 2:
            start, end = pd.Timestamp(picked[0]), pd.Timestamp(picked[1])
    else:
        c2.caption(f"{start:%d %b} – {end:%d %b %Y}")

p = targets.progress(A.fc, products, start, end, metric)
last_year = targets.same_period_last_year(A.fc, products, start, end, metric)
suggested = round((last_year * 1.1 if last_year > 0 else p["projected"] * 1.1) or 1.0, -2 if metric == "revenue" else 0)
def fmt(x: float) -> str:
    return ui.money(x) if metric == "revenue" else f"{ui.num(x)} units"


target = c3.number_input(f"Target ({ui.currency() if metric == 'revenue' else 'units'})", min_value=1.0,
                         value=float(max(suggested, 1.0)), step=100.0 if metric == "revenue" else 10.0,
                         key=f"tg_value_{scope}_{label}_{metric}_{start:%Y%m%d}_{end:%Y%m%d}",
                         help=f"Starts at last year's same period +10% ({fmt(last_year)} last year).")

share = p["projected"] / target if target else 0.0
odds = targets.chance(p, target)
tone = "good" if odds >= 0.7 else "warn" if odds >= 0.35 else "bad"
ui.spotlight(f"{label} · {start:%d %b} – {end:%d %b %Y}", f"{share:.0%}", "of target expected",
             f"Projected <b>{fmt(p['projected'])}</b> vs target <b>{fmt(target)}</b>: "
             + (f"{fmt(p['actual'])} sold in the first {p['days_done']} days + " if p["days_done"] else "the period "
                "hasn't started yet, so this is all forecast: ")
             + (f"{fmt(p['forecast_rest'])} forecast for the {p['days_left']} days left" if p["days_left"]
                else "the period is over"),
             [(f"{odds:.0%} chance of making it", tone), (f"{p['days_left']} days left", ""),
              (f"{(p['projected'] / last_year - 1):+.0%} vs last year" if last_year else "No sales last year", "acc")])

c1, c2 = st.columns([1, 2])
ui.show(ui.gauge(min(share * 100, 150), "Projected vs target", max_value=150, good_below=None), c1)
fig = go.Figure()
cum_act = p["daily_actual"].cumsum()
if len(cum_act):
    fig.add_scatter(x=cum_act.index, y=cum_act.values, name="Sold so far", line=dict(color=ui.BRAND, width=3))
if len(p["daily_forecast"]):
    cum_fc = p["actual"] + p["daily_forecast"].cumsum()
    spread = 1.96 * p["sd"] * (p["daily_forecast"].cumsum() / max(p["forecast_rest"], 1e-9))
    fig.add_scatter(x=cum_fc.index, y=(cum_fc + spread).values, line=dict(width=0), showlegend=False, hoverinfo="skip")
    fig.add_scatter(x=cum_fc.index, y=np.maximum(cum_fc - spread, p["actual"]).values, fill="tonexty",
                    fillcolor=ui.BAND, line=dict(width=0), name="Likely range")
    ui.glow_line(fig, cum_fc.index, cum_fc.values, ui.FC, "Forecast")
fig.add_hline(y=target, line_dash="dot", line_color=ui.GOOD, annotation_text="target")
ui.show(ui.style(fig, f"Running total · {targets.METRICS[metric].lower()}", height=320), c2)

advice = targets.levers(p, target, A.elasticities[0], targets.promo_lift(A.sales, products), ui.currency())
ui.finding_cards([line if line.startswith("**") else f"**What it takes:** {line}" for line in advice])

if len(products) > 1:
    with st.expander(f"By product ({len(products)})", icon=":material/table_rows:"):
        st.dataframe(p["products"], hide_index=True, width="stretch", column_config={
            "product_id": None, "product_name": "Product", "category": "Category",
            "actual": st.column_config.NumberColumn("So far", format="%,.0f"),
            "forecast_rest": st.column_config.NumberColumn("Forecast for the rest", format="%,.0f"),
            "projected": st.column_config.ProgressColumn("Projected", format="%,.0f", min_value=0,
                                                         max_value=float(p["products"]["projected"].max()) or 1)})

if A.user_id is not None:
    key = f"targets:{A.base_key}"
    saved = db.get_setting(A.user_id, key, [])
    if st.button("Save this target", icon=":material/flag:"):
        saved = [t for t in saved if not (t["label"] == label and t["start"] == f"{start:%Y-%m-%d}"
                                         and t["end"] == f"{end:%Y-%m-%d}" and t["metric"] == metric)]
        saved.append({"label": label, "products": products, "start": f"{start:%Y-%m-%d}", "end": f"{end:%Y-%m-%d}",
                      "metric": metric, "target": float(target)})
        db.set_setting(A.user_id, key, saved[-20:])
        st.toast("Target saved.", icon=":material/flag:")
    if saved:
        st.markdown("#### Your targets")
        rows = []
        for t in saved:
            q = targets.progress(A.fc, t["products"], pd.Timestamp(t["start"]), pd.Timestamp(t["end"]), t["metric"])
            rows.append({"Target": t["label"], "Period": f"{t['start']} → {t['end']}",
                         "Measure": targets.METRICS[t["metric"]], "Goal": t["target"], "Projected": q["projected"],
                         "Of target": q["projected"] / t["target"], "Chance": targets.chance(q, t["target"])})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
            "Goal": st.column_config.NumberColumn(format="%,.0f"),
            "Projected": st.column_config.NumberColumn(format="%,.0f"),
            "Of target": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1.5),
            "Chance": st.column_config.NumberColumn(format="percent")})
        if st.button("Clear saved targets", icon=":material/delete_sweep:"):
            db.set_setting(A.user_id, key, [])
            st.rerun()
else:
    st.caption("Sign in to save targets and follow them over time.")
st.download_button("Download target progress by product (CSV)", safe_csv(p["products"]), "target_progress.csv", "text/csv", icon=":material/download:",
                   key="dl_target_progress")
