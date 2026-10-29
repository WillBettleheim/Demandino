import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import account, db, ui
from demandino.state import app

A = app()
ui.page_header("Forecast tracking",
               "Forecasts you track are compared with the real sales that arrive later, so you can see how well the "
               "model is doing for your business over time.", eyebrow="Model")

uid = account.current_user()
if uid is None:
    st.info("Sign in (sidebar), then use **Track this forecast** on the Forecast page.", icon=":material/lock:")
    st.stop()

logs = db.list_forecasts(uid)
if not logs:
    st.caption("Nothing tracked yet. On the Forecast page, open the Export tab and press **Track this forecast**.")
    st.stop()

rows, series = [], {}
for lg in logs:
    p = lg["payload"]
    f = pd.DataFrame({"date": pd.to_datetime(p["dates"]), "forecast": p["mean"], "lower": p["lower"],
                      "upper": p["upper"]})
    act = A.sales[A.sales["product_id"] == lg["product_id"]].set_index("date")["units_sold"]
    f["actual"] = act.reindex(f["date"]).to_numpy()
    seen = f.dropna(subset=["actual"])
    n = len(seen)
    miss = float((~seen["actual"].between(seen["lower"], seen["upper"])).mean()) if n else np.nan
    wape = float(np.abs(seen["actual"] - seen["forecast"]).sum() / max(seen["actual"].sum(), 1e-9)) if n else np.nan
    rows.append({"id": lg["id"], "product": lg["product_name"], "made": pd.Timestamp(lg["created"], unit="s"),
                 "start": f["date"].min(), "days": len(f), "days_with_actuals": n,
                 "forecast_so_far": float(seen["forecast"].sum()) if n else np.nan,
                 "actual_so_far": float(seen["actual"].sum()) if n else np.nan, "miss_rate": miss, "wape": wape,
                 "same_data": lg["data_key"] == A.data_key})
    series[lg["id"]] = f
t = pd.DataFrame(rows)
done = t[t["days_with_actuals"] > 0]
ui.stat_strip([("Tracked forecasts", str(len(t))), ("With actual sales so far", str(len(done))),
               ("Days outside range", f"{(done['miss_rate'] * done['days_with_actuals']).sum() / max(done['days_with_actuals'].sum(), 1):.1%}" if len(done) else "–"),
               ("Average error (WAPE)", f"{done['wape'].mean():.0%}" if len(done) else "–")])
if not t["same_data"].all():
    st.caption("Some forecasts were made on a different dataset than the one selected now; actuals are matched by "
               "product ID and date in the current data.")

st.dataframe(t.drop(columns=["same_data"]), hide_index=True, width="stretch", column_config={
    "id": None, "product": "Product", "made": st.column_config.DatetimeColumn("Made", format="D MMM YYYY"),
    "start": st.column_config.DateColumn("Starts", format="D MMM YYYY"), "days": "Horizon",
    "days_with_actuals": st.column_config.ProgressColumn("Days with actuals", format="%d", min_value=0,
                                                         max_value=int(t["days"].max())),
    "forecast_so_far": st.column_config.NumberColumn("Forecast so far", format="%,.0f"),
    "actual_so_far": st.column_config.NumberColumn("Actual so far", format="%,.0f"),
    "miss_rate": st.column_config.NumberColumn("Days outside range", format="percent"),
    "wape": st.column_config.NumberColumn("Error (WAPE)", format="percent")})

pick = st.selectbox("Inspect", t["id"].tolist(), format_func=lambda i: f"{t.set_index('id').loc[i, 'product']} · made "
                                                                       f"{t.set_index('id').loc[i, 'made']:%d %b %Y}")
f = series[pick]
fig = go.Figure()
fig.add_scatter(x=f["date"], y=f["upper"], line=dict(width=0), showlegend=False, hoverinfo="skip")
fig.add_scatter(x=f["date"], y=f["lower"], fill="tonexty", fillcolor=ui.BAND, line=dict(width=0), name="Range")
ui.glow_line(fig, f["date"], f["forecast"], ui.FC, "Forecast")
seen = f.dropna(subset=["actual"])
inside = seen["actual"].between(seen["lower"], seen["upper"])
fig.add_scatter(x=seen.loc[inside, "date"], y=seen.loc[inside, "actual"], mode="markers", name="Actual · inside",
                marker=dict(color=ui.LAST_YEAR, size=6))
fig.add_scatter(x=seen.loc[~inside, "date"], y=seen.loc[~inside, "actual"], mode="markers", name="Actual · outside",
                marker=dict(color=ui.BAD, size=8, symbol="x"))
ui.show(ui.style(fig, "Forecast vs what actually sold", height=400))
c1, c2 = st.columns(2)
if c1.button("Stop tracking this forecast", icon=":material/delete:"):
    db.delete_forecast(uid, int(pick))
    st.rerun()
if seen.empty:
    c2.caption("No actual sales for these dates yet — add newer data (upload, or refresh a connected store).")
