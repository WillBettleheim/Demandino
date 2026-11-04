import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import guide, inbox, ui
from demandino.state import app

A = app()


@st.cache_data(show_spinner="Adding up the shop's forecast…", max_entries=8)
def shop_forecast(data_key: str, days: int) -> pd.DataFrame:
    """Daily units and revenue for the whole shop, with a range for the total of each day."""
    rows = []
    for pid in A.sales["product_id"].unique():
        res = A.fc.forecast_product(pid, days)
        rows.append(pd.DataFrame({"date": res.dates, "units": res.mean, "revenue": res.mean * res.price,
                                  "var": res.daily_sd ** 2}))
    d = pd.concat(rows).groupby("date", as_index=False).sum()
    d["sd"] = np.sqrt(d["var"])
    return d


def change(now: float, before: float) -> str | None:
    return f"{now / before - 1:+.0%} vs prior 30d" if before > 0 else None


s = A.sales
last = A.last_date
recent = s[s["date"] > last - pd.Timedelta(days=30)]
prior = s[(s["date"] <= last - pd.Timedelta(days=30)) & (s["date"] > last - pd.Timedelta(days=60))]
rev_now, rev_before = (recent["units_sold"] * recent["price"]).sum(), (prior["units_sold"] * prior["price"]).sum()
fc = shop_forecast(A.data_key, 60)
next30 = fc.head(30)
items = inbox.gather(A, ui.currency())
urgent = sum(i["priority"] == "now" for i in items)

ui.page_header("Your shop at a glance", f"{s['product_id'].nunique()} products · sales up to {last:%d %b %Y}. "
               "What has happened, what's coming and what needs you.", eyebrow="Home")
guide.welcome()

with st.container(horizontal=True, key="fc_kpis"):
    st.metric("Sales · last 30d", ui.short_money(rev_now), change(rev_now, rev_before), border=True,
              help=f"Exactly {ui.money(rev_now)}.")
    st.metric("Units · last 30d", ui.num(recent["units_sold"].sum()),
              change(recent["units_sold"].sum(), prior["units_sold"].sum()), border=True)
    st.metric("Forecast · next 30d", ui.num(next30["units"].sum()),
              f"{next30['units'].sum() / max(recent['units_sold'].sum(), 1) - 1:+.0%} vs last 30d", border=True,
              help=f"About {ui.money(next30['revenue'].sum())} of sales.")
    st.metric("Needs attention", len(items), f"{urgent} to act on now" if urgent else "nothing urgent",
              delta_color="inverse" if urgent else "off", delta_arrow="off", border=True)

c1, c2 = st.columns([1.7, 1])
# Whole weeks only: the week in progress adds its sales so far to its forecast, so nothing looks like a sudden dip.
sold_daily = s[s["date"] > last - pd.Timedelta(days=189)].groupby("date")["units_sold"].sum()
daily = pd.concat([sold_daily, fc.set_index("date")["units"]])
counts = daily.resample("W").count()
full = daily.resample("W").sum()[counts == 7]
done = full.index <= last
weekly, ahead = full[done], full[~done]
if len(weekly):  # join the two lines
    ahead = pd.concat([weekly.iloc[[-1]], ahead])
fig = go.Figure()
fig.add_scatter(x=weekly.index, y=weekly.values, name="Sold", line=dict(color=ui.HIST, width=2), line_shape="spline",
                hovertemplate="%{y:,.0f} units<extra>sold</extra>")
fig.add_scatter(x=ahead.index, y=ahead.values, name="Forecast", line=dict(color=ui.FC, width=3), line_shape="spline",
                hovertemplate="%{y:,.0f} units<extra>forecast</extra>")
fig.update_yaxes(rangemode="tozero")
ui.show(ui.style(fig, "Whole shop · units per week", height=350), c1)

with c2:
    st.markdown("##### What needs you")
    if not items:
        st.success("Nothing needs attention right now.", icon=":material/check_circle:")
    icons = {"now": ":material/priority_high:", "week": ":material/schedule:", "info": ":material/info:"}
    for i in items[:4]:
        st.markdown(f"{icons[i['priority']]} **{i['title']}**  \n<small>{i['text']}</small>".replace("$", "\\$"),
                    unsafe_allow_html=True)
    st.page_link("views/inbox.py", label=f"See all {len(items)} in the Inbox", icon=":material/inbox:")

c1, c2 = st.columns([1, 1.7])
with c1:
    st.markdown("##### Jump to a product")
    pid = st.selectbox("Product", list(A.label_of), format_func=A.label_of.get, index=None,
                       placeholder="Type a product name…", key="home_pid", label_visibility="collapsed")
    if pid and st.button("Open its forecast", type="primary", icon=":material/online_prediction:", key="home_open"):
        st.session_state["open_product"] = pid
        st.switch_page("views/forecast.py")
    st.markdown("##### Common jobs")
    st.page_link("views/planning.py", label="Plan this week's orders", icon=":material/inventory:")
    st.page_link("views/targets.py", label="Check a sales target", icon=":material/flag:")
    st.page_link("views/whatif.py", label="Try a price change or sale", icon=":material/science:")
    st.page_link("views/help.py", label="Help & glossary", icon=":material/menu_book:")
top = (recent.assign(revenue=recent["units_sold"] * recent["price"]).groupby("product_name")["revenue"].sum()
       .nlargest(8).iloc[::-1])
fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=ui.FC, marker_line_width=0,
                       hovertemplate="%{y}: %{x:,.0f}<extra></extra>"))
ui.show(ui.style(fig, f"Best sellers, last 30 days ({ui.currency()})", height=330, legend=False, hover="closest"), c2)
