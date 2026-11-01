import plotly.graph_objects as go
import streamlit as st

from demandino import monitor, ui
from demandino.security import safe_csv
from demandino.state import app

A = app()

ui.page_header("Demand alerts",
               "Products whose recent sales keep landing outside what the model expected. Something has changed for "
               "them — check stock, price and listings before it costs you.", eyebrow="Forecast")
weeks = st.segmented_control("Look back", [4, 6, 8], default=4, required=True, format_func=lambda w: f"{w} weeks",
                             key="al_weeks", help="How many recent weeks to check. The model is retrained without "
                                                 "them, so it can't have seen these sales.")
rep, cutoff = A.drift(int(weeks))
if rep.empty:
    st.info("Not enough history to check recent weeks yet.", icon=":material/hourglass_empty:")
    st.stop()

flagged = rep[rep["flagged"]]
up, down = int((flagged["direction"] == "Above").sum()), int((flagged["direction"] == "Below").sum())
ui.spotlight(f"Last {weeks} weeks · since {cutoff:%d %b %Y}", str(len(flagged)),
             "product" + ("" if len(flagged) == 1 else "s") + " need a look",
             f"<b>{up}</b> selling faster and <b>{down}</b> slower than the forecast range · "
             f"{len(rep) - len(flagged)} on track",
             [("All on track", "good")] if flagged.empty else
             [(f"{up} selling faster", "warn")] * bool(up) + [(f"{down} selling slower", "bad")] * bool(down),
             count=(len(flagged), f"alerts_{weeks}", 0))
st.caption(f"A product is flagged when weekly sales sat outside the forecast range on the same side for at least "
           f"{monitor.MIN_WEEKS} weeks, or the whole period did.")

for r in flagged.head(12).itertuples():
    with ui.card(f"alert-{r.product_id}"):
        c1, c2 = st.columns([1.3, 1])
        icon = ":material/trending_up:" if r.direction == "Above" else ":material/trending_down:"
        c1.markdown(f"**{icon} {r.product_name}** · {r.category}")
        c1.markdown(f"Sold **{ui.num(r.actual)}** vs **{ui.num(r.expected)}** expected "
                    f"(range {ui.num(r.low)}–{ui.num(r.high)}) · **{r.change:+.0%}** · outside the range "
                    f"{r.weeks_outside} of {r.weeks} weeks")
        c1.caption(monitor.advice(r._asdict()))
        if c1.button("Open in Forecast", key=f"al_open_{r.product_id}", icon=":material/online_prediction:"):
            st.session_state["open_product"] = r.product_id  # picked up by the Forecast page
            st.switch_page("views/forecast.py")
        fig = go.Figure(go.Bar(x=[f"Wk {i + 1}" for i in range(len(r.weekly_change))],
                               y=[v * 100 for v in r.weekly_change],
                               marker_color=[ui.FC if v >= 0 else ui.BAD for v in r.weekly_change],
                               hovertemplate="%{x}: %{y:+.0f}% vs forecast<extra></extra>"))
        ui.style(fig, None, height=150, legend=False, hover="closest")
        fig.update_layout(margin=dict(l=4, r=4, t=8, b=4), yaxis_ticksuffix="%")
        ui.show(fig, c2)

with st.expander(f"All products ({len(rep)})", icon=":material/table_rows:", expanded=flagged.empty):
    st.dataframe(rep[["product_name", "category", "direction", "expected", "actual", "change", "weeks_outside"]],
                 hide_index=True, width="stretch", column_config={
                     "product_name": "Product", "category": "Category", "direction": "Status",
                     "expected": st.column_config.NumberColumn("Expected", format="%,.0f"),
                     "actual": st.column_config.NumberColumn("Sold", format="%,.0f"),
                     "change": st.column_config.NumberColumn("Vs expected", format="percent"),
                     "weeks_outside": st.column_config.NumberColumn("Weeks outside range", format="%d")})
st.caption("These alerts are also added to the weekly digest on **Connections & alerts**.")
st.download_button("Download demand alerts (CSV)", safe_csv(rep.drop(columns=['weekly_change'])), "demand_alerts.csv", "text/csv", icon=":material/download:",
                   key="dl_demand_alerts")
