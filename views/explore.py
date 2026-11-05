import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import ui
from demandino.analysis import elasticity_effect, product_profile
from demandino.components import findings, show_patterns
from demandino.model import detect_bulk_orders
from demandino.state import app, set_bulk_choices

A = app()

ui.page_header("Explore sales patterns",
               "See how a product sells over time: trend, seasonality, weekday rhythm, promotions, price "
               "sensitivity and unusual days.")

c1, c2 = st.columns([2, 1])
pid = c1.selectbox("Product", list(A.label_of), format_func=A.label_of.get, key="ex_pid")
full = A.history(pid)
span = c2.segmented_control("Period", ["6M", "1Y", "2Y", "All"], default="All", required=True, key="ex_span")
days = {"6M": 182, "1Y": 365, "2Y": 730, "All": None}[span]
g = full if days is None else full[full["date"] > A.last_date - pd.Timedelta(days=days)]
prof = product_profile(g)
name = g["product_name"].iloc[0]

lines = [f"Sells **{prof['avg_daily']:,.1f} units/day** on average ({prof['total_units']:,.0f} units, revenue "
         f"{ui.money(prof['revenue'])} over {prof['days']} days).",
         f"Strongest weekday **{prof['best_dow']}**, weakest **{prof['worst_dow']}**; seasonal peak in "
         f"**{prof['peak_month']}**, low in **{prof['low_month']}**."]
if not np.isnan(prof["growth_90_vs_prev"]):
    trend = "growing" if prof["growth_90_vs_prev"] > 0.05 else "declining" if prof["growth_90_vs_prev"] < -0.05 else "flat"
    lines.append(f"Recent momentum is **{trend}** ({prof['growth_90_vs_prev'] * 100:+.0f}% last 90 days vs the 90 "
                 f"before; part of this can be seasonal).")
if not np.isnan(prof["promo_lift"]):
    lines.append(f"Promotions ran on {prof['promo_share'] * 100:.0f}% of days and lifted sales by "
                 f"**{prof['promo_lift'] * 100:+.0f}%** on average.")
if not np.isnan(prof["elasticity"]):
    lines.append(f"Price sensitivity **{prof['elasticity']:.2f}**: {elasticity_effect(prof['elasticity'])}.")
if not np.isnan(prof["cv"]):
    vol = "very volatile" if prof["cv"] > 0.8 else "moderately volatile" if prof["cv"] > 0.4 else "stable"
    lines.append(f"Daily demand is **{vol}** (coefficient of variation {prof['cv']:.2f}).")
findings(lines, f"What the history says about {ui.md_safe(name)}")

show_patterns(g, prof, name)

st.divider()
st.markdown("#### One-off bulk orders")
st.caption("Days far above normal that no promotion explains — usually a single large order. The model learns from "
           "a normal day instead, and adds this product's average bulk volume back as a steady allowance, so "
           "totals stay right while everyday forecasts get steadier. Untick a day if it was real, repeatable demand.")
keep, add = A.bulk
closed = full["date"].isin(A.fc.closed_dates).to_numpy()
auto, normal = detect_bulk_orders(full["units_sold"].to_numpy(), closed, full["promo"].to_numpy())
day_str = full["date"].dt.strftime("%Y-%m-%d").to_numpy()
cand = auto | np.isin(day_str, [d for p_, d in add if p_ == pid])
table = pd.DataFrame({"date": full["date"].to_numpy()[cand], "units_sold": full["units_sold"].to_numpy()[cand],
                      "normal": normal[cand].round(1)})
table["one_off"] = [((pid, d) not in keep) for d in day_str[cand]]
if table.empty:
    st.success(f"No one-off bulk orders found for {ui.md_safe(name)}.", icon=":material/check_circle:")
edited = st.data_editor(table, key=f"bulk_{A.base_key}_{pid}", hide_index=True, width="stretch",
                        disabled=["date", "units_sold", "normal"], column_config={
                            "date": st.column_config.DateColumn("Day", format="D MMM YYYY"),
                            "units_sold": st.column_config.NumberColumn("Sold", format="%,.0f"),
                            "normal": st.column_config.NumberColumn("A normal day", format="%,.1f"),
                            "one_off": st.column_config.CheckboxColumn("One-off?", help="Untick if this was real, "
                                                                       "repeatable demand the model should learn.")}
                        ) if not table.empty else table
c1, c2, c3 = st.columns([1.2, 1, 1], vertical_alignment="bottom")
extra = c1.date_input("Mark another day as one-off", value=None, min_value=full["date"].min(),
                      max_value=full["date"].max(), key=f"bulk_extra_{pid}")
new_keep = {k for k in keep if k[0] != pid} | {(pid, d.strftime("%Y-%m-%d")) for d, on in
                                               zip(edited["date"], edited["one_off"]) if not on}
new_add = {a for a in add if a[0] != pid} | {(pid, d) for d in day_str[cand] if (pid, d) in add}
if extra is not None:
    new_add.add((pid, pd.Timestamp(extra).strftime("%Y-%m-%d")))
    new_keep.discard((pid, pd.Timestamp(extra).strftime("%Y-%m-%d")))
changed = (frozenset(new_keep), frozenset(new_add)) != (keep, add)
if c2.button("Update the model", type="primary", icon=":material/model_training:", disabled=not changed,
             width="stretch", help="Retrains the forecasting model with your choices (about 20 seconds)."):
    set_bulk_choices(A.base_key, new_keep, new_add)
    st.session_state.pop(f"bulk_extra_{pid}", None)
    st.rerun()
if (keep or add) and c3.button("Reset to automatic", icon=":material/restart_alt:", width="stretch"):
    set_bulk_choices(A.base_key, set(), set())
    st.rerun()
allowance = A.fc.context_for(pid).bulk_allowance
if allowance > 0:
    st.caption(f"Bulk orders add about **{allowance:,.2f} units a day** back to {name}'s forecasts (their average "
               "over the last year).")

st.divider()
st.markdown("#### Compare products")
others = st.multiselect("Products to compare", list(A.label_of), default=[pid], format_func=A.label_of.get,
                        max_selections=6, key="ex_cmp")
if others:
    norm = st.toggle("Index to 100 (compare shapes, not volumes)", value=len(others) > 1, key="ex_norm")
    fig = go.Figure()
    for p in others:
        h = A.history(p).set_index("date")["units_sold"].resample("W").sum()
        if days:
            h = h[h.index > A.last_date - pd.Timedelta(days=days)]
        y = h / h.mean() * 100 if norm else h
        fig.add_scatter(x=y.index, y=y.rolling(4, min_periods=1).mean(), name=A.label_of[p], mode="lines")
    ui.style(fig, "Weekly units (4-week average)" + (" · indexed, average = 100" if norm else ""), height=380)
    ui.show(fig)
