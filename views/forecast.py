import pickle
from statistics import NormalDist

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from demandino import assistant, db, details, ui
from demandino.analysis import DOW_NAMES, product_profile
from demandino.components import show_patterns
from demandino.model import RANGE_LABEL
from demandino.model import demand_class
from demandino.pdf_report import scenario_pdf
from demandino.report import scenario_workbook
from demandino.security import MAX_TEXT, clean_text, safe_csv
from demandino.scenario import (BASES, PROMO_KINDS, ForecastInputs, ScenarioResult, compute, default_events,
                                events_tuple, inputs_to_json, read_history_csv)
from demandino.state import app

A = app()
fc = A.fc


@st.cache_data(show_spinner=False, max_entries=64, hash_funcs={ForecastInputs: lambda i: pickle.dumps(i)})
def run_scenario(data_key: str, inp: ForecastInputs) -> ScenarioResult:
    return compute(fc, A.sales, A.elasticities, A.level_uncertainty, inp)


ui.page_header("Forecast a product",
               "Describe a product and your plans — the model predicts daily demand and explains what drives it. "
               "Results update as you change the inputs.")

if "load_scenario" in st.session_state:
    st.session_state["pinned_scenario"] = st.session_state.pop("load_scenario")
pinned: ForecastInputs | None = st.session_state.get("pinned_scenario")

left, right = st.container(key="fc_layout").columns([1, 2.6], gap="large")

# ======================================================================== inputs
with left:
    with ui.card("product"):
        st.markdown("**:material/inventory_2: Product**")
        mode = st.segmented_control("Product type", ["Existing", "New product"], default="Existing", required=True,
                                    key="fc_mode", label_visibility="collapsed")
        new = mode == "New product"
        history_df, comparable, expected, ramp_days, basis = None, None, None, 0, "auto"
        if not new:
            pid = st.selectbox("Product", list(A.label_of), format_func=A.label_of.get, key="fc_pid")
            hist = A.history(pid)
            name, category = hist["product_name"].iloc[0], hist["category"].iloc[0]
            default_price = float(hist["regular_price"].iloc[-1])
            st.caption(f"{category} · {len(hist):,} days of history · last price {ui.money(default_price, 2)}")
            key_sfx = pid
        else:
            pid = None
            name = clean_text(st.text_input("Product name", "My new product", key="fc_name", max_chars=MAX_TEXT)) or "New product"
            category = st.selectbox("Category", A.categories, key="fc_cat")
            default_price = float(np.median(A.sales[A.sales["category"] == category]["regular_price"]))
            basis = st.radio("Estimate demand from", list(BASES), format_func=BASES.get, key="fc_basis",
                             help="How the model sets this product's baseline demand level.")
            if basis == "similar":
                same_cat = [p for p in A.label_of if A.products.set_index("product_id").loc[p, "category"] == category]
                comparable = st.selectbox("Similar existing product", same_cat + [p for p in A.label_of if p not in same_cat],
                                          format_func=A.label_of.get, key="fc_comp")
            elif basis == "history":
                f = st.file_uploader("Sales history CSV", type=["csv"], key="fc_hist",
                                     help="Columns: date, units_sold, optional price and promo (0/1).")
                if f is not None:
                    try:
                        history_df = read_history_csv(f.getvalue(), default_price)
                        st.caption(f"Loaded {len(history_df)} days ({history_df['date'].min():%d %b} → "
                                   f"{history_df['date'].max():%d %b %Y}).")
                    except Exception as exc:  # noqa: BLE001 - user file
                        st.error(f"Could not read the file: {str(exc)[:200]}")
            expected = st.number_input("Expected daily sales at full demand", min_value=0.0, value=0.0, step=1.0,
                                       key="fc_expected", help="Optional. Leave at 0 to let the model estimate it.") or None
            if st.toggle("Brand-new launch (ramp-up)", value=basis != "history", key="fc_launch"):
                ramp_days = 7 * st.slider("Weeks to reach full demand", 1, 12, 4, key="fc_ramp")
            key_sfx = f"{category}"

    with ui.card("price"):
        st.markdown("**:material/sell: Price & horizon**")
        c1, c2 = st.columns(2)
        price = c1.number_input(f"Regular price ({ui.currency()}) *", min_value=0.01, value=round(default_price, 2),
                                max_value=1_000_000.0, step=1.0, key=f"fc_price_{key_sfx}",
                                help="Required. Your everyday selling price; demand responds to it.")
        cost = c2.number_input(f"Unit cost ({ui.currency()}) *", min_value=0.0, value=round(default_price * 0.55, 2),
                               max_value=1_000_000.0, step=1.0, key=f"fc_cost_{key_sfx}",
                               help="Required for profit and pricing advice. The default (55% of price) is a guess.")
        st.caption(f"Margin {(1 - cost / price) * 100:.0f}%" + (
            f" · price {price / default_price - 1:+.0%} vs current" if not new and abs(price / default_price - 1) > 0.005 else ""))
        horizon = st.select_slider("Horizon (days)", [14, 28, 30, 45, 60, 90, 120, 150, 180], value=90, key="fc_h")
        start = pd.Timestamp(st.date_input("Start date", A.last_date + pd.Timedelta(days=1),
                                           min_value=A.last_date + pd.Timedelta(days=1),
                                           max_value=A.last_date + pd.Timedelta(days=60), key="fc_start"))

    with ui.card("promo"):
        st.markdown("**:material/local_offer: Promotions**")
        promo_kind = st.selectbox("Promotion plan", PROMO_KINDS, key="fc_promo", label_visibility="collapsed")
        discount, every, plen, promo_dates = 0.2, 4, 5, None
        if promo_kind != "None":
            discount = st.slider("Discount on promo days", 5, 50, 20, format="%d%%", key="fc_disc") / 100
        if promo_kind == "Recurring":
            c1, c2 = st.columns(2)
            every = c1.number_input("Every N weeks", 1, 12, 4, key="fc_every")
            plen = c2.number_input("Length (days)", 1, 14, 5, key="fc_plen")
        elif promo_kind == "Specific dates":
            promo_dates = st.date_input("Promo period", (start + pd.Timedelta(days=14), start + pd.Timedelta(days=20)),
                                        key="fc_pdates")

    with ui.card("events"):
        st.markdown("**:material/event: Demand events**")
        st.caption("Known events that change demand: a marketing campaign (+30%), a store closure (−100%), "
                   "a competitor launch (−15%)…")
        events_df = st.data_editor(
            default_events(), num_rows="dynamic", key="fc_events", width="stretch", hide_index=True,
            column_config={"event": st.column_config.TextColumn("Event", default="Campaign"),
                           "start": st.column_config.DateColumn("Start", min_value=start.date()),
                           "end": st.column_config.DateColumn("End", min_value=start.date()),
                           "change": st.column_config.NumberColumn("Change %", min_value=-100, max_value=500,
                                                                   default=20, format="%+d%%")})

    with ui.card("inventory"):
        st.markdown("**:material/warehouse: Inventory**")
        c1, c2 = st.columns(2)
        stock = c1.number_input("Units in stock", 0, 10_000_000, 500, 10, key="fc_stock")
        lead = c2.number_input("Lead time (days)", 1, 180, 14, key="fc_lead")
        review = c1.number_input("Days between orders", 1, 90, 14, key="fc_review")
        service = c2.number_input("Service level %", 80.0, 99.9, 95.0, 0.5, key="fc_service",
                                  help="Chance of not running out during a replenishment cycle.") / 100

    driver_plan = ()
    if A.fc.driver_names:
        with ui.card("drivers"):
            st.markdown("**:material/thermostat: Outside factors**")
            st.caption("Planned average values for the forecast period (defaults: the last 4 weeks).")
            recent = A.fc.drivers.tail(28).mean()
            driver_plan = tuple((n, float(st.number_input(n.replace("_", " ").capitalize(), value=float(recent[n]),
                                                          key=f"fc_drv_{n}"))) for n in A.fc.driver_names)

inputs = ForecastInputs(
    new=new, pid=pid, name=name, category=category, price=float(price), cost=float(cost), horizon=int(horizon),
    start=start, basis=basis, comparable=comparable, expected=expected, ramp_days=ramp_days, history=history_df,
    promo_kind=promo_kind, discount=discount, every=int(every), plen=int(plen),
    promo_dates=tuple(promo_dates) if promo_dates else None, events=events_tuple(events_df), stock=float(stock),
    lead=int(lead), review=int(review), service=float(service), driver_plan=driver_plan)
if pinned is not None:
    inputs = pinned
st.session_state["last_inputs"] = inputs  # lets the Grow demand page analyse this product

# ======================================================================= results
with right:
    if new and basis == "history" and history_df is None:
        st.info("Upload the product's sales history in the panel on the left, or pick another basis.",
                icon=":material/upload_file:")
        st.stop()
    if history_df is not None and history_df["date"].max() >= start:
        st.error("The uploaded history must end before the forecast start date.")
        st.stop()
    if pinned is not None:
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.info(f"Showing the saved scenario **{pinned.name}** (price {ui.money(pinned.price, 2)}, "
                f"{pinned.promo_kind.lower()} promotions). The inputs on the left are not applied.",
                icon=":material/bookmark:")
        if c2.button("Back to my inputs", icon=":material/edit:", width="stretch"):
            st.session_state.pop("pinned_scenario", None)
            st.rerun()
        if pinned.new and pinned.basis == "history" and pinned.history is None:
            st.stop()
    with st.spinner("Forecasting…"):
        r = run_scenario(A.data_key, inputs)
    res, inv = r.res, r.inventory
    st.session_state["last_result"] = r
    dtype = None
    if r.history is not None and len(r.history) >= 60:
        dtype = demand_class(r.history["units_sold"].tail(180).to_numpy())[0]
        if dtype in ("Intermittent", "Lumpy"):
            r.insights = r.insights + [f"**Demand type:** {dtype.lower()} — sales come in irregular bursts, so single "
                                       "days are hard to predict. Plan and order by the week, and keep a higher service "
                                       "level for this product."]

    conf_cls = {"High": "good", "Medium": "warn", "Low": "bad"}[r.confidence]
    per_day = r.total / horizon
    vs_recent = (f" · <b>{(per_day / r.recent_daily - 1) * 100:+.0f}%</b> vs the last 8 weeks"
                 if r.recent_daily else "")
    stock_chip = ((f"Stock runs out {inv['stockout_date']:%d %b} · {inv['days_of_cover']} days", "bad")
                  if inv["stockout_date"] is not None else ("Stock covers the horizon", "good"))
    ui.spotlight(
        f"{r.inputs.name} · next {horizon} days from {start:%d %b %Y}", ui.num(r.total), "units expected",
        f"{RANGE_LABEL} range <b>{ui.num(r.low)} – {ui.num(r.high)}</b> · {ui.num(per_day, 1)} per day{vs_recent}",
        [(f"{r.confidence} confidence", conf_cls), (r.inputs.category, ""), stock_chip]
        + ([(f"{dtype} demand", "warn" if dtype in ("Intermittent", "Lumpy") else "")] if dtype else []) + [
         (f"In testing, {A.backtest['metrics']['Daily miss rate']:.1%} of real days fell outside the range", "acc")],
        count=(r.total, "fc_total", 0))

    weekly = pd.Series(res.mean, index=res.dates).resample("W").sum()
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.metric("Weekly demand", ui.num(weekly.max()), border=True, chart_data=weekly.round(1).tolist(),
                  chart_type="area", delta="peak week", delta_color="off", delta_arrow="off")
        st.metric("Revenue", ui.money(r.revenue), border=True)
        st.metric("Gross profit", ui.money(r.profit), border=True,
                  delta=f"{r.profit / r.revenue * 100:.0f}% margin" if r.revenue else None, delta_color="off",
                  delta_arrow="off")
        st.metric("Order by", "Now" if inv["needs_order_now"] else
                  (f"{inv['reorder_by']:%d %b}" if inv["reorder_by"] is not None else "Not needed"), border=True,
                  delta=f"{ui.num(inv['order_qty'])} units" if inv["reorder_by"] is not None else None,
                  delta_color="off", delta_arrow="off")
        with st.popover("Save scenario", icon=":material/bookmark_add:"):
            sname = clean_text(st.text_input("Scenario name", f"{r.inputs.name} · {r.inputs.promo_kind.lower()} · "
                                             f"{ui.money(r.inputs.price, 2)}"[:MAX_TEXT], max_chars=MAX_TEXT))
            to_ws = st.checkbox("Also save to my workspace", value=True, disabled=A.user_id is None,
                                help="Sign in to keep scenarios after you close the page.")
            if st.button("Save", type="primary", icon=":material/save:", disabled=not sname):
                st.session_state.setdefault("scenarios", {})[sname] = r
                if to_ws and A.user_id is not None:
                    db.save_scenario(A.user_id, sname, inputs_to_json(inputs),
                                     {"units": r.total, "profit": r.profit, "revenue": r.revenue})
                st.toast(f"Saved “{sname}” — compare it in the Scenarios tab.", icon=":material/bookmark:")

    rows = details.checklist(inputs, default_price, round(default_price * 0.55, 2))
    done = details.score(rows)
    icon = {"provided": ":material/check_circle:", "assumed": ":material/error:", "missing": ":material/cancel:",
            "optional": ":material/radio_button_unchecked:"}
    todo = [x for x in rows if x["status"] in ("assumed", "missing")]
    with st.expander(f"Product details · {done:.0%} of the essentials provided"
                     + (f" — add {len(todo)} more for a sharper forecast" if todo else " ✓"),
                     icon=":material/fact_check:", expanded=bool([x for x in rows if x["status"] == "missing"])):
        st.progress(done)
        for x in rows:
            label = {"provided": "provided", "assumed": "**assumed — please enter**", "missing": "**missing**",
                     "optional": "optional"}[x["status"]]
            st.markdown(f"{icon[x['status']]} **{x['detail']}** · {label}  \n<small>{x['why']}</small>",
                        unsafe_allow_html=True)

    ui.finding_cards(r.insights)

    tabs = st.tabs([":material/show_chart: Forecast", ":material/waterfall_chart: Drivers",
                    ":material/calendar_month: Breakdown", ":material/price_change: Price & promo",
                    ":material/inventory: Inventory", ":material/insights: Patterns",
                    ":material/compare_arrows: Scenarios", ":material/download: Export"])

    with tabs[0]:
        show_ly = r.last_year is not None and st.toggle("Compare with the same period last year", value=True)
        ui.show(ui.forecast_chart(res, r.history, None, last_year=r.last_year if show_ly else None))
        st.caption(f"Shaded band: {RANGE_LABEL} of real daily sales should land inside it. Demand basis: {r.method}.")
        c1, c2 = st.columns(2)
        dow = r.frame().groupby(r.frame()["date"].dt.day_of_week)["forecast"].mean()
        fig = go.Figure(go.Bar(x=[DOW_NAMES[i] for i in dow.index], y=dow.values, marker_color=ui.FC,
                               hovertemplate="%{x}: %{y:,.1f}<extra></extra>"))
        ui.show(ui.style(fig, "Average by weekday", height=290, legend=False, hover="closest"), c1)
        fig = go.Figure()
        fig.add_scatter(x=res.dates, y=np.cumsum(res.mean), name="Cumulative forecast", line=dict(color=ui.FC),
                        fill="tozeroy", fillcolor="rgba(249,115,80,0.12)", line_shape="spline")
        fig.add_scatter(x=res.dates, y=np.cumsum(res.mean * res.price), name="Cumulative revenue", yaxis="y2",
                        line=dict(color=ui.BRAND, dash="dot"), line_shape="spline")
        fig.add_hline(y=inputs.stock, line_dash="dash", line_color=ui.HIST, annotation_text="current stock")
        ui.style(fig, "Cumulative units & revenue", height=290,
                 yaxis2=dict(overlaying="y", side="right", showgrid=False, tickprefix=ui.currency(), rangemode="tozero"))
        ui.show(fig, c2)

        c1, c2 = st.columns(2)
        f = r.frame()
        f["week"] = f["date"] - pd.to_timedelta(f["date"].dt.day_of_week, unit="D")
        cal = f.pivot_table(index="week", columns=f["date"].dt.day_of_week, values="forecast", aggfunc="sum")
        cal = cal.reindex(columns=range(7))
        fig = go.Figure(go.Heatmap(z=cal.T.values, x=cal.index, y=DOW_NAMES, colorscale=ui.SEQ, xgap=3, ygap=3,
                                   hovertemplate="Week of %{x|%d %b} · %{y}: %{z:,.1f} units<extra></extra>",
                                   colorbar=dict(thickness=8, len=0.8)))
        ui.style(fig, "Demand calendar · units per day", height=300, legend=False, hover="closest")
        fig.update_yaxes(autorange="reversed")
        ui.show(fig, c1)
        # Chance of selling at least N units over the horizon, from the forecast's total range.
        dist = NormalDist(r.total, max((r.high - r.total) / res.z, 1e-9))
        lo_x = max(0.0, min(r.low * 0.7, inputs.stock * 0.9))
        hi_x = max(r.high * 1.15, min(inputs.stock * 1.1, r.high * 3))
        xs = np.linspace(lo_x, hi_x, 140)
        prob = np.array([1 - dist.cdf(x) for x in xs]) * 100
        cover = dist.cdf(inputs.stock) * 100
        fig = go.Figure()
        ui.glow_line(fig, xs, prob, ui.BRAND, "Chance", hovertemplate="≥ %{x:,.0f} units: %{y:.0f}%<extra></extra>")
        if lo_x <= inputs.stock <= hi_x:
            fig.add_vline(x=inputs.stock, line_dash="dash", line_color=ui.FC, annotation_text="your stock",
                          annotation_position="top right")
        ui.style(fig, f"Chance of selling at least N units in {horizon} days", height=300, legend=False,
                 hover="closest")
        fig.update_yaxes(ticksuffix="%", range=[0, 102])
        fig.update_xaxes(title_text="Units", range=[lo_x, hi_x])
        ui.show(fig, c2)
        c2.caption(f"Your stock of {ui.num(inputs.stock)} units covers the whole horizon's demand with about "
                   f"**{cover:.0f}%** probability.")

    with tabs[1]:
        labels = [d[0] for d in r.drivers] + ["Forecast"]
        values = [d[1] for d in r.drivers]
        fig = go.Figure(go.Waterfall(
            x=labels, y=values + [r.total], measure=["absolute"] + ["relative"] * (len(values) - 1) + ["total"],
            connector=dict(line=dict(color="rgba(128,128,128,0.4)")), increasing=dict(marker_color=ui.GOOD),
            decreasing=dict(marker_color=ui.BAD), totals=dict(marker_color=ui.FC),
            text=[f"{v:+,.0f}" if i else f"{v:,.0f}" for i, v in enumerate(values)] + [f"{r.total:,.0f}"],
            textposition="outside", hovertemplate="%{x}: %{y:,.0f}<extra></extra>"))
        ui.style(fig, f"From recent run-rate to the {horizon}-day forecast (units)", height=420, legend=False,
                 hover="closest")
        ui.show(fig)
        st.caption("**Recent run-rate**: the last 56 days' average × horizon. **Season, trend & calendar**: what the "
                   "ML model adds for time of year, momentum, weekdays and holidays. **Price change**: effect of your "
                   f"price vs the current one (elasticity {r.elasticity:.2f}). **Promotions**, **Events** and "
                   "**Launch ramp-up** follow your plan.")
        st.markdown("##### :material/help: Why this day?")
        day = st.select_slider("Pick a day", options=list(range(len(res.dates))), value=int(np.argmax(res.mean)),
                               format_func=lambda i: f"{res.dates[i]:%a %d %b}", key="fc_explain_day")
        target, avg, parts = A.fc.explain_day(day=day, **r.explain_args)
        k = float(res.mean[day]) / target if target > 0 else 1.0  # price, events and ramp-up applied after the model
        parts = [(n, v * k) for n, v in parts]
        fig = go.Figure(go.Waterfall(
            orientation="h", y=["Average day"] + [n for n, _ in parts] + [f"{res.dates[day]:%a %d %b}"],
            x=[avg * k] + [v for _, v in parts] + [float(res.mean[day])],
            measure=["absolute"] + ["relative"] * len(parts) + ["total"],
            increasing=dict(marker_color=ui.GOOD), decreasing=dict(marker_color=ui.BAD), totals=dict(marker_color=ui.FC),
            connector=dict(line=dict(color="rgba(124,139,154,0.4)")),
            text=[f"{avg * k:,.1f}"] + [f"{v:+,.1f}" for _, v in parts] + [f"{res.mean[day]:,.1f}"],
            textposition="outside", hovertemplate="%{y}: %{x:,.1f}<extra></extra>"))
        ui.style(fig, f"{res.dates[day]:%A %d %B}: {res.mean[day]:,.1f} units vs {avg * k:,.1f} on an average day",
                 height=380, legend=False, hover="closest")
        fig.update_yaxes(autorange="reversed")
        ui.show(fig)
        st.caption("Each bar is how much that kind of input moves this day away from an average day in the forecast "
                   "(the model is re-run with that input taken from every other day).")
        imp = A.backtest["importance"].head(10)[::-1]
        fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color=ui.BRAND,
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        ui.show(ui.style(fig, "What the model relies on most (all products)", height=360, legend=False, hover="closest"))

    with tabs[2]:
        freq = st.segmented_control("Group by", ["Weekly", "Monthly"], default="Weekly", required=True, key="fc_freq")
        f = r.frame().set_index("date")
        agg = f.resample("W-SUN" if freq == "Weekly" else "MS").agg(
            units=("forecast", "sum"), low=("lower", "sum"), high=("upper", "sum"), revenue=("revenue", "sum"),
            promo_days=("promo", "sum"), days=("forecast", "size"))
        if r.last_year is not None:
            agg["last_year"] = f["last_year_actual"].resample("W-SUN" if freq == "Weekly" else "MS").sum()
        fig = go.Figure()
        fig.add_bar(x=agg.index, y=agg["units"], name="Forecast", marker_color=ui.FC,
                    error_y=dict(type="data", symmetric=False, array=(agg["high"] - agg["units"]) * 0.45,
                                 arrayminus=(agg["units"] - agg["low"]) * 0.45, color="rgba(128,128,128,0.6)"))
        if "last_year" in agg:
            fig.add_scatter(x=agg.index, y=agg["last_year"], name="Last year", mode="lines+markers",
                            line=dict(color=ui.LAST_YEAR, dash="dash"))
        ui.show(ui.style(fig, f"{freq} expected units", height=330))
        show = agg.drop(columns=["low", "high"]).reset_index()
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "date": st.column_config.DateColumn("Week ending" if freq == "Weekly" else "Month",
                                                format="D MMM YYYY" if freq == "Weekly" else "MMM YYYY"),
            "units": st.column_config.ProgressColumn("Units", format="%.0f", min_value=0,
                                                     max_value=float(agg["units"].max()) or 1),
            "revenue": st.column_config.NumberColumn(f"Revenue ({ui.currency()})", format="%,.0f"),
            "promo_days": "Promo days", "days": "Days",
            "last_year": st.column_config.NumberColumn("Last year", format="%.0f")})

    with tabs[3]:
        sens = r.sensitivity
        best = sens.loc[sens["profit"].idxmax()]
        c1, c2 = st.columns(2)
        fig = go.Figure()
        fig.add_scatter(x=sens["price"], y=sens["units"], name="Units", line=dict(color=ui.HIST))
        fig.add_scatter(x=sens["price"], y=sens["profit"], name="Profit", yaxis="y2", line=dict(color=ui.GOOD, width=3))
        fig.add_scatter(x=[best["price"]], y=[best["profit"]], yaxis="y2", mode="markers", name="Max profit",
                        marker=dict(color=ui.GOOD, size=13, symbol="diamond"))
        fig.add_vline(x=price, line_dash="dash", line_color="gray", annotation_text="your price")
        ui.style(fig, f"Regular price what-if ({horizon}-day totals)", height=380,
                 yaxis2=dict(title="Profit", overlaying="y", side="right", showgrid=False))
        fig.update_xaxes(title_text=f"Regular price ({ui.currency()})")
        fig.update_yaxes(title_text="Units", selector=dict(overlaying=None))
        ui.show(fig, c1)
        c1.caption(f"Price elasticity {r.elasticity:.2f}: a 1% price rise changes units by {r.elasticity:.2f}%. "
                   "Far from today's price the estimate is less certain.")
        with c2:
            if r.promo_effect:
                fig = go.Figure()
                fig.add_scatter(x=r.base.dates, y=r.base.mean, name="Without promotions", line=dict(color=ui.HIST))
                fig.add_scatter(x=res.dates, y=res.mean, name="With your plan", line=dict(color=ui.FC))
                ui.show(ui.style(fig, "Promotion impact on daily demand", height=300))
                pe = r.promo_effect
                with st.container(horizontal=True):
                    st.metric("Extra units", f"{pe['extra_units']:+,.0f}", border=True)
                    st.metric("Lift on promo days", ui.pct(pe["lift"]), border=True)
                    st.metric("Profit impact", ui.money(pe["extra_profit"]), border=True,
                              delta="worth it" if pe["extra_profit"] > 0 else "costs margin",
                              delta_color="normal" if pe["extra_profit"] > 0 else "inverse")
            else:
                st.markdown("**Promotion ideas** — 5-day promos every 4 weeks")
                rows = [{"Plan": "No promotions (current)", "Units": r.total, "Profit": r.profit}]
                for d in (0.1, 0.2, 0.3):
                    alt = run_scenario(A.data_key, ForecastInputs(**{**inputs.__dict__, "promo_kind": "Recurring",
                                                                      "discount": d, "every": 4, "plen": 5}))
                    rows.append({"Plan": f"{d:.0%} off", "Units": alt.total, "Profit": alt.profit})
                t = pd.DataFrame(rows)
                t["Profit vs now"] = t["Profit"] - r.profit
                st.dataframe(t, hide_index=True, width="stretch", column_config={
                    "Units": st.column_config.NumberColumn(format="%,.0f"),
                    "Profit": st.column_config.NumberColumn(f"Profit ({ui.currency()})", format="%,.0f"),
                    "Profit vs now": st.column_config.NumberColumn(format="%+,.0f")})
                st.caption("Choose a plan in **Promotions** on the left to see it in detail.")
        st.dataframe(sens, hide_index=True, width="stretch", column_config={
            "price": st.column_config.NumberColumn("Regular price", format="%.2f"),
            "price_change_%": st.column_config.NumberColumn("Change", format="%+.0f%%"),
            "units": st.column_config.NumberColumn("Units", format="%,.0f"),
            "revenue": st.column_config.NumberColumn("Revenue", format="%,.0f"),
            "profit": st.column_config.ProgressColumn("Profit", format="%,.0f", min_value=float(min(0, sens["profit"].min())),
                                                      max_value=float(sens["profit"].max()) or 1)})

    with tabs[4]:
        with st.container(horizontal=True):
            st.metric("Safety stock", ui.num(inv["safety_stock"]), border=True)
            st.metric("Reorder point", ui.num(inv["reorder_point"]), border=True,
                      help="Order when stock on hand falls to this level.")
            st.metric("Order by", "Now" if inv["needs_order_now"] else
                      (f"{inv['reorder_by']:%d %b %Y}" if inv["reorder_by"] is not None else "Not needed"), border=True)
            st.metric("Order quantity", ui.num(inv["order_qty"]), border=True,
                      help=f"Covers lead time + review period ({inputs.lead + inputs.review} days) at "
                           f"{inputs.service:.1%} service, minus stock still on hand at the order date.")
        on_hand = inputs.stock - np.cumsum(res.mean)
        fig = go.Figure()
        fig.add_scatter(x=res.dates, y=on_hand, name="Expected demand", line=dict(color=ui.BRAND, width=2.5))
        fig.add_scatter(x=res.dates, y=inputs.stock - np.cumsum(res.upper), name="High demand (p90 days)",
                        line=dict(color=ui.BAD, dash="dot"))
        fig.add_hrect(y0=0, y1=inv["safety_stock"], fillcolor="rgba(220,38,38,0.08)", line_width=0,
                      annotation_text="safety stock", annotation_position="top left")
        fig.add_hline(y=inv["reorder_point"], line_dash="dash", line_color=ui.FC, annotation_text="reorder point")
        if inv["reorder_by"] is not None:
            fig.add_vline(x=inv["reorder_by"], line_dash="dot", line_color=ui.FC)
        ui.style(fig, "Projected stock if no new orders arrive", height=380)
        ui.show(fig)

    with tabs[5]:
        if r.profile is not None:
            show_patterns(r.history, r.profile, r.inputs.name)
        else:
            st.info("No sales history for this product yet — showing its category's average pattern.",
                    icon=":material/category:")
            cat = A.sales[A.sales["category"] == inputs.category].groupby("date", as_index=False).agg(
                units_sold=("units_sold", "mean"), price=("price", "mean"), promo=("promo", "mean"))
            cat["promo"] = (cat["promo"] > 0.5).astype(int)
            show_patterns(cat, product_profile(cat), f"{inputs.category} (category average)", show_anomalies=False)

    with tabs[6]:
        saved: dict[str, ScenarioResult] = st.session_state.get("scenarios", {})
        if not saved:
            st.info("Save scenarios with **Save scenario** (top of the results) to compare prices, promotions or "
                    "products side by side.", icon=":material/bookmark_add:")
        else:
            rows = []
            fig = go.Figure()
            for nm, s in saved.items():
                rows.append({"Scenario": nm, "Product": s.inputs.name, "Price": s.inputs.price,
                             "Promotions": s.inputs.promo_kind, "Days": s.inputs.horizon, "Units": s.total,
                             "Low": s.low, "High": s.high, "Revenue": s.revenue, "Profit": s.profit,
                             "Stock-out": s.inventory["stockout_date"], "Confidence": s.confidence})
                fig.add_scatter(x=s.res.dates, y=pd.Series(s.res.mean).rolling(7, min_periods=1).mean(), name=nm)
            fig.add_scatter(x=res.dates, y=pd.Series(res.mean).rolling(7, min_periods=1).mean(), name="Current (unsaved)",
                            line=dict(color="gray", dash="dot"))
            ui.show(ui.style(fig, "Scenarios · 7-day average daily demand", height=360))
            table = pd.DataFrame(rows)
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                "Price": st.column_config.NumberColumn(format="%.2f"),
                **{c: st.column_config.NumberColumn(format="%,.0f") for c in ("Units", "Low", "High", "Revenue")},
                "Profit": st.column_config.ProgressColumn(format="%,.0f", min_value=0,
                                                          max_value=float(table["Profit"].max()) or 1),
                "Stock-out": st.column_config.DateColumn()})
            c1, c2 = st.columns(2)
            drop = c1.selectbox("Remove a scenario", list(saved), index=None, placeholder="Choose…")
            if drop and c1.button("Remove", icon=":material/delete:"):
                saved.pop(drop)
                st.rerun()
            if c2.button("Clear all scenarios", icon=":material/clear_all:"):
                st.session_state["scenarios"] = {}
                st.rerun()

    with tabs[7]:
        out = r.frame()
        out.insert(0, "product", inputs.name)
        st.markdown("Download this forecast to share it or load it into your planning tools.")
        c1, c2 = st.columns(2)
        c1.download_button("Excel report (summary, findings, daily/weekly/monthly, drivers, pricing)",
                           scenario_workbook(r, ui.currency()), f"forecast_{inputs.name}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                           icon=":material/table_view:", type="primary", width="stretch")
        c2.download_button("Daily forecast (CSV)", safe_csv(out), f"forecast_{inputs.name}.csv", "text/csv",
                           icon=":material/download:", width="stretch")
        c1.download_button("PDF report", scenario_pdf(r, ui.currency()), f"forecast_{inputs.name}.pdf",
                           "application/pdf", icon=":material/picture_as_pdf:", width="stretch")
        if A.user_id is not None:
            if c2.button("Track this forecast", icon=":material/track_changes:", width="stretch",
                         help="Compare it with real sales as they arrive (Forecast tracking page)."):
                db.log_forecast(A.user_id, A.data_key, inputs.pid or f"new:{inputs.name}", inputs.name,
                                {"dates": [d.strftime("%Y-%m-%d") for d in res.dates], "mean": res.mean.round(3).tolist(),
                                 "lower": res.lower.round(3).tolist(), "upper": res.upper.round(3).tolist()})
                st.toast("Tracking — see the Forecast tracking page.", icon=":material/check:")
            if st.button("Create a share link (valid 7 days)", icon=":material/link:"):
                token = db.create_share(A.user_id, f"{inputs.name} · {inputs.horizon}-day forecast",
                                        {"summary": assistant.forecast_summary(r, ui.currency()),
                                         "currency": ui.currency()})
                base = str(st.context.url).split("?")[0].rsplit("/", 1)[0] if st.context.url else ""
                st.code(f"{base}/?share={token}", language=None)
                st.caption("Anyone with this link can view a read-only summary. Revoke it on the Workspace page.")
        else:
            st.caption("Sign in to track this forecast against real sales or create a share link.")
        st.dataframe(out, hide_index=True, width="stretch", column_config={
            "date": st.column_config.DateColumn("Date"),
            **{c: st.column_config.NumberColumn(format="%.1f") for c in ("forecast", "lower", "upper",
                                                                          "revenue", "last_year_actual")},
            "price": st.column_config.NumberColumn(format="%.2f")})
