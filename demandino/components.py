"""Reusable page sections."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import ui
from .analysis import DOW_NAMES, MONTH_NAMES, decompose, detect_anomalies, heatmap_month_dow


def _fmt(x: float, kind: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return {"pct": f"{x * 100:+.0f}%", "num": f"{x:,.1f}", "e": f"{x:.2f}"}[kind]


def pattern_metrics(prof: dict) -> None:
    with st.container(horizontal=True):
        st.metric("Avg daily units (90d)", _fmt(prof["avg_daily_90"], "num"), border=True)
        st.metric("Last 90d vs prior 90d", _fmt(prof["growth_90_vs_prev"], "pct"), border=True)
        st.metric("Year over year (90d)", _fmt(prof["growth_yoy_90"], "pct"), border=True)
        st.metric("Promo lift", _fmt(prof["promo_lift"], "pct"), border=True,
                  help="Average units on promo days vs regular-price days.")
        st.metric("Price elasticity", _fmt(prof["elasticity"], "e"), border=True,
                  help="% change in units for a 1% price change, controlling for promotions, weekday, month and trend.")


def sales_over_time(g: pd.DataFrame, title: str, anomalies: pd.DataFrame | None = None) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=g["date"], y=g["units_sold"], name="Daily", line=dict(color="rgba(100,116,139,0.35)", width=1),
                    hovertemplate="%{y:,.0f}")
    fig.add_scatter(x=g["date"], y=g["units_sold"].rolling(28, min_periods=7).mean(), name="28-day average",
                    line=dict(color=ui.BRAND, width=2.5), hovertemplate="%{y:,.1f}")
    if "promo" in g and g["promo"].any():
        p = g[g["promo"] == 1]
        fig.add_scatter(x=p["date"], y=p["units_sold"], mode="markers", name="Promo day",
                        marker=dict(color=ui.PROMO, size=4), hoverinfo="skip")
    if anomalies is not None and len(anomalies):
        fig.add_scatter(x=anomalies["date"], y=anomalies["units_sold"], mode="markers", name="Unusual day",
                        marker=dict(color=ui.BAD, size=9, symbol="circle-open", line=dict(width=2)),
                        customdata=anomalies["expected"], hovertemplate="%{y:,.0f} (typical ≈ %{customdata:,.0f})")
    ui.style(fig, title, height=380)
    fig.update_xaxes(rangeselector=dict(buttons=[dict(count=3, label="3M", step="month", stepmode="backward"),
                                                 dict(count=1, label="1Y", step="year", stepmode="backward"),
                                                 dict(step="all", label="All")], y=1.0, yanchor="bottom",
                                        x=0, bgcolor="rgba(128,128,128,0.12)"))
    return fig


def show_patterns(g: pd.DataFrame, prof: dict, title: str, show_anomalies: bool = True) -> None:
    g = g.sort_values("date")
    pattern_metrics(prof)
    anomalies = detect_anomalies(g) if show_anomalies and len(g) > 60 else None
    ui.show(sales_over_time(g, f"{title}: sales over time", anomalies))

    c1, c2 = st.columns(2)
    ui.show(ui.bar_index(DOW_NAMES, prof["dow_index"].values - 1, f"Weekday pattern · best {prof['best_dow']}"), c1)
    ui.show(ui.polar_index(MONTH_NAMES, prof["month_index"].values, f"Seasonal radar · peak {prof['peak_month']}"), c2)

    if len(g) >= 120:
        c1, c2 = st.columns(2)
        dec = decompose(g)
        fig = go.Figure()
        fig.add_scatter(x=dec.index, y=dec["actual"], name="Weekly units", line=dict(color="rgba(100,116,139,0.5)"))
        fig.add_scatter(x=dec.index, y=dec["trend"], name="Trend", line=dict(color=ui.FC, width=3))
        ui.show(ui.style(fig, "Weekly sales & underlying trend", height=330), c1)
        w = g.set_index("date")["units_sold"].resample("W").sum().to_frame("units")
        w["year"], w["week"] = w.index.year, w.index.isocalendar().week.to_numpy()
        fig = go.Figure()
        for y, wy in w.groupby("year"):
            fig.add_scatter(x=wy["week"], y=wy["units"], name=str(y), mode="lines")
        ui.style(fig, "Year over year · weekly units", height=330)
        fig.update_xaxes(title_text="Week of year")
        ui.show(fig, c2)

    c1, c2 = st.columns(2)
    hm = heatmap_month_dow(g)
    fig = go.Figure(go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, colorscale=ui.SEQ,
                               hovertemplate="%{y} · %{x}: %{z:,.1f} units/day<extra></extra>",
                               colorbar=dict(thickness=10)))
    ui.style(fig, "When does it sell? (avg units/day)", height=380, legend=False, hover="closest")
    fig.update_yaxes(autorange="reversed")
    ui.show(fig, c1)
    if g["price"].nunique() > 3:
        promo = g["promo"] == 1 if "promo" in g else pd.Series(False, index=g.index)
        fig = go.Figure()
        fig.add_scatter(x=g.loc[~promo, "price"], y=g.loc[~promo, "units_sold"], mode="markers", name="Regular",
                        marker=dict(color=ui.HIST, opacity=0.4, size=5))
        fig.add_scatter(x=g.loc[promo, "price"], y=g.loc[promo, "units_sold"], mode="markers", name="Promo",
                        marker=dict(color=ui.PROMO, opacity=0.6, size=5))
        ui.style(fig, "Price vs units sold", height=380, hover="closest")
        fig.update_xaxes(title_text="Price")
        fig.update_yaxes(title_text="Units")
        ui.show(fig, c2)

    if anomalies is not None and len(anomalies):
        with st.expander(f"{len(anomalies)} unusual days detected (not explained by promotions)",
                         icon=":material/troubleshoot:"):
            st.caption("Days far from the typical level for that time of year and weekday — check for stock-outs, data "
                       "errors, events or competitor activity. Large unexplained drops often mean lost sales.")
            st.dataframe(anomalies.sort_values("date", ascending=False), hide_index=True, width="stretch",
                         column_config={"date": st.column_config.DateColumn("Date"),
                                        "units_sold": st.column_config.NumberColumn("Units"),
                                        "expected": st.column_config.NumberColumn("Typical", format="%.0f"),
                                        "z": st.column_config.NumberColumn("Score", format="%.1f"),
                                        "type": "Type", "price": st.column_config.NumberColumn("Price", format="%.2f")})


def findings(lines: list[str], title: str = "Key findings") -> None:
    st.markdown(f"##### :material/auto_awesome: {title}")
    ui.finding_cards(lines)
