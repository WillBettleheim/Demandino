"""Read-only page for a share link (no data upload, model or sign-in needed)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import db, ui


def render(token: str) -> None:
    share = db.get_share(token)
    if share is None:
        ui.page_header("Link expired", "This share link has expired or was revoked. Ask the sender for a new one.",
                       eyebrow="Shared forecast")
        return
    p = share["payload"]
    s = p["summary"]
    ui.page_header(share["title"], f"Shared read-only forecast · link expires "
                                   f"{pd.Timestamp(share['expires'], unit='s'):%d %b %Y}.", eyebrow="Shared forecast")
    cur = p.get("currency", "$")
    ui.spotlight(f"{s['product']} · {s['horizon_days']} days from {s['forecast_start']}", f"{s['expected_units']:,}",
                 "units expected", f"95% range <b>{s['range_95'][0]:,} – {s['range_95'][1]:,}</b> · revenue "
                 f"<b>{cur}{s['revenue']:,}</b> · gross profit <b>{cur}{s['gross_profit']:,}</b>",
                 [(f"{s['confidence']} confidence", {"High": "good", "Medium": "warn", "Low": "bad"}[s["confidence"]]),
                  (s["category"], "")])
    w = pd.Series(s["weekly_units"])
    fig = go.Figure(go.Bar(x=pd.to_datetime(w.index), y=w.values, marker_color=ui.FC, marker_line_width=0))
    ui.show(ui.style(fig, "Weekly expected units", height=320, legend=False, hover="closest"))
    ui.finding_cards(s["findings"])
    st.caption("Forecasts are estimates, not guarantees. Shared from Demandino.")
