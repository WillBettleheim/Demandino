"""Shared look & feel: cinematic dark theme, glass cards, hero blocks, formatting and chart styling."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .model import RANGE_LABEL, ForecastResult

BRAND = "#8B7CFF"
HIST = "#94A3B8"
FC = "#FF8A3D"
BAND = "rgba(255,138,61,0.14)"
PROMO = "#F472B6"
GOOD = "#34D399"
BAD = "#F87171"
WARN = "#FACC15"
LAST_YEAR = "#38BDF8"
PALETTE = ["#8B7CFF", "#FF8A3D", "#38BDF8", "#34D399", "#F472B6", "#FACC15", "#A78BFA", "#94A3B8"]

CSS = """
<style>
:root {
  --dm-violet: #8B7CFF; --dm-amber: #FF8A3D; --dm-cyan: #38BDF8;
  --dm-glass: rgba(255,255,255,0.035);
  --dm-edge: rgba(139,124,255,0.20);
  --dm-glow: 0 0 0 1px rgba(139,124,255,0.10), 0 18px 50px -24px rgba(139,124,255,0.45);
}
/* Aurora backdrop: soft coloured light behind everything (static: animating a full-screen layer
   causes repaint glitches in some browsers). */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(40vw 30vh at 18% 4%, rgba(139,124,255,0.18), transparent 70%),
    radial-gradient(35vw 28vh at 88% 0%, rgba(255,138,61,0.12), transparent 70%),
    radial-gradient(45vw 35vh at 60% 100%, rgba(56,189,248,0.07), transparent 70%) !important;
  background-attachment: fixed !important;
}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1440px;}

/* Entrance: content rises in softly when a page opens. */
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"],
[data-testid="stMainBlockContainer"] [data-testid="stLayoutWrapper"] {animation: dm-rise .55s cubic-bezier(.2,.7,.2,1) both;}
@keyframes dm-rise { from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: none;} }

h1, h2, h3, h4 {letter-spacing: -0.02em;}

/* Hero header */
.dm-hero {position: relative; overflow: hidden; padding: 1.6rem 1.8rem 1.5rem; margin-bottom: .4rem;
  border-radius: 1.2rem; border: 1px solid var(--dm-edge);
  background: linear-gradient(135deg, rgba(139,124,255,0.16), rgba(255,138,61,0.07) 55%, rgba(56,189,248,0.06));
  box-shadow: var(--dm-glow);}
.dm-hero::after {content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--dm-violet), var(--dm-amber), transparent);
  background-size: 200% 100%; animation: dm-shimmer 6s linear infinite;}
@keyframes dm-shimmer { from {background-position: 200% 0;} to {background-position: -200% 0;} }
.dm-eyebrow {font-size: .72rem; font-weight: 600; letter-spacing: .16em; text-transform: uppercase;
  color: var(--dm-violet); margin-bottom: .35rem;}
.dm-hero h1 {font-family: Sora, Inter, sans-serif; font-size: 2.15rem; font-weight: 750; margin: 0; padding: 0;
  line-height: 1.15; background: linear-gradient(90deg, #FFFFFF 10%, #CFC8FF 55%, #FFC59A);
  -webkit-background-clip: text; background-clip: text; color: transparent;}
.dm-hero p {margin: .55rem 0 0; opacity: .72; font-size: 1rem; max-width: 62rem;}

/* Spotlight: the headline number */
.dm-spot {position: relative; padding: 1.4rem 1.6rem; border-radius: 1.2rem; overflow: hidden;
  border: 1px solid rgba(255,138,61,0.25); box-shadow: 0 30px 80px -40px rgba(255,138,61,0.55);
  background: radial-gradient(120% 140% at 0% 0%, rgba(255,138,61,0.16), transparent 55%),
              radial-gradient(120% 140% at 100% 100%, rgba(139,124,255,0.18), transparent 60%), var(--dm-glass);}
.dm-spot .dm-big {font-family: Sora, Inter, sans-serif; font-size: 3.4rem; font-weight: 800; line-height: 1.05;
  letter-spacing: -0.04em; animation: dm-pop .7s cubic-bezier(.2,.8,.2,1) both;}
.dm-spot .dm-big span {background: linear-gradient(90deg, #FFE0C7, #FF8A3D 45%, #C9BFFF);
  -webkit-background-clip: text; background-clip: text; color: transparent;}
.dm-spot .dm-big small {font-size: 1.05rem; font-weight: 600; letter-spacing: 0; opacity: .65; margin-left: .45rem;}
@keyframes dm-pop { from {opacity: 0; transform: scale(.96) translateY(6px); filter: blur(4px);} to {opacity: 1; transform: none; filter: none;} }
.dm-spot .dm-line {margin-top: .7rem; opacity: .82; font-size: .95rem;}
.dm-chips {display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .9rem;}
.dm-chip {font-size: .78rem; font-weight: 600; padding: .28rem .7rem; border-radius: 999px;
  border: 1px solid rgba(148,163,184,0.25); background: rgba(148,163,184,0.08);}
.dm-chip.good {color: #34D399; border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.08);}
.dm-chip.warn {color: #EAB308; border-color: rgba(250,204,21,.35); background: rgba(250,204,21,.08);}
.dm-chip.bad {color: #F87171; border-color: rgba(248,113,113,.4); background: rgba(248,113,113,.09);}
.dm-chip.violet {color: #A99DFF; border-color: rgba(139,124,255,.4); background: rgba(139,124,255,.10);}

/* Glass cards (st.container keyed "card-…") and metrics */
[class*="st-key-card-"] {background: var(--dm-glass) !important; border: 1px solid var(--dm-edge) !important;
  border-radius: 1.05rem !important; box-shadow: var(--dm-glow);
  transition: border-color .25s ease, box-shadow .25s ease;}
[class*="st-key-card-"]:hover {border-color: rgba(139,124,255,0.38) !important;
  box-shadow: 0 0 0 1px rgba(139,124,255,0.18), 0 24px 60px -26px rgba(139,124,255,0.6);}
div[data-testid="stMetric"] {background: var(--dm-glass); border-color: var(--dm-edge) !important; border-radius: 1rem;
  padding: .8rem 1rem; position: relative; overflow: hidden;
  transition: transform .2s ease, box-shadow .2s ease;}
div[data-testid="stMetric"]::before {content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
  background: linear-gradient(90deg, var(--dm-violet), var(--dm-amber)); opacity: .7;}
div[data-testid="stMetric"]:hover {transform: translateY(-2px); box-shadow: 0 18px 40px -24px rgba(139,124,255,.6);}
div[data-testid="stMetricLabel"] p {font-size: .74rem; letter-spacing: .08em; text-transform: uppercase; opacity: .7;}
div[data-testid="stMetricValue"] {font-family: Sora, Inter, sans-serif; letter-spacing: -0.02em;}

/* Tabs as glowing pills */
.stTabs [data-baseweb="tab-list"] {gap: .3rem; flex-wrap: wrap; padding: .3rem;
  background: rgba(148,163,184,0.05); border: 1px solid rgba(148,163,184,0.12); border-radius: .9rem;}
.stTabs [data-baseweb="tab"] {padding: .45rem .9rem; border-radius: .65rem; height: auto;}
.stTabs [aria-selected="true"] {background: linear-gradient(135deg, rgba(139,124,255,.28), rgba(255,138,61,.16));
  box-shadow: inset 0 0 0 1px rgba(139,124,255,.35);}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

/* Buttons */
[data-testid="stBaseButton-primary"] {background: linear-gradient(135deg, #8B7CFF, #FF8A3D) !important;
  border: none !important; color: #fff !important; box-shadow: 0 10px 30px -12px rgba(255,138,61,.7);
  transition: transform .15s ease, filter .15s ease;}
[data-testid="stBaseButton-primary"]:hover {transform: translateY(-1px); filter: brightness(1.08);}

/* Sidebar */
[data-testid="stSidebar"] {border-right: 1px solid rgba(139,124,255,0.12);}
[data-testid="stSidebarContent"] {background: linear-gradient(180deg, rgba(139,124,255,0.08), transparent 45%);}
.dm-brand {font-family: Sora, Inter, sans-serif; font-weight: 800; font-size: 1.45rem; letter-spacing: -0.03em;}
.dm-brand span {background: linear-gradient(90deg, #FF8A3D, #FFC59A); -webkit-background-clip: text;
  background-clip: text; color: transparent;}
.dm-sub {opacity: .6; font-size: .8rem; margin-top: -.1rem;}

/* Findings grid */
.dm-findings {display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: .7rem;}
.dm-finding {padding: .85rem 1rem; border-radius: .9rem; background: var(--dm-glass);
  border: 1px solid rgba(148,163,184,0.14); font-size: .9rem; line-height: 1.45; animation: dm-rise .6s both;}
.dm-finding .dm-ft {display: block; font-size: .68rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--dm-amber); margin-bottom: .3rem; font-weight: 700;}
::-webkit-scrollbar {width: 10px; height: 10px;}
::-webkit-scrollbar-thumb {background: rgba(139,124,255,.25); border-radius: 10px;}

/* Forecast page: input panel + results wrap onto separate rows when space is tight. */
.st-key-fc_layout > div > [data-testid="stHorizontalBlock"] {flex-wrap: wrap; row-gap: 1rem;}
.st-key-fc_layout > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child
  {flex: 1 1 300px !important; width: auto !important; min-width: 0;}
.st-key-fc_layout > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child
  {flex: 3 1 560px !important; width: auto !important; min-width: 0;}

@media (prefers-reduced-motion: reduce) { *, *::before, *::after {animation: none !important; transition: none !important;} }
@media (max-width: 640px) {.dm-hero h1 {font-size: 1.55rem;} .dm-spot .dm-big {font-size: 2.5rem;}
  .block-container {padding-top: 1.2rem;}}
</style>
"""


def inject_css() -> None:
    st.html(CSS)


def page_header(title: str, subtitle: str, eyebrow: str = "Demandino") -> None:
    st.html(f'<div class="dm-hero"><div class="dm-eyebrow">{html.escape(eyebrow)}</div>'
            f'<h1>{html.escape(title)}</h1><p>{subtitle}</p></div>')


def card(name: str):
    """A glass card container (styled through its key)."""
    return st.container(border=True, key=f"card-{name}")


def spotlight(eyebrow: str, value: str, unit: str, line: str, chips: list[tuple[str, str]]) -> None:
    """The headline number of a page, large and glowing, with status chips underneath."""
    chip_html = "".join(f'<span class="dm-chip {cls}">{html.escape(text)}</span>' for text, cls in chips)
    st.html(f'<div class="dm-spot"><div class="dm-eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="dm-big"><span>{html.escape(value)}</span><small>{html.escape(unit)}</small></div>'
            f'<div class="dm-line">{line}</div><div class="dm-chips">{chip_html}</div></div>')


def _md_bold(text: str) -> str:
    parts = html.escape(text).split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def finding_cards(lines: list[str]) -> None:
    """Findings as a grid of cards; a leading '**Title:**' becomes the card's label."""
    cards = []
    for i, line in enumerate(lines):
        title, body = "", line
        if line.startswith("**") and ":**" in line[:60]:
            title, body = line[2:line.index(":**")], line[line.index(":**") + 3:].strip()
        cards.append(f'<div class="dm-finding" style="animation-delay:{i * 45}ms">'
                     f'<span class="dm-ft">{html.escape(title)}</span>{_md_bold(body)}</div>')
    st.html(f'<div class="dm-findings">{"".join(cards)}</div>')


# ------------------------------------------------------------- formatting
def currency() -> str:
    return st.session_state.get("currency", "$")


def money(x: float, decimals: int = 0) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}{currency()}{abs(x):,.{decimals}f}"


def num(x: float, decimals: int = 0) -> str:
    return f"{x:,.{decimals}f}"


def pct(x: float, signed: bool = True) -> str:
    return f"{x * 100:+.0f}%" if signed else f"{x * 100:.0f}%"


# ----------------------------------------------------------------- charts
GRID = "rgba(148,163,184,0.10)"


def style(fig: go.Figure, title: str | None = None, height: int = 360, legend: bool = True,
          hover: str = "x unified", **layout) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family="Sora, Inter, sans-serif")) if title else None,
        height=height, margin=dict(l=8, r=8, t=48 if title else 12, b=8), hovermode=hover, colorway=PALETTE,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"),
        hoverlabel=dict(bgcolor="rgba(14,16,30,0.92)", bordercolor="rgba(139,124,255,0.5)",
                        font=dict(color="#ECEEF7", family="Inter, sans-serif")),
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1,
                                       font=dict(size=11), bgcolor="rgba(0,0,0,0)"),
        **layout,
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, showline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, showline=False)
    if title and legend:
        # Keep the title on its own line above the legend.
        fig.update_layout(title=dict(y=0.98, yanchor="top"), margin=dict(t=72))
    return fig


def show(fig: go.Figure, container=st) -> None:
    container.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def glow_line(fig: go.Figure, x, y, color: str, name: str, width: float = 2.6, **kw) -> None:
    """A line with a soft halo, for the series the eye should follow."""
    fig.add_scatter(x=x, y=y, line=dict(color=color, width=width * 5), opacity=0.12, hoverinfo="skip",
                    showlegend=False, line_shape="spline")
    fig.add_scatter(x=x, y=y, name=name, line=dict(color=color, width=width), line_shape="spline", **kw)


def forecast_chart(res: ForecastResult, history: pd.DataFrame | None, title: str | None = None,
                   last_year: np.ndarray | None = None, height: int = 450) -> go.Figure:
    fig = go.Figure()
    if history is not None and len(history):
        h = history.tail(180)
        fig.add_bar(x=h["date"], y=h["units_sold"], name="Actual", marker_color="rgba(148,163,184,0.22)",
                    marker_line_width=0, hovertemplate="%{y:,.0f}")
        fig.add_scatter(x=h["date"], y=h["units_sold"].rolling(7, min_periods=1).mean(), name="Actual · 7-day avg",
                        line=dict(color=HIST, width=2), line_shape="spline", hovertemplate="%{y:,.1f}")
    if last_year is not None and np.isfinite(last_year).any():
        fig.add_scatter(x=res.dates, y=pd.Series(last_year).rolling(7, min_periods=1).mean(),
                        name="Last year · 7-day avg", line=dict(color=LAST_YEAR, width=1.6, dash="dot"),
                        line_shape="spline", hovertemplate="%{y:,.1f}")
    fig.add_scatter(x=res.dates, y=res.upper, line=dict(width=0), showlegend=False, hoverinfo="skip",
                    line_shape="spline")
    fig.add_scatter(x=res.dates, y=res.lower, fill="tonexty", fillcolor=BAND, line=dict(width=0),
                    name=f"{RANGE_LABEL} range", hoverinfo="skip", line_shape="spline")
    glow_line(fig, res.dates, res.mean, FC, "Forecast", customdata=np.stack([res.lower, res.upper], axis=1),
              hovertemplate="%{y:,.1f} <span style='opacity:.7'>(%{customdata[0]:,.0f}–%{customdata[1]:,.0f})</span>")
    promo = res.promo == 1
    if promo.any():
        fig.add_scatter(x=res.dates[promo], y=res.mean[promo], mode="markers", name="Promo day",
                        marker=dict(color=PROMO, size=8, symbol="star", line=dict(width=0)), hoverinfo="skip")
    if history is not None and len(history):
        fig.add_vline(x=res.dates[0], line_dash="dot", line_color="rgba(139,124,255,0.6)")
        fig.add_annotation(x=res.dates[0], y=1, yref="paper", text="FORECAST →", showarrow=False, xanchor="left",
                           font=dict(size=10, color="#A99DFF"), xshift=6)
    style(fig, title, height=height)
    fig.update_yaxes(title_text="Units per day", rangemode="tozero")
    return fig


def bar_index(labels: list[str], values: np.ndarray, title: str, height: int = 300) -> go.Figure:
    """Bars of a % deviation from average, green above and red below."""
    v = np.nan_to_num(np.asarray(values, dtype=float))
    fig = go.Figure(go.Bar(x=labels, y=v * 100, marker_color=[GOOD if x >= 0 else BAD for x in v],
                           marker_line_width=0, hovertemplate="%{x}: %{y:+.0f}%<extra></extra>"))
    style(fig, title, height=height, legend=False, hover="closest")
    fig.update_yaxes(title_text="% vs average", ticksuffix="%")
    return fig
