"""Shared look & feel: the "Lagoon" theme (light + dark), glass cards, hero blocks, formatting and charts."""
from __future__ import annotations

import html
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .model import RANGE_LABEL, ForecastResult

# Mid-tone colours that read well on both the dark and the light background.
BRAND = "#14B8A6"   # teal
HIST = "#7C8B9A"    # slate
FC = "#F97350"      # coral: the forecast
BAND = "rgba(249,115,80,0.16)"
PROMO = "#EC4899"
GOOD = "#22C55E"
BAD = "#EF4444"
WARN = "#F59E0B"
LAST_YEAR = "#3B82F6"
PALETTE = ["#14B8A6", "#F97350", "#3B82F6", "#F59E0B", "#EC4899", "#8B5CF6", "#22C55E", "#64748B"]
SEQ = [[0, "rgba(20,184,166,0.06)"], [0.5, "#14B8A6"], [1, "#F97350"]]  # heatmaps

PAGE_PATHS = ["/", "/planning", "/advisor", "/explore", "/overview", "/hierarchy", "/assistant", "/workspace",
              "/connections", "/orders", "/accuracy", "/tracking", "/terms"]

CSS = """
<style>
:root, html[data-dm="dark"] {
  --a: #2DD4BF; --b: #FF7A59; --c: #60A5FA;
  --ink: #E6F1F5; --ink-2: rgba(230,241,245,.72);
  --glass: rgba(255,255,255,0.035); --glass-2: rgba(255,255,255,0.06);
  --edge: rgba(45,212,191,0.20); --edge-hi: rgba(45,212,191,0.42);
  --glow: 0 0 0 1px rgba(45,212,191,0.08), 0 22px 60px -30px rgba(45,212,191,0.55);
  --hero-bg: linear-gradient(135deg, rgba(45,212,191,0.16), rgba(255,122,89,0.08) 55%, rgba(96,165,250,0.07));
  --title: linear-gradient(90deg, #FFFFFF 5%, #A7F3E9 55%, #FFC2B0);
  --big: linear-gradient(90deg, #CCFBF1, #2DD4BF 40%, #FF7A59);
  --aurora-1: rgba(45,212,191,0.16); --aurora-2: rgba(255,122,89,0.11); --aurora-3: rgba(96,165,250,0.08);
  --tab-on: linear-gradient(135deg, rgba(45,212,191,.26), rgba(255,122,89,.14));
  --chip-good: #4ADE80; --chip-warn: #FBBF24; --chip-bad: #F87171; --chip-acc: #5EEAD4;
}
html[data-dm="light"] {
  --a: #0D9488; --b: #EA580C; --c: #2563EB;
  --ink: #0B1F2A; --ink-2: rgba(11,31,42,.70);
  --glass: rgba(255,255,255,0.78); --glass-2: rgba(255,255,255,0.95);
  --edge: rgba(13,148,136,0.18); --edge-hi: rgba(13,148,136,0.40);
  --glow: 0 1px 2px rgba(11,31,42,0.05), 0 16px 40px -24px rgba(13,148,136,0.35);
  --hero-bg: linear-gradient(135deg, rgba(45,212,191,0.20), rgba(255,122,89,0.10) 55%, rgba(96,165,250,0.10));
  --title: linear-gradient(90deg, #0B1F2A 10%, #0F766E 60%, #C2410C);
  --big: linear-gradient(90deg, #0F766E, #0D9488 40%, #EA580C);
  --aurora-1: rgba(45,212,191,0.18); --aurora-2: rgba(255,122,89,0.10); --aurora-3: rgba(96,165,250,0.10);
  --tab-on: linear-gradient(135deg, rgba(45,212,191,.22), rgba(255,122,89,.12));
  --chip-good: #15803D; --chip-warn: #B45309; --chip-bad: #B91C1C; --chip-acc: #0F766E;
}

/* Aurora backdrop (static: animating a full-screen layer causes repaint glitches in some browsers). */
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(42vw 32vh at 16% 2%, var(--aurora-1), transparent 70%),
    radial-gradient(36vw 28vh at 90% 0%, var(--aurora-2), transparent 70%),
    radial-gradient(46vw 36vh at 60% 100%, var(--aurora-3), transparent 70%) !important;
  background-attachment: fixed !important;
}
[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1440px;}

/* Entrance: content rises in softly when a page opens. */
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"],
[data-testid="stMainBlockContainer"] [data-testid="stLayoutWrapper"] {animation: dm-rise .55s cubic-bezier(.2,.7,.2,1) both;}
@keyframes dm-rise { from {opacity: 0; transform: translateY(10px);} to {opacity: 1; transform: none;} }
h1, h2, h3, h4, h5 {letter-spacing: -0.02em;}

/* Hero header */
.dm-hero {position: relative; overflow: hidden; padding: 1.7rem 1.9rem 1.6rem; margin-bottom: .4rem;
  border-radius: 1.3rem; border: 1px solid var(--edge); background: var(--hero-bg); box-shadow: var(--glow);}
.dm-hero::before {content: ""; position: absolute; right: -6%; top: -60%; width: 38%; aspect-ratio: 1; border-radius: 50%;
  background: radial-gradient(circle, var(--aurora-2), transparent 65%); animation: dm-float 9s ease-in-out infinite alternate;}
@keyframes dm-float { from {transform: translate(0,0);} to {transform: translate(-6%, 10%);} }
.dm-hero::after {content: ""; position: absolute; left: 0; right: 0; bottom: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--a), var(--b), transparent);
  background-size: 200% 100%; animation: dm-shimmer 6s linear infinite;}
@keyframes dm-shimmer { from {background-position: 200% 0;} to {background-position: -200% 0;} }
.dm-eyebrow {font-size: .72rem; font-weight: 700; letter-spacing: .18em; text-transform: uppercase; color: var(--a);
  margin-bottom: .35rem; position: relative;}
.dm-hero h1 {font-family: "Space Grotesk", Inter, sans-serif; font-size: 2.3rem; font-weight: 700; margin: 0; padding: 0;
  line-height: 1.12; background: var(--title); -webkit-background-clip: text; background-clip: text; color: transparent;
  position: relative;}
.dm-hero p {margin: .55rem 0 0; color: var(--ink-2); font-size: 1rem; max-width: 62rem; position: relative;}

/* Spotlight: the headline number */
.dm-spot {position: relative; padding: 1.45rem 1.65rem; border-radius: 1.3rem; overflow: hidden;
  border: 1px solid var(--edge-hi); box-shadow: var(--glow);
  background: radial-gradient(120% 140% at 0% 0%, var(--aurora-1), transparent 55%),
              radial-gradient(120% 140% at 100% 100%, var(--aurora-2), transparent 60%), var(--glass);}
.dm-spot .dm-big {font-family: "Space Grotesk", Inter, sans-serif; font-size: 3.5rem; font-weight: 700; line-height: 1.05;
  letter-spacing: -0.035em; animation: dm-pop .7s cubic-bezier(.2,.8,.2,1) both; font-variant-numeric: tabular-nums;}
.dm-spot .dm-big span {background: var(--big); -webkit-background-clip: text; background-clip: text; color: transparent;}
.dm-spot .dm-big small {font-size: 1.05rem; font-weight: 600; letter-spacing: 0; color: var(--ink-2); margin-left: .5rem;}
@keyframes dm-pop { from {opacity: 0; transform: scale(.97) translateY(6px);} to {opacity: 1; transform: none;} }
.dm-spot .dm-line {margin-top: .7rem; color: var(--ink-2); font-size: .96rem;}
.dm-spot .dm-line b {color: var(--ink);}
.dm-chips {display: flex; flex-wrap: wrap; gap: .45rem; margin-top: .95rem;}
.dm-chip {font-size: .78rem; font-weight: 600; padding: .3rem .72rem; border-radius: 999px;
  border: 1px solid rgba(124,139,154,0.30); background: rgba(124,139,154,0.08);}
.dm-chip.good {color: var(--chip-good); border-color: rgba(34,197,94,.35); background: rgba(34,197,94,.08);}
.dm-chip.warn {color: var(--chip-warn); border-color: rgba(245,158,11,.38); background: rgba(245,158,11,.08);}
.dm-chip.bad {color: var(--chip-bad); border-color: rgba(239,68,68,.38); background: rgba(239,68,68,.08);}
.dm-chip.violet, .dm-chip.acc {color: var(--chip-acc); border-color: var(--edge-hi); background: rgba(45,212,191,.09);}

/* Glass cards (st.container keyed "card-…") and metrics */
[class*="st-key-card-"] {background: var(--glass) !important; border: 1px solid var(--edge) !important;
  border-radius: 1.1rem !important; box-shadow: var(--glow); transition: border-color .25s ease, box-shadow .25s ease;}
[class*="st-key-card-"]:hover {border-color: var(--edge-hi) !important;}
div[data-testid="stMetric"] {background: var(--glass); border-color: var(--edge) !important; border-radius: 1rem;
  padding: .85rem 1rem; position: relative; overflow: hidden; box-shadow: var(--glow);
  transition: transform .2s ease, border-color .2s ease;}
div[data-testid="stMetric"]::before {content: ""; position: absolute; inset: 0 0 auto 0; height: 2px;
  background: linear-gradient(90deg, var(--a), var(--b)); opacity: .75;}
div[data-testid="stMetric"]:hover {transform: translateY(-2px); border-color: var(--edge-hi) !important;}
div[data-testid="stMetricLabel"] p {font-size: .74rem; letter-spacing: .08em; text-transform: uppercase; opacity: .72;}
div[data-testid="stMetricValue"] {font-family: "Space Grotesk", Inter, sans-serif; letter-spacing: -0.02em;}

/* Tabs as glowing pills */
.stTabs [data-baseweb="tab-list"] {gap: .3rem; flex-wrap: wrap; padding: .3rem; background: var(--glass);
  border: 1px solid var(--edge); border-radius: .95rem;}
.stTabs [data-baseweb="tab"] {padding: .45rem .9rem; border-radius: .7rem; height: auto;}
.stTabs [aria-selected="true"] {background: var(--tab-on); box-shadow: inset 0 0 0 1px var(--edge-hi);}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

/* Buttons */
[data-testid="stBaseButton-primary"] {background: linear-gradient(135deg, #14B8A6, #F97350) !important;
  border: none !important; color: #fff !important; box-shadow: 0 10px 30px -14px rgba(249,115,80,.8);
  transition: transform .15s ease, filter .15s ease;}
[data-testid="stBaseButton-primary"]:hover {transform: translateY(-1px); filter: brightness(1.06);}

/* Sidebar */
[data-testid="stSidebar"] {border-right: 1px solid var(--edge);}
[data-testid="stSidebarContent"] {background: linear-gradient(180deg, var(--aurora-1), transparent 40%);}
.dm-brand {font-family: "Space Grotesk", Inter, sans-serif; font-weight: 700; font-size: 1.5rem; letter-spacing: -0.03em;
  display: flex; align-items: center; gap: .5rem;}
.dm-brand i {width: 1.35rem; height: 1.35rem; border-radius: .45rem; display: inline-block;
  background: conic-gradient(from 200deg, #14B8A6, #3B82F6, #F97350, #14B8A6); box-shadow: 0 0 18px rgba(45,212,191,.5);}
.dm-brand b {font-weight: 700;}
.dm-brand span {background: linear-gradient(90deg, #14B8A6, #F97350); -webkit-background-clip: text;
  background-clip: text; color: transparent;}
.dm-sub {color: var(--ink-2); font-size: .8rem; margin-top: -.1rem;}

/* Findings grid */
.dm-findings {display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: .7rem;}
.dm-finding {padding: .9rem 1rem; border-radius: .95rem; background: var(--glass); border: 1px solid var(--edge);
  font-size: .9rem; line-height: 1.45; animation: dm-rise .6s both; box-shadow: var(--glow);}
.dm-finding .dm-ft {display: block; font-size: .68rem; letter-spacing: .14em; text-transform: uppercase;
  color: var(--b); margin-bottom: .3rem; font-weight: 700;}

/* Stat strip: small headline numbers in a row */
.dm-strip {display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: .6rem; margin: .2rem 0 .4rem;}
.dm-stat {padding: .75rem .9rem; border-radius: .9rem; background: var(--glass); border: 1px solid var(--edge);}
.dm-stat b {display: block; font-family: "Space Grotesk", Inter, sans-serif; font-size: 1.45rem; font-weight: 700;
  font-variant-numeric: tabular-nums; color: var(--ink);}
.dm-stat span {font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: var(--ink-2);}
::-webkit-scrollbar {width: 10px; height: 10px;}
::-webkit-scrollbar-thumb {background: var(--edge-hi); border-radius: 10px;}

/* Forecast page: input panel + results wrap onto separate rows when space is tight. */
.st-key-fc_layout > div > [data-testid="stHorizontalBlock"] {flex-wrap: wrap; row-gap: 1rem;}
.st-key-fc_layout > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:first-child
  {flex: 1 1 300px !important; width: auto !important; min-width: 0;}
.st-key-fc_layout > div > [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:last-child
  {flex: 3 1 560px !important; width: auto !important; min-width: 0;}

@media (prefers-reduced-motion: reduce) { *, *::before, *::after {animation: none !important; transition: none !important;} }
@media (max-width: 640px) {
  .dm-hero {padding: 1.1rem 1.1rem 1rem;} .dm-hero h1 {font-size: 1.55rem;} .dm-hero p {font-size: .92rem;}
  .dm-spot {padding: 1.1rem;} .dm-spot .dm-big {font-size: 2.4rem;}
  .block-container {padding: 1rem .75rem 3rem;}
  .dm-findings {grid-template-columns: 1fr;} .dm-strip {grid-template-columns: repeat(2, 1fr);}
  .stTabs [data-baseweb="tab-list"] {flex-wrap: nowrap; overflow-x: auto;}
  div[data-testid="stMetric"] {padding: .65rem .75rem;}
  div[data-testid="stMetricValue"] {font-size: 1.6rem;}
  [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {min-height: 44px;}
}
</style>
"""

# Runs once per page load (our own static code; no user content). It (1) marks <html> with the active
# light/dark mode so the CSS above can follow Streamlit's theme, and (2) animates headline numbers.
JS = """
<script>
(() => {
  if (window.__dmBooted) return; window.__dmBooted = true;
  const root = document.documentElement;
  const last = (window.__dmLast = window.__dmLast || {});
  const mode = () => {
    const app = document.querySelector('.stApp'); if (!app) return;
    const m = getComputedStyle(app).backgroundColor.match(/\\d+/g); if (!m) return;
    const lum = (0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2]) / 255;
    const t = lum > 0.5 ? 'light' : 'dark'; if (root.dataset.dm !== t) root.dataset.dm = t;
  };
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const count = () => document.querySelectorAll('.dm-count:not([data-done])').forEach(el => {
    el.dataset.done = '1';
    const target = parseFloat(el.dataset.value), dec = +el.dataset.dec || 0, key = el.dataset.key || '';
    const pre = el.dataset.prefix || '', suf = el.dataset.suffix || '';
    const from = key in last ? last[key] : 0; last[key] = target;
    if (reduce || !isFinite(target) || from === target) return;
    const t0 = performance.now(), dur = 900;
    const fmt = v => pre + v.toLocaleString('en-US', {minimumFractionDigits: dec, maximumFractionDigits: dec}) + suf;
    const step = t => { const p = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - p, 3);
      el.textContent = fmt(from + (target - from) * e); if (p < 1) requestAnimationFrame(step); };
    requestAnimationFrame(step);
  });
  new MutationObserver(() => { mode(); count(); }).observe(document.body, {childList: true, subtree: true});
  setInterval(mode, 700); mode(); count();
})();
</script>
"""


def inject_css() -> None:
    st.html(CSS)
    st.html(JS, unsafe_allow_javascript=True)


def is_light() -> bool:
    try:
        return st.context.theme.type == "light"
    except Exception:  # noqa: BLE001 - outside a browser session (tests)
        return False


def set_theme_script(theme: str) -> None:
    """Switch Streamlit's own light/dark theme for every page, then reload (the browser remembers it)."""
    value = json.dumps(json.dumps({"light": "Light", "dark": "Dark"}.get(theme, "System")))
    paths = json.dumps(PAGE_PATHS)
    st.html(f"<script>(() => {{ for (const p of {paths}) localStorage.setItem('stActiveTheme-' + p + '-v2', {value});"
            f" setTimeout(() => location.reload(), 60); }})();</script>", unsafe_allow_javascript=True)


def theme_toggle() -> None:
    """Sun/moon switch in the sidebar."""
    def _changed() -> None:
        st.session_state["_theme_switch"] = st.session_state["dm_theme"]

    current = "light" if is_light() else "dark"
    st.segmented_control("Appearance", ["dark", "light"], default=current, required=True, key="dm_theme",
                         format_func=lambda m: {"dark": ":material/dark_mode: Dark", "light": ":material/light_mode: Light"}[m],
                         on_change=_changed, help="Switches the whole app between dark and light mode (reloads the page).")
    if st.session_state.pop("_theme_switch", None) in ("light", "dark"):
        set_theme_script(st.session_state["dm_theme"])


def page_header(title: str, subtitle: str, eyebrow: str = "Demandino") -> None:
    st.html(f'<div class="dm-hero"><div class="dm-eyebrow">{html.escape(eyebrow)}</div>'
            f'<h1>{html.escape(title)}</h1><p>{subtitle}</p></div>')


def card(name: str):
    """A glass card container (styled through its key)."""
    return st.container(border=True, key=f"card-{name}")


def counter(value: float, text: str, key: str, decimals: int = 0, prefix: str = "", suffix: str = "") -> str:
    """HTML for a number that counts up to its new value (shows the final text if scripts are off)."""
    return (f'<span class="dm-count" data-value="{float(value):.6f}" data-dec="{decimals}" '
            f'data-prefix="{html.escape(prefix)}" data-suffix="{html.escape(suffix)}" data-key="{html.escape(key)}">'
            f'{html.escape(text)}</span>')


def spotlight(eyebrow: str, value: str, unit: str, line: str, chips: list[tuple[str, str]],
              count: tuple | None = None) -> None:
    """The headline number of a page, large and glowing, with status chips underneath.

    `count` = (number, key, decimals, prefix, suffix) animates the value."""
    chip_html = "".join(f'<span class="dm-chip {cls}">{html.escape(text)}</span>' for text, cls in chips)
    big = counter(count[0], value, *count[1:]) if count else f"<span>{html.escape(value)}</span>"
    if count:
        big = f"<span>{big}</span>"
    st.html(f'<div class="dm-spot"><div class="dm-eyebrow">{html.escape(eyebrow)}</div>'
            f'<div class="dm-big">{big}<small>{html.escape(unit)}</small></div>'
            f'<div class="dm-line">{line}</div><div class="dm-chips">{chip_html}</div></div>')


def stat_strip(items: list[tuple[str, str]]) -> None:
    """A row of small headline stats: (label, value)."""
    cells = "".join(f'<div class="dm-stat"><span>{html.escape(k)}</span><b>{html.escape(v)}</b></div>' for k, v in items)
    st.html(f'<div class="dm-strip">{cells}</div>')


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
GRID = "rgba(124,139,154,0.16)"


def style(fig: go.Figure, title: str | None = None, height: int = 360, legend: bool = True,
          hover: str = "x unified", **layout) -> go.Figure:
    light = is_light()
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family="Space Grotesk, Inter, sans-serif")) if title else None,
        height=height, margin=dict(l=8, r=8, t=48 if title else 12, b=8), hovermode=hover, colorway=PALETTE,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif"),
        hoverlabel=dict(bgcolor="rgba(255,255,255,0.96)" if light else "rgba(7,17,27,0.94)",
                        bordercolor="rgba(13,148,136,0.45)",
                        font=dict(color="#0B1F2A" if light else "#E6F1F5", family="Inter, sans-serif")),
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
                    showlegend=False, line_shape="spline", **({"yaxis": kw["yaxis"]} if "yaxis" in kw else {}))
    fig.add_scatter(x=x, y=y, name=name, line=dict(color=color, width=width), line_shape="spline", **kw)


def forecast_chart(res: ForecastResult, history: pd.DataFrame | None, title: str | None = None,
                   last_year: np.ndarray | None = None, height: int = 460) -> go.Figure:
    fig = go.Figure()
    if history is not None and len(history):
        h = history.tail(180)
        fig.add_bar(x=h["date"], y=h["units_sold"], name="Actual", marker_color="rgba(124,139,154,0.28)",
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
        fig.add_vrect(x0=res.dates[0], x1=res.dates[-1], fillcolor="rgba(20,184,166,0.05)", line_width=0)
        fig.add_vline(x=res.dates[0], line_dash="dot", line_color="rgba(20,184,166,0.7)")
        fig.add_annotation(x=res.dates[0], y=1, yref="paper", text="FORECAST →", showarrow=False, xanchor="left",
                           font=dict(size=10, color=BRAND), xshift=6)
    style(fig, title, height=height)
    fig.update_yaxes(title_text="Units per day", rangemode="tozero")
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.06, bgcolor="rgba(124,139,154,0.08)"))
    return fig


def bar_index(labels: list[str], values: np.ndarray, title: str, height: int = 300) -> go.Figure:
    """Bars of a % deviation from average, green above and red below."""
    v = np.nan_to_num(np.asarray(values, dtype=float))
    fig = go.Figure(go.Bar(x=labels, y=v * 100, marker_color=[GOOD if x >= 0 else BAD for x in v],
                           marker_line_width=0, hovertemplate="%{x}: %{y:+.0f}%<extra></extra>"))
    style(fig, title, height=height, legend=False, hover="closest")
    fig.update_yaxes(title_text="% vs average", ticksuffix="%")
    return fig


def polar_index(labels: list[str], values: np.ndarray, title: str, height: int = 320) -> go.Figure:
    """A radar of an index around 100 (e.g. demand by month), filled like a glow."""
    v = (np.nan_to_num(np.asarray(values, dtype=float), nan=1.0)) * 100
    fig = go.Figure(go.Scatterpolar(r=list(v) + [v[0]], theta=list(labels) + [labels[0]], fill="toself",
                                    fillcolor="rgba(20,184,166,0.18)", line=dict(color=BRAND, width=2.5),
                                    hovertemplate="%{theta}: %{r:.0f}<extra></extra>"))
    fig.add_trace(go.Scatterpolar(r=[100] * (len(labels) + 1), theta=list(labels) + [labels[0]], mode="lines",
                                  line=dict(color=HIST, dash="dot", width=1), hoverinfo="skip", showlegend=False))
    style(fig, title, height=height, legend=False, hover="closest",
          polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(gridcolor=GRID, showticklabels=False),
                     angularaxis=dict(gridcolor=GRID)))
    return fig


def gauge(value: float, title: str, suffix: str = "%", max_value: float = 100, good_below: float | None = None,
          height: int = 230) -> go.Figure:
    """A semicircle gauge; green zone up to `good_below` when lower is better."""
    steps = [dict(range=[0, good_below], color="rgba(34,197,94,0.18)")] if good_below else []
    fig = go.Figure(go.Indicator(mode="gauge+number", value=value, number=dict(suffix=suffix, font=dict(size=34)),
                                 gauge=dict(axis=dict(range=[0, max_value], tickwidth=0), bar=dict(color=FC),
                                            bgcolor="rgba(124,139,154,0.10)", borderwidth=0, steps=steps,
                                            threshold=dict(line=dict(color=GOOD, width=3), value=good_below)
                                            if good_below else None)))
    style(fig, title, height=height, legend=False, hover="closest")
    fig.update_layout(margin=dict(l=20, r=20, t=50, b=0))
    return fig
