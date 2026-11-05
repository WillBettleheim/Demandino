"""The welcome page: a cinematic navy-and-coral hero, an interactive product tour and the story of the product.

Everything here is static, our own code with no user content, so it needs no data and opens instantly. Numbers in the
tour are labelled as examples.
"""
from __future__ import annotations

import json

# Streamlit's HTML blocks remove inline SVG, and any script holding tags, so the script draws icons from these paths.
ICON = {
    "forecast": "M3 17l5-6 4 3 6-8M15 6h4v4",
    "plan": "M6 5h12a2 2 0 0 1 2 2v11a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2zM4 10h16M9 3v4M15 3v4",
    "grow": "M4 20V10M10 20V4M16 20v-7M22 20H2",
    "alert": "M6 16v-5a6 6 0 1 1 12 0v5l2 2H4zM10 21h4",
    "trust": "M12 3l8 3v6c0 5-3.4 8-8 9-4.6-1-8-4-8-9V6zM9 12l2 2 4-4",
}


TOUR = [
    ("forecast", "Forecast", "Every product,<br>day by day"),
    ("plan", "Plan", "Know what to order,<br>and when"),
    ("grow", "Grow", "Test a price<br>before you change it"),
    ("alert", "Alerts", "Hear about surprises<br>early"),
    ("trust", "Honest", "Tested on sales<br>it hasn't seen"),
]

STORY = [
    ("Forecast a product", "Demand for each product, up to 180 days ahead, with a likely range rather than one "
     "falsely precise number. New products borrow patterns from similar ones."),
    ("Plan your orders", "Order quantities for the whole shop within a budget, reorder dates, purchase orders and "
     "your suppliers' real delivery times."),
    ("Grow demand", "Try a price change or a sale on one product or the whole shop, and see the effect on units, "
     "revenue and profit first."),
    ("Stay ahead", "An inbox of what needs you this week: products moving faster or slower than expected, sales "
     "lost to stock-outs and slow stock to clear."),
    ("Use your own data", "Upload a CSV of daily sales. Demandino checks it for duplicates, gaps and odd prices, "
     "then learns your seasons, holidays and promotions."),
    ("Keep it private", "Your workspace belongs to your account. Saved keys are encrypted and never shown again, "
     "and you can sign out everywhere at any time."),
]

STEPS = [
    ("01", "Bring your sales", "Start with the sample shop, or upload your own daily sales file."),
    ("02", "Pick a product", "Set the price, promotions and stock you have planned."),
    ("03", "Act on the answer", "See how many will sell, when to reorder and what it earns."),
]

CSS = """
<style>
.stApp, [data-testid="stAppViewContainer"] {background: #0B1730 !important;}
.block-container, [data-testid="stMainBlockContainer"] {padding: 0 !important; max-width: 100% !important;}
[data-testid="stMainBlockContainer"] [data-testid="stElementContainer"],
[data-testid="stMainBlockContainer"] [data-testid="stLayoutWrapper"] {animation: none !important;}
[data-testid="stMainBlockContainer"] [data-testid="stVerticalBlock"] {gap: 0;}
[data-testid="stElementContainer"]:has(> .stHtml > style:only-child) {height: 0; margin: 0;}
.st-key-dml_stage {position: relative;}
.dml {--navy: #0B1730; --white: #F8F8F6; --coral: #FF7F6E; --mute: rgba(248,248,246,.74); color: var(--white);
  font-family: Inter, sans-serif; background: var(--navy); overflow: hidden;}
.dml * {box-sizing: border-box;}

/* Hero */
.dml-hero {position: relative; min-height: 100svh; display: flex; flex-direction: column; justify-content: flex-end;
  padding: 6rem clamp(1.25rem, 3.4vw, 3.2rem) 0; isolation: isolate;
  background: radial-gradient(60vw 60vh at 58% 42%, #E8604C, transparent 70%),
              radial-gradient(40vw 50vh at 8% 34%, #7E8CA8, transparent 70%), #0B1730;}
.dml-hero canvas {position: absolute; inset: 0; width: 100%; height: 100%; z-index: -2; display: block;}
.dml-hero::after {content: ""; position: absolute; inset: 0; z-index: -1; pointer-events: none;
  background: linear-gradient(180deg, rgba(11,23,48,.34), transparent 22%, transparent 42%, rgba(11,23,48,.86) 92%);}
.dml-brand {position: absolute; top: 5rem; left: clamp(1.25rem, 3.4vw, 3.2rem); display: flex; align-items: center;
  gap: .6rem; font-family: "Space Grotesk", Inter, sans-serif; font-size: 1.5rem; font-weight: 600;
  letter-spacing: -.02em; opacity: 0; animation: dml-in 1s .1s forwards;}
.dml-mark, .dml-mark svg {display: block; width: 1.9rem; height: 1.9rem;}
.dml-brand small {font: 500 .8rem Inter, sans-serif; letter-spacing: 0; color: var(--mute); margin-left: .5rem;
  padding-left: .85rem; border-left: 1px solid rgba(248,248,246,.3);}
.dml h1 {font-family: Inter, sans-serif; font-weight: 400; font-size: clamp(3rem, 7.2vw, 7rem); line-height: 1.02;
  letter-spacing: -.045em; margin: 0; padding: 0; color: var(--white); max-width: 11.5em;}
.dml h1 span {display: block; overflow: hidden; padding-bottom: .08em;}
.dml h1 span i {display: block; font-style: normal; transform: translateY(110%); animation: dml-up 1.15s cubic-bezier(.16,1,.3,1) forwards;}
.dml h1 span:nth-child(2) i {animation-delay: .14s;}
@keyframes dml-up {to {transform: none;}}
@keyframes dml-in {from {opacity: 0; transform: translateY(12px);} to {opacity: 1; transform: none;}}
.dml-lede {max-width: 35rem; margin: 1.1rem 0 0; font-size: clamp(1rem, 1.25vw, 1.15rem); line-height: 1.5;
  color: rgba(248,248,246,.88); opacity: 0; animation: dml-in 1s .55s forwards;}

/* Feature strip */
.dml-strip {display: grid; grid-template-columns: repeat(5, 1fr); gap: 1.25rem; margin-top: 2.2rem;
  border-top: 1px solid rgba(248,248,246,.2); opacity: 0; animation: dml-in 1s .8s forwards;}
.dml-tab {all: unset; position: relative; cursor: pointer; padding: 1.05rem 0 1.5rem; color: rgba(248,248,246,.78);
  font-size: .92rem; line-height: 1.3; transition: color .25s ease;}
.dml-ico {display: block; width: 1.3rem; height: 1.3rem; margin-bottom: .55rem;}
.dml-ico svg {display: block; width: 100%; height: 100%;}
.dml-tab::before {content: ""; position: absolute; left: 0; right: 0; top: -1px; height: 2px; background: var(--coral);
  transform: scaleX(0); transform-origin: left;}
.dml-tab:hover, .dml-tab:focus-visible {color: var(--white);}
.dml-tab:focus-visible {outline: 2px solid var(--coral); outline-offset: 3px; border-radius: 4px;}
.dml-tab.on {color: var(--coral);}
.dml-tab.on::before {animation: dml-bar var(--tour, 7s) linear forwards;}
.dml.held .dml-tab.on::before {animation: none; transform: scaleX(1);}
@keyframes dml-bar {to {transform: scaleX(1);}}

/* Floating glass card */
.dml-card {position: absolute; top: 5rem; right: clamp(1.25rem, 3.4vw, 3.2rem); width: min(25rem, 36vw); padding: 1.1rem;
  border-radius: 1.3rem; background: rgba(226,232,245,.14); border: 1px solid rgba(248,248,246,.2);
  backdrop-filter: blur(22px) saturate(1.3); -webkit-backdrop-filter: blur(22px) saturate(1.3);
  box-shadow: 0 30px 70px -36px rgba(4,10,26,.9); opacity: 0; animation: dml-in 1.1s .45s forwards;
  transition: transform .25s ease-out; transform-style: preserve-3d;}
.dml-grip {width: 2.2rem; height: 3px; border-radius: 3px; background: rgba(248,248,246,.75); margin: 0 auto .75rem;}
.dml-panel {display: none;} .dml-panel.on {display: block; animation: dml-in .5s both;}
.dml-where {display: flex; justify-content: space-between; gap: .5rem; font-size: .78rem; color: var(--mute); margin-bottom: .7rem;}
.dml-where b {font-weight: 500; color: var(--white);}
.dml-tile {padding: .8rem .9rem; border-radius: .95rem; background: rgba(248,248,246,.1);
  border: 1px solid rgba(248,248,246,.12); margin-bottom: .6rem;}
.dml-row {display: flex; justify-content: space-between; align-items: flex-end; gap: .75rem;}
.dml-row + .dml-row {margin-top: .35rem;}
.dml-k {font-size: .74rem; color: var(--mute);}
.dml-v {font-size: 1.45rem; letter-spacing: -.02em; font-weight: 500; font-variant-numeric: tabular-nums; line-height: 1.2;}
.dml-t {font-size: .86rem; font-weight: 600;}
.dml-r {text-align: right;}
.dml-pill {display: inline-block; font-size: .72rem; font-weight: 700; padding: .32rem .75rem; border-radius: 99px;
  background: #D6452F; color: #fff;}
.dml-chip {font-size: .7rem; padding: .15rem .5rem; border-radius: 99px; background: rgba(248,248,246,.18);}
.dml-chart {width: 100%; height: 6.6rem; margin-top: .4rem; touch-action: none; cursor: crosshair;}
.dml-chart svg {display: block; width: 100%; height: 100%; overflow: visible;}
.dml-range {width: 100%; accent-color: #FF7F6E; margin: .5rem 0 .15rem; cursor: grab;}
.dml-note {font-size: .72rem; color: var(--mute);}

/* Story */
.dml-sec {padding: clamp(3.5rem, 8vw, 7rem) clamp(1.25rem, 3.4vw, 3.2rem) 0; max-width: 1500px; margin: 0 auto;}
.dml-eyebrow {color: var(--coral); font-size: .78rem; font-weight: 600; letter-spacing: .16em; text-transform: uppercase;}
.dml h2 {font-family: Inter, sans-serif; font-weight: 400; font-size: clamp(2rem, 4.4vw, 4rem); letter-spacing: -.035em;
  line-height: 1.08; margin: .7rem 0 0; padding: 0; color: var(--white); max-width: 18em;}
.dml-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(min(19rem, 100%), 1fr)); gap: 1rem; margin-top: 2.6rem;}
.dml-box {--x: 50%; --y: 50%; position: relative; padding: 1.5rem; border-radius: 1.2rem; overflow: hidden;
  background: #122145; border: 1px solid rgba(248,248,246,.1); transition: border-color .25s ease, transform .25s ease;}
.dml-box::before {content: ""; position: absolute; inset: 0; opacity: 0; transition: opacity .3s ease; pointer-events: none;
  background: radial-gradient(18rem at var(--x) var(--y), rgba(255,127,110,.22), transparent 70%);}
.dml-box:hover {border-color: rgba(255,127,110,.5); transform: translateY(-3px);}
.dml-box:hover::before {opacity: 1;}
.dml-box h3 {font-family: Inter, sans-serif; font-size: 1.15rem; font-weight: 600; letter-spacing: -.01em; margin: 0 0 .5rem;
  padding: 0; color: var(--white); position: relative;}
.dml-box p {margin: 0; font-size: .95rem; line-height: 1.5; color: var(--mute); position: relative;}
.dml-num {display: block; color: var(--coral); font-size: .85rem; font-weight: 600; margin-bottom: 1.8rem; position: relative;}
.dml-reveal {opacity: 0; transform: translateY(26px); transition: opacity .9s ease, transform .9s cubic-bezier(.16,1,.3,1);}
.dml-reveal.seen {opacity: 1; transform: none;}
.dml-end {padding: clamp(3.5rem, 8vw, 7rem) clamp(1.25rem, 3.4vw, 3.2rem) 1.5rem; text-align: center;}
.dml-end h2 {margin: 0 auto; max-width: 15em;}
.dml-foot {background: #0B1730; color: rgba(248,248,246,.72); text-align: center; font-size: .82rem; padding: 2.5rem 1.25rem 2.5rem;}

/* Real Streamlit buttons, placed over the hero */
.st-key-dml_cta {position: absolute; right: clamp(1.25rem, 3.4vw, 3.2rem); bottom: 9.4rem; width: auto !important; z-index: 3;
  flex-direction: row; gap: .6rem; align-items: center; opacity: 0; animation: dml-in 1s .7s forwards;}
.st-key-dml_cta > div, .st-key-dml_cta [data-testid="stElementContainer"] {width: auto !important;}
.st-key-dml_cta button, .st-key-dml_cta2 button {border-radius: 99px !important; min-height: 3rem; padding: .3rem 1.4rem !important;
  white-space: nowrap;}
.st-key-dml_cta button p, .st-key-dml_cta2 button p {font-weight: 600;}
.st-key-dml_cta [data-testid="stBaseButton-secondary"] {background: rgba(11,23,48,.35) !important; color: #F8F8F6 !important;
  border: 1px solid rgba(248,248,246,.45) !important;}
.st-key-dml_cta [data-testid="stBaseButton-secondary"]:hover {border-color: #FF7F6E !important; color: #FF7F6E !important;}
.st-key-dml_cta [data-testid="stBaseButton-primary"], .st-key-dml_cta2 [data-testid="stBaseButton-primary"] {background: #D6452F !important;}
.st-key-dml_cta2 {background: #0B1730; align-items: center; padding-bottom: 1rem;}
.st-key-dml_cta2 > div {width: auto !important;}

@media (max-width: 980px) {
  .dml-hero {min-height: 0; padding-top: 8.6rem;}
  .dml-card {position: relative; top: auto; right: auto; width: 100%; max-width: 30rem; order: 3; margin-top: 1.8rem;}
  .dml-strip {order: 4; grid-template-columns: repeat(5, minmax(7.5rem, 1fr)); overflow-x: auto; gap: 1rem;}
  .dml h1 {font-size: clamp(2.7rem, 12vw, 5rem);}
  .dml-brand small {display: none;}
  .st-key-dml_cta {position: static; background: #0B1730; padding: 1.25rem; flex-wrap: wrap;}
}
@media (prefers-reduced-motion: reduce) {
  .dml *, .st-key-dml_cta {animation: none !important; opacity: 1 !important; transform: none !important; transition: none !important;}
  .dml h1 span i {transform: none !important;} .dml-tab.on::before {transform: scaleX(1);} .dml-reveal {opacity: 1;}
}
</style>
"""

PANELS = """
<div class="dml-panel on" data-panel="forecast">
  <div class="dml-where"><span>Example · Wireless Earbuds</span><b>Move across the chart</b></div>
  <div class="dml-tile">
    <div class="dml-row"><div><div class="dml-k" id="dml-day">Next Friday</div><div class="dml-v" id="dml-units">42 units</div></div>
      <div class="dml-r"><div class="dml-k">Likely range</div><div class="dml-v" id="dml-band" style="font-size:1.05rem">34 – 51</div></div></div>
    <div class="dml-chart" id="dml-chart" role="img" aria-label="Example chart of a daily sales forecast with a likely range"></div>
  </div>
  <span class="dml-pill">A range, not a guess</span>
</div>
<div class="dml-panel" data-panel="plan">
  <div class="dml-where"><span>Example · this week's orders</span><b>Budget on track</b></div>
  <div class="dml-tile"><div class="dml-row"><span class="dml-t">Smart Watch</span><span class="dml-k">Runs out in 9 days</span></div>
    <div class="dml-row"><div><div class="dml-k">Reorder by</div><div class="dml-v">Fri 14</div></div><span class="dml-chip">6-day delivery</span>
      <div class="dml-r"><div class="dml-k">Order</div><div class="dml-v">240</div></div></div></div>
  <div class="dml-tile"><div class="dml-row"><span class="dml-t">Wireless Earbuds</span><span class="dml-k">Runs out in 21 days</span></div>
    <div class="dml-row"><div><div class="dml-k">Reorder by</div><div class="dml-v">Sat 22</div></div><span class="dml-chip">4-day delivery</span>
      <div class="dml-r"><div class="dml-k">Order</div><div class="dml-v">410</div></div></div></div>
  <span class="dml-pill">Fewer stock-outs, less cash tied up</span>
</div>
<div class="dml-panel" data-panel="grow">
  <div class="dml-where"><span>Example · try a price</span><b>Drag the slider</b></div>
  <div class="dml-tile">
    <div class="dml-row"><div><div class="dml-k">Price change</div><div class="dml-v" id="dml-price">+5%</div></div>
      <div class="dml-r"><div class="dml-k">Profit, next 60 days</div><div class="dml-v" id="dml-profit">+3.4%</div></div></div>
    <input class="dml-range" id="dml-slider" type="range" min="-30" max="30" value="5" aria-label="Example price change">
    <div class="dml-row"><span class="dml-k">Units <b id="dml-vol">−6.8%</b></span><span class="dml-k">Revenue <b id="dml-rev">−2.1%</b></span></div>
  </div>
  <span class="dml-pill">See the effect before you commit</span>
</div>
<div class="dml-panel" data-panel="alert">
  <div class="dml-where"><span>Example · your inbox</span><b>3 need you</b></div>
  <div class="dml-tile"><div class="dml-row"><span class="dml-t">Yoga Mat is selling faster</span><span class="dml-chip">+38%</span></div>
    <div class="dml-k">Above its forecast for 6 days. Stock may run out early.</div></div>
  <div class="dml-tile"><div class="dml-row"><span class="dml-t">Desk Lamp has slowed</span><span class="dml-chip">−24%</span></div>
    <div class="dml-k">A small markdown would clear what's left.</div></div>
  <span class="dml-pill">Most urgent first</span>
</div>
<div class="dml-panel" data-panel="trust">
  <div class="dml-where"><span>How we test it</span><b>Open on Model accuracy</b></div>
  <div class="dml-tile"><div class="dml-t">Checked against sales it never saw</div>
    <div class="dml-k" style="margin-top:.35rem">Every model is scored on the most recent sales, held back from training, so its record is a fair one.</div></div>
  <div class="dml-tile"><div class="dml-t">About 95 days in 100 inside the range</div>
    <div class="dml-k" style="margin-top:.35rem">From testing on real shop data. Single days are hard; weekly and monthly totals are much steadier.</div></div>
  <span class="dml-pill">Honest about what it doesn't know</span>
</div>
"""


def hero_html() -> str:
    tabs = "".join(f'<button class="dml-tab{" on" if i == 0 else ""}" data-tab="{key}" type="button"><span class="dml-ico" data-ico="{key}"></span>'
                   f'<b style="font-weight:600">{name}</b><br>{text}</button>' for i, (key, name, text) in enumerate(TOUR))
    return (f'<div class="dml" id="dml-root"><section class="dml-hero" id="dml-hero"><canvas id="dml-bg" aria-hidden="true"></canvas>'
            f'<div class="dml-brand"><span class="dml-mark"></span>Demandino<small>Machine-learning demand forecasting</small></div>'
            f'<aside class="dml-card" id="dml-card" aria-label="Product tour"><div class="dml-grip"></div>{PANELS}</aside>'
            '<h1><span><i>Know what sells</i></span><span><i>before it sells.</i></span></h1>'
            '<p class="dml-lede">Demandino learns from your sales, then tells you how much of each product will sell, '
            'when to reorder and what a price change would do. No statistics degree, no spreadsheets at midnight.</p>'
            f'<nav class="dml-strip" aria-label="What Demandino does">{tabs}</nav></section></div>')


def story_html() -> str:
    boxes = "".join(f'<article class="dml-box dml-reveal"><h3>{t}</h3><p>{d}</p></article>' for t, d in STORY)
    steps = "".join(f'<article class="dml-box dml-reveal"><span class="dml-num">{n}</span><h3>{t}</h3><p>{d}</p></article>'
                    for n, t, d in STEPS)
    return ('<div class="dml"><section class="dml-sec"><div class="dml-eyebrow dml-reveal">What it does</div>'
            '<h2 class="dml-reveal">From last month\'s sales to next month\'s decisions.</h2>'
            f'<div class="dml-grid">{boxes}</div></section>'
            '<section class="dml-sec"><div class="dml-eyebrow dml-reveal">How it works</div>'
            '<h2 class="dml-reveal">Three steps to your first forecast.</h2>'
            f'<div class="dml-grid">{steps}</div></section>'
            '<section class="dml-end"><div class="dml-eyebrow dml-reveal">Try it now</div>'
            '<h2 class="dml-reveal" style="margin-top:.7rem">Your next order deserves better than a hunch.</h2></section></div>')


FOOT = ('<div class="dml-foot">Demandino · Forecasts are estimates to support your decisions, not guarantees.</div>')

# Runs each time the page opens and stops itself when the page is left.
JS_BODY = r"""
(() => {
  const root = document.getElementById('dml-root'); if (!root || root.dataset.live) return; root.dataset.live = '1';
  const $ = id => document.getElementById(id);
  const svg = (tag, attrs, parent) => { const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v)); if (parent) parent.appendChild(el); return el; };
  const ICONS = __ICONS__;
  root.querySelectorAll('.dml-ico').forEach(host => { const s = svg('svg', {viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor',
      'stroke-width': 1.6, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true'}, host);
    svg('path', {d: ICONS[host.dataset.ico]}, s); });
  const mark = svg('svg', {viewBox: '0 0 32 32', fill: 'none', 'aria-hidden': 'true'}, root.querySelector('.dml-mark'));
  svg('path', {d: 'M3 22c5 0 6-11 12-11s6 7 14-3', stroke: '#FF7F6E', 'stroke-width': 3, 'stroke-linecap': 'round'}, mark);
  svg('path', {d: 'M3 28h26', stroke: '#F8F8F6', 'stroke-width': 2, 'stroke-linecap': 'round', opacity: .55}, mark);
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const alive = () => root.isConnected;

  // ---- Background: fluted glass over slowly moving navy, white and coral light; the coral follows the pointer.
  const hero = $('dml-hero'), canvas = $('dml-bg');
  const gl = canvas.getContext('webgl', {antialias: false, alpha: false, powerPreference: 'low-power'});
  const pointer = {x: .58, y: .55, tx: .58, ty: .55};
  hero.addEventListener('pointermove', e => { const b = hero.getBoundingClientRect();
    pointer.tx = (e.clientX - b.left) / b.width; pointer.ty = 1 - (e.clientY - b.top) / b.height; });
  if (gl) {
    const vs = 'attribute vec2 p; void main(){gl_Position=vec4(p,0.,1.);}';
    const fs = `precision mediump float; uniform vec2 r; uniform float t; uniform vec2 m;
      float b(vec2 p, vec2 c, float s){vec2 d=p-c; return exp(-dot(d,d)/s);}
      void main(){ vec2 uv=gl_FragCoord.xy/r; float a=r.x/r.y; float n=floor(r.x/9.);
        float cell=fract(uv.x*n)-.5; float x=(floor(uv.x*n)+.5+cell*-.55)/n;
        vec2 q=vec2(x*a, uv.y+.012*sin(x*38.+t*.25));
        vec2 c=vec2(a*(.46+.1*m.x+.05*sin(t*.17)), .50+.16*(m.y-.5)+.05*cos(t*.13));
        float coral=b(q,c,.13)+.5*b(q,c+vec2(.28*a*sin(t*.09),-.2),.07);
        float pale=.72*b(q,vec2(a*(.05+.05*sin(t*.11)),.58),.16)+.5*b(q,vec2(a*(.97-.04*cos(t*.1)),.42),.09);
        float deep=.8*b(q,vec2(a*.8,.86),.12)+.6*b(q,vec2(a*.34,.92),.08);
        vec3 col=vec3(.043,.09,.19);
        col=mix(col,vec3(.47,.53,.66),clamp(pale,0.,1.));
        col=mix(col,vec3(.11,.17,.33),clamp(deep,0.,.8));
        col=mix(col,vec3(.93,.40,.34),clamp(coral*.92,0.,1.));
        col=mix(col,vec3(.98,.62,.54),clamp(coral-.72,0.,.5));
        col*=.88+.2*(.5+cell)*(1.-abs(cell)*.6);
        col-=.05*smoothstep(.40,.5,abs(cell));
        gl_FragColor=vec4(col,1.); }`;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
    const prog = gl.createProgram(); gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(prog);
    if (gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      gl.useProgram(prog); gl.bindBuffer(gl.ARRAY_BUFFER, gl.createBuffer());
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
      const loc = gl.getAttribLocation(prog, 'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      const ur = gl.getUniformLocation(prog, 'r'), ut = gl.getUniformLocation(prog, 't'), um = gl.getUniformLocation(prog, 'm');
      const size = () => { const b = canvas.getBoundingClientRect(), d = Math.min(devicePixelRatio || 1, 1.5);
        const w = Math.max(1, Math.round(b.width * d)), h = Math.max(1, Math.round(b.height * d));
        if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; gl.viewport(0, 0, w, h); } };
      let seen = true; new IntersectionObserver(e => { seen = e[0].isIntersecting; }).observe(canvas);
      const frame = ms => { if (!alive()) return;
        if (seen && !document.hidden) { size();
          pointer.x += (pointer.tx - pointer.x) * .04; pointer.y += (pointer.ty - pointer.y) * .04;
          gl.uniform2f(ur, canvas.width, canvas.height); gl.uniform1f(ut, ms / 1000 + 20); gl.uniform2f(um, pointer.x, pointer.y);
          gl.drawArrays(gl.TRIANGLES, 0, 3); }
        if (!reduce) requestAnimationFrame(frame); };
      requestAnimationFrame(frame);
    }
  }

  // ---- Card tilts gently towards the pointer.
  const card = $('dml-card');
  if (!reduce) { hero.addEventListener('pointermove', () => {
      if (innerWidth > 980) card.style.transform = `perspective(900px) rotateY(${(pointer.tx - .5) * 5}deg) rotateX(${(pointer.ty - .5) * 4}deg)`; });
    hero.addEventListener('pointerleave', () => { card.style.transform = ''; }); }

  // ---- Tour: tabs advance by themselves until someone chooses one.
  const tabs = [...root.querySelectorAll('.dml-tab')], panels = [...root.querySelectorAll('.dml-panel')];
  const show = key => { tabs.forEach(t => t.classList.toggle('on', t.dataset.tab === key));
    panels.forEach(p => p.classList.toggle('on', p.dataset.panel === key)); };
  let held = reduce; if (reduce) root.classList.add('held');
  const hold = () => { held = true; root.classList.add('held'); };
  tabs.forEach(t => { t.addEventListener('click', () => { hold(); show(t.dataset.tab); });
    t.addEventListener('pointerenter', () => { if (matchMedia('(hover: hover)').matches) { hold(); show(t.dataset.tab); } }); });
  card.addEventListener('pointerenter', hold); card.addEventListener('focusin', hold);
  const auto = setInterval(() => { if (!alive()) return clearInterval(auto); if (held || document.hidden) return;
    const i = tabs.findIndex(t => t.classList.contains('on')); show(tabs[(i + 1) % tabs.length].dataset.tab); }, 7000);

  // ---- Example forecast chart: move across it to read a day.
  const host = $('dml-chart'), n = 42, today = new Date();
  const units = [...Array(n)].map((_, i) => 34 + 9 * Math.sin(i / n * 5 + .6) + 7 * Math.sin(i * 2 * Math.PI / 7) + i * .22);
  const spread = i => 5 + i * .22, top = 78, sx = i => i / (n - 1) * 300, sy = v => 96 - v / top * 90;
  const path = pts => pts.map((p, i) => (i ? 'L' : 'M') + p[0].toFixed(1) + ' ' + p[1].toFixed(1)).join('');
  const up = units.map((u, i) => [sx(i), sy(u + spread(i))]), dn = units.map((u, i) => [sx(i), sy(u - spread(i))]).reverse();
  const chart = svg('svg', {viewBox: '0 0 300 100', preserveAspectRatio: 'none'}, host);
  svg('path', {d: path(up.concat(dn)) + 'Z', fill: 'rgba(255,127,110,.26)'}, chart);
  svg('path', {d: path(units.map((u, i) => [sx(i), sy(u)])), fill: 'none', stroke: '#FF7F6E', 'stroke-width': 2, 'vector-effect': 'non-scaling-stroke'}, chart);
  const cross = svg('line', {y1: 0, y2: 100, stroke: '#F8F8F6', 'stroke-width': 1, 'vector-effect': 'non-scaling-stroke', opacity: .8}, chart);
  const read = i => { const d = new Date(today); d.setDate(d.getDate() + i + 1);
    $('dml-day').textContent = d.toLocaleDateString('en-US', {weekday: 'short', month: 'short', day: 'numeric'});
    $('dml-units').textContent = Math.round(units[i]) + ' units';
    $('dml-band').textContent = Math.round(units[i] - spread(i)) + ' – ' + Math.round(units[i] + spread(i));
    cross.setAttribute('x1', sx(i)); cross.setAttribute('x2', sx(i)); };
  host.addEventListener('pointermove', e => { const b = host.getBoundingClientRect();
    read(Math.max(0, Math.min(n - 1, Math.round((e.clientX - b.left) / b.width * (n - 1))))); });
  read(8);

  // ---- Example price test, using a fixed price sensitivity and a 45% margin.
  const slider = $('dml-slider'), sign = v => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(1) + '%';
  const price = () => { const c = +slider.value / 100, vol = Math.pow(1 + c, -1.4), rev = (1 + c) * vol, profit = vol * (c + .45) / .45;
    $('dml-price').textContent = (c >= 0 ? '+' : '−') + Math.abs(+slider.value) + '%';
    $('dml-profit').textContent = sign((profit - 1) * 100); $('dml-vol').textContent = sign((vol - 1) * 100);
    $('dml-rev').textContent = sign((rev - 1) * 100); };
  slider.addEventListener('input', price); price();

  // ---- Story: sections rise in as they scroll into view; boxes glow under the pointer.
  const wire = () => { if (!alive()) return;
    const items = document.querySelectorAll('.dml-reveal:not([data-w])');
    const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('seen'); io.unobserve(e.target); } }), {threshold: .12});
    items.forEach((el, i) => { el.dataset.w = '1'; el.style.transitionDelay = (i % 3) * 70 + 'ms'; io.observe(el); });
    document.querySelectorAll('.dml-box:not([data-g])').forEach(el => { el.dataset.g = '1';
      el.addEventListener('pointermove', e => { const b = el.getBoundingClientRect();
        el.style.setProperty('--x', e.clientX - b.left + 'px'); el.style.setProperty('--y', e.clientY - b.top + 'px'); }); }); };
  wire(); setTimeout(wire, 500); setTimeout(wire, 1500);
})();
"""
JS = "<script>" + JS_BODY.replace("__ICONS__", json.dumps(ICON)) + "</script>"
