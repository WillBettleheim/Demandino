import pandas as pd
import streamlit as st

from demandino import db, inbox, lifecycle, lostsales, targets, ui
from demandino.planning import default_stock, portfolio_plan
from demandino.state import app

A = app()


@st.cache_data(show_spinner="Gathering everything that needs attention…", max_entries=8)
def plan_for(data_key: str, stock: pd.DataFrame) -> pd.DataFrame:
    return portfolio_plan(A.fc, A.level_uncertainty, stock, 30, 14, 0.95)[0]


ui.page_header("Inbox", "Everything that needs your attention, from every part of Demandino, in one place — "
               "most urgent first.", eyebrow="Workspace")
saved = db.get_setting(A.user_id, f"stock:{A.data_key}") if A.user_id else None
stock = st.session_state.get(f"pl_stock_{A.data_key}")
stock = stock if stock is not None else (pd.DataFrame(saved) if saved else default_stock(A.sales))
plan = plan_for(A.data_key, stock)
stages = lifecycle.stages(A.sales).set_index("product_id")["stage"].astype(str).to_dict()
last = A.fc.sales["date"].max()
lost = lostsales.window(lostsales.daily(A.fc.sales), last, 14)
saved_targets = []
for t in (db.get_setting(A.user_id, f"targets:{A.base_key}", []) if A.user_id else []):
    p = targets.progress(A.fc, t["products"], pd.Timestamp(t["start"]), pd.Timestamp(t["end"]), t["metric"])
    saved_targets.append({**t, "chance": targets.chance(p, t["target"])})
items = inbox.collect(plan, A.drift()[0], stages, lost, saved_targets, ui.currency())

counts = {k: sum(i["priority"] == k for i in items) for k in inbox.PRIORITY}
ui.spotlight("Needs attention", str(len(items)), "item" + ("" if len(items) == 1 else "s"),
             f"<b>{counts['now']}</b> to act on now · <b>{counts['week']}</b> this week · {counts['info']} for "
             "information", [("All clear", "good")] if not items else
             [(f"{counts['now']} now", "bad")] * bool(counts["now"]) + [(f"{counts['week']} this week", "warn")]
             * bool(counts["week"]))
if not items:
    st.success("Nothing needs attention right now.", icon=":material/check_circle:")
icons = {"now": ":material/priority_high:", "week": ":material/schedule:", "info": ":material/info:"}
for k, i in enumerate(items):
    with ui.card(f"inbox-{k}"):
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.markdown(f"**{icons[i['priority']]} {i['title']}**  \n{i['text']}".replace("$", "\\$"))
        c2.page_link(i["page"], label="Open", icon=":material/arrow_forward:")
if not saved:
    st.caption("Ordering items use illustrative stock levels until you save your own on Demand planning.")
