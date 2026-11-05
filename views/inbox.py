import streamlit as st

from demandino import db, inbox, ui
from demandino.state import app

A = app()
ui.page_header("Inbox", "Everything that needs your attention, from every part of Demandino, in one place — "
               "most urgent first.", eyebrow="Dashboard")
saved = db.get_setting(A.user_id, f"stock:{A.settings_key}") if A.user_id else None
items = inbox.gather(A, ui.currency())

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
        c1.markdown(f"**{icons[i['priority']]} {i['title']}**  \n{ui.md_safe(i['text'])}")
        c2.page_link(i["page"], label="Open", icon=":material/arrow_forward:")
if not saved:
    st.caption("Ordering items use illustrative stock levels until you save your own on Demand planning.")
