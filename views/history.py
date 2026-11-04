import pandas as pd
import streamlit as st

from demandino import account, db, ui
from demandino.security import safe_csv

ui.page_header("Change history", "A record of what was changed in your workspace and when: stock tables, targets, "
               "scenarios, purchase orders and settings.", eyebrow="Settings")
uid = account.current_user()
if uid is None:
    st.info("Sign in (sidebar) to keep a history of your changes.", icon=":material/lock:")
    st.stop()
log = db.list_activity(uid)
if not log:
    st.caption("Nothing recorded yet. Save a stock table, a target or a scenario, or approve a purchase order, and "
               "it will appear here.")
    st.stop()
df = pd.DataFrame(log)
df["when"] = pd.to_datetime(df["at"], unit="s")
kinds = sorted(df["kind"].unique())
pick = st.pills("Show", kinds, selection_mode="multi", default=kinds, key="hist_kinds")
view = df[df["kind"].isin(pick or kinds)][["when", "kind", "detail"]]
st.dataframe(view, hide_index=True, width="stretch", height=min(42 + 35 * len(view), 640), column_config={
    "when": st.column_config.DatetimeColumn("When", format="D MMM YYYY, HH:mm"), "kind": "Area", "detail": "What changed"})
st.download_button("Download history (CSV)", safe_csv(view), "change_history.csv", "text/csv",
                   icon=":material/download:")
