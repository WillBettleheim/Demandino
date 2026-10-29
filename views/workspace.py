import pandas as pd
import streamlit as st

from demandino import account, db, ui
from demandino.scenario import inputs_from_json
from demandino.state import app

A = app()
ui.page_header("Workspace", "Your saved datasets, scenarios, share links and account settings.", eyebrow="Workspace")

uid = account.current_user()
if uid is None:
    st.info("Sign in (sidebar) to save datasets, scenarios, stock tables and alerts. Without an account nothing is "
            "kept after you close the page.", icon=":material/lock:")
    st.stop()

datasets = db.list_datasets(uid)
scenarios = db.list_scenarios(uid)
shares = db.list_shares(uid)
ui.stat_strip([("Saved datasets", str(len(datasets))), ("Saved scenarios", str(len(scenarios))),
               ("Active share links", str(len(shares))), ("Tracked forecasts", str(len(db.list_forecasts(uid))))])

t1, t2, t3, t4 = st.tabs([":material/database: Datasets", ":material/bookmarks: Scenarios", ":material/share: Share links",
                          ":material/manage_accounts: Account"])
with t1:
    if datasets.empty:
        st.caption("Upload a CSV (sidebar → My CSV → Save to my workspace) or import from a store on the "
                   "**Connections & alerts** page.")
    else:
        show = datasets.assign(created=pd.to_datetime(datasets["created"], unit="s"))
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "id": None, "name": "Name", "source": "Source", "created": st.column_config.DatetimeColumn("Saved"),
            "rows": st.column_config.NumberColumn("Rows", format="%,d"), "products": "Products"})
        pick = st.selectbox("Delete a dataset", datasets["id"].tolist(), index=None, placeholder="Choose…",
                            format_func=lambda i: datasets.set_index("id").loc[i, "name"])
        if pick and st.button("Delete permanently", icon=":material/delete:", type="primary"):
            db.delete_dataset(uid, int(pick))
            st.toast("Dataset deleted.")
            st.rerun()

with t2:
    if not scenarios:
        st.caption("Save scenarios from the Forecast page (Save scenario → Save to workspace).")
    for sc in scenarios:
        with ui.card(f"sc-{sc['id']}"):
            c1, c2, c3 = st.columns([3, 1, 1], vertical_alignment="center")
            sm = sc["summary"]
            c1.markdown(f"**{sc['name']}**  \n{sm.get('units', 0):,.0f} units · {ui.money(sm.get('profit', 0))} profit · "
                        f"saved {pd.Timestamp(sc['created'], unit='s'):%d %b %Y}")
            if c2.button("Open", key=f"open_{sc['id']}", icon=":material/open_in_new:", width="stretch"):
                st.session_state["load_scenario"] = inputs_from_json(sc["inputs"])
                st.switch_page("views/forecast.py")
            if c3.button("Delete", key=f"del_{sc['id']}", icon=":material/delete:", width="stretch"):
                db.delete_scenario(uid, sc["id"])
                st.rerun()

with t3:
    if not shares:
        st.caption("Create share links from the Forecast page (Export tab). Links expire after 7 days.")
    for sh in shares:
        c1, c2 = st.columns([4, 1], vertical_alignment="center")
        c1.markdown(f"**{sh['title']}** · expires {pd.Timestamp(sh['expires'], unit='s'):%d %b %Y}")
        if c2.button("Revoke", key=f"rev_{sh['token_hash'][:12]}", icon=":material/link_off:", width="stretch"):
            db.revoke_share(uid, sh["token_hash"])
            st.rerun()

with t4:
    st.markdown(f"Signed in as **{account.current_name()}**.")
    with st.form("pw"):
        st.markdown("##### Change password")
        old = st.text_input("Current password", type="password", max_chars=128)
        new = st.text_input(f"New password (at least {db.MIN_PASSWORD} characters)", type="password", max_chars=128)
        if st.form_submit_button("Change password"):
            try:
                db.change_password(uid, old, new)
                account.sign_out()
                st.success("Password changed. Please sign in again.")
                st.rerun()
            except db.AuthError as exc:
                st.error(str(exc))
    with st.expander("Delete account", icon=":material/warning:"):
        st.warning("This permanently deletes your account and everything saved in it: datasets, scenarios, stock "
                   "tables, alerts, connector keys, tracked forecasts, share links and purchase orders.")
        confirm = st.text_input("Type DELETE to confirm", max_chars=10)
        if st.button("Delete my account", type="primary", disabled=confirm != "DELETE", icon=":material/delete_forever:"):
            db.delete_user(uid)
            account.sign_out()
            st.rerun()
