import pandas as pd
import streamlit as st

from demandino import account, db, ui

ui.page_header("Security", "Who's signed in to your account, which keys are stored, and what has happened recently.",
               eyebrow="Settings")
uid = account.current_user()
if uid is None:
    st.info("Sign in (sidebar) to see your account's security.", icon=":material/lock:")
    st.stop()

sessions = db.session_count(uid)
keys = db.secret_names(uid)
events = [e for e in db.list_activity(uid) if e["kind"] == "Security"]
failed = sum("Failed" in e["detail"] for e in events[:50])
ui.stat_strip([("Signed-in browsers", str(sessions)), ("Stored keys", str(len(keys))),
               ("Recent failed sign-ins", str(failed))])
if failed:
    st.warning("There were failed sign-in attempts on your account. If they weren't you, change your password and "
               "sign out everywhere.", icon=":material/gpp_maybe:")

c1, c2 = st.columns(2)
with c1, ui.card("sec-sessions"):
    st.markdown("**:material/devices: Signed-in browsers**")
    st.caption("If you used a shared computer or lost a device, sign out everywhere. You'll need to sign in again too.")
    if st.button("Sign out everywhere", icon=":material/logout:", key="sec_out_all"):
        db.end_all_sessions(uid)
        account.sign_out()
        st.rerun()
with c2, ui.card("sec-keys"):
    st.markdown("**:material/key: Stored keys**")
    st.caption("Encrypted on the server. Values are never shown again once saved, here or anywhere else.")
    if not keys:
        st.caption("None stored.")
    for name in keys:
        a, b = st.columns([3, 1], vertical_alignment="center")
        a.markdown(f"`{ui.md_safe(name)}` · ••••••••")
        if b.button("Remove", key=f"sec_rm_{name}", icon=":material/delete:", width="stretch"):
            db.delete_secret(uid, name)
            st.rerun()

st.subheader("Recent security events")
if events:
    df = pd.DataFrame(events)
    df["when"] = pd.to_datetime(df["at"], unit="s")
    st.dataframe(df[["when", "detail"]].head(50), hide_index=True, width="stretch", column_config={
        "when": st.column_config.DatetimeColumn("When", format="D MMM YYYY, HH:mm"), "detail": "What happened"})
else:
    st.caption("Nothing yet.")
