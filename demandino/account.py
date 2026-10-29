"""Sign-in, sign-up and sign-out in the sidebar; keeps the user signed in across reloads via a session cookie."""
from __future__ import annotations

import streamlit as st

from . import db

COOKIE = "dm_session"
MAX_SESSION_ATTEMPTS = 10


def _cookie_token() -> str | None:
    try:
        return st.context.cookies.get(COOKIE)
    except Exception:  # noqa: BLE001 - no browser context (tests)
        return None


def current_user() -> int | None:
    uid = st.session_state.get("user_id")
    if uid is None and not st.session_state.get("signed_out"):
        uid = db.session_user(_cookie_token())
        if uid is not None:
            st.session_state["user_id"] = uid
            st.session_state["session_token"] = _cookie_token()
    return uid


def current_name() -> str | None:
    uid = current_user()
    return db.username(uid) if uid is not None else None


def _sign_in(uid: int) -> None:
    token = db.new_session(uid)
    st.session_state.update(user_id=uid, session_token=token, signed_out=False, _session_cookie=("set", token))


def sign_out() -> None:
    db.end_session(st.session_state.get("session_token"))
    for k in ("user_id", "session_token", "dataset_id"):
        st.session_state.pop(k, None)
    st.session_state["signed_out"] = True
    st.session_state["_session_cookie"] = ("clear", "")


def sync_cookie() -> None:
    action = st.session_state.pop("_session_cookie", None)
    if not action:
        return
    kind, token = action
    if kind == "set" and token.replace("-", "").replace("_", "").isalnum():
        st.html(f"<script>document.cookie='{COOKIE}={token}; Max-Age={db.SESSION_DAYS * 86400}; Path=/; "
                "SameSite=Strict';</script>", unsafe_allow_javascript=True)
    elif kind == "clear":
        st.html(f"<script>document.cookie='{COOKIE}=; Max-Age=0; Path=/; SameSite=Strict';</script>",
                unsafe_allow_javascript=True)


def _attempt_allowed() -> bool:
    n = st.session_state.get("_auth_attempts", 0)
    st.session_state["_auth_attempts"] = n + 1
    return n < MAX_SESSION_ATTEMPTS


def sidebar_account() -> None:
    """Account controls for the sidebar."""
    name = current_name()
    if name:
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.markdown(f":material/account_circle: **{name}**")
        if c2.button("Sign out", key="sign_out", width="stretch"):
            sign_out()
            st.rerun()
        return
    with st.popover("Sign in to save your work", icon=":material/login:", width="stretch"):
        tab_in, tab_up = st.tabs(["Sign in", "Create account"])
        with tab_in, st.form("sign_in", border=False):
            user = st.text_input("Username", max_chars=32, key="si_user")
            pw = st.text_input("Password", type="password", max_chars=128, key="si_pw")
            if st.form_submit_button("Sign in", type="primary", width="stretch"):
                if not _attempt_allowed():
                    st.error("Too many attempts in this session. Reload the page later.")
                else:
                    try:
                        _sign_in(db.authenticate(user, pw))
                        st.rerun()
                    except db.AuthError as exc:
                        st.error(str(exc))
        with tab_up, st.form("sign_up", border=False):
            st.caption("Accounts are stored on the server running this app. Your saved datasets, scenarios and "
                       "settings stay private to you.")
            user = st.text_input("Choose a username", max_chars=32, key="su_user")
            pw = st.text_input(f"Password (at least {db.MIN_PASSWORD} characters)", type="password", max_chars=128,
                               key="su_pw")
            pw2 = st.text_input("Repeat password", type="password", max_chars=128, key="su_pw2")
            if st.form_submit_button("Create account", type="primary", width="stretch"):
                if pw != pw2:
                    st.error("The passwords don't match.")
                elif not _attempt_allowed():
                    st.error("Too many attempts in this session. Reload the page later.")
                else:
                    try:
                        _sign_in(db.create_user(user, pw))
                        st.rerun()
                    except db.AuthError as exc:
                        st.error(str(exc))
