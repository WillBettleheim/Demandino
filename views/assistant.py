import os

import streamlit as st

from demandino import account, assistant, db, ui
from demandino.state import app

A = app()
ui.page_header("Ask Demandino",
               "Ask questions about the forecast you're looking at, in plain language — “Why is December higher?”, "
               "“What if I raise the price 10%?”, “When should I reorder?”", eyebrow="AI assistant")

r = st.session_state.get("last_result")
if r is None:
    st.info("Open a product on the **Forecast a product** page first — the assistant answers about that forecast.",
            icon=":material/online_prediction:")
    st.stop()

uid = account.current_user()
key = os.environ.get("ANTHROPIC_API_KEY") or (db.get_secret(uid, "anthropic_api_key") if uid else None)
with st.expander("Assistant settings & privacy", icon=":material/shield:", expanded=not key):
    st.caption("Answers come from Claude (Anthropic). Each question sends a summary of the forecast on screen — "
               "product name, numbers, dates and findings — not your raw sales files. Anthropic's terms apply.")
    if os.environ.get("ANTHROPIC_API_KEY"):
        st.caption("Using the server's ANTHROPIC_API_KEY.")
    elif uid:
        new_key = st.text_input("Anthropic API key" + (" (saved — leave blank to keep)" if key else ""),
                                type="password", max_chars=200, help="Stored encrypted in your workspace.")
        c1, c2 = st.columns(2)
        if c1.button("Save key", icon=":material/key:", disabled=not new_key):
            db.set_secret(uid, "anthropic_api_key", new_key.strip())
            st.rerun()
        if key and c2.button("Remove key", icon=":material/key_off:"):
            db.delete_secret(uid, "anthropic_api_key")
            st.rerun()
    else:
        st.caption("Sign in to save an API key, or set ANTHROPIC_API_KEY on the server.")
if not key:
    st.info("Add an Anthropic API key above to turn the assistant on.", icon=":material/key:")
    st.stop()

ui.stat_strip([("Product", r.inputs.name), ("Horizon", f"{r.inputs.horizon} days"),
               ("Expected units", ui.num(r.total)), ("Profit", ui.money(r.profit))])

chat_key = f"chat::{r.inputs.name}::{r.inputs.start:%Y%m%d}::{r.inputs.horizon}"
history = st.session_state.setdefault(chat_key, [])
for m in history:
    with st.chat_message(m["role"], avatar=":material/person:" if m["role"] == "user" else ":material/auto_awesome:"):
        st.markdown(m["content"])

suggest = None
if not history:
    suggest = st.pills("Try asking", ["Why does demand peak when it does?", "What if I raise the price by 10%?",
                                      "Is the planned promotion worth it?", "When should I reorder, and how much?"],
                       key=f"sugg_{chat_key}")
question = st.chat_input("Ask about this forecast…", max_chars=assistant.MAX_QUESTION) or suggest
if question:
    if len(history) >= 2 * assistant.MAX_TURNS * 2:
        st.warning("This conversation is long — clear it to start fresh.")
    else:
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(question)
        summary = assistant.forecast_summary(r, ui.currency(), A.backtest["metrics"])
        with st.chat_message("assistant", avatar=":material/auto_awesome:"):
            answer = st.write_stream(assistant.ask(assistant.client_for(key), summary, history, question))
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": str(answer)}]
if history and st.button("Clear conversation", icon=":material/delete_sweep:"):
    st.session_state[chat_key] = []
    st.rerun()
