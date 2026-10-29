import pandas as pd
import streamlit as st

from demandino import account, db, ui
from demandino.security import safe_csv
from demandino.state import app

A = app()
ui.page_header("Purchase orders", "Review purchase lists sent from Demand planning, then approve, reject or mark them "
                                  "as ordered.", eyebrow="Workspace")
uid = account.current_user()
if uid is None:
    st.info("Sign in (sidebar) to create and approve purchase orders.", icon=":material/lock:")
    st.stop()

pos = db.list_pos(uid)
if not pos:
    st.caption("No purchase orders yet. On **Demand planning**, use *Send purchase list for approval*.")
    st.stop()

status_chip = {"Draft": "warn", "Approved": "good", "Ordered": "acc", "Rejected": "bad"}
ui.stat_strip([(s, str(sum(p["status"] == s for p in pos))) for s in ("Draft", "Approved", "Ordered", "Rejected")])
for po in pos:
    lines = pd.DataFrame(po["lines"])
    total = float(lines["order_cost"].sum()) if "order_cost" in lines else 0.0
    with ui.card(f"po-{po['id']}"):
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.html(f'<b>PO #{po["id"]}</b> · {pd.Timestamp(po["created"], unit="s"):%d %b %Y %H:%M} '
                f'<span class="dm-chip {status_chip.get(po["status"], "")}">{po["status"]}</span>')
        c1.caption(f"{len(lines)} lines · {ui.money(total)}"
                   + (f" · {po['status'].lower()} by {po['decided_by']}" if po.get("decided_by") else ""))
        with c2.container(horizontal=True):
            if po["status"] == "Draft":
                if st.button("Approve", key=f"ap_{po['id']}", type="primary", icon=":material/check:"):
                    db.decide_po(uid, po["id"], "Approved", account.current_name() or "")
                    st.rerun()
                if st.button("Reject", key=f"rj_{po['id']}", icon=":material/close:"):
                    db.decide_po(uid, po["id"], "Rejected", account.current_name() or "")
                    st.rerun()
            elif po["status"] == "Approved":
                if st.button("Mark as ordered", key=f"or_{po['id']}", icon=":material/local_shipping:"):
                    db.decide_po(uid, po["id"], "Ordered", account.current_name() or "")
                    st.rerun()
            st.download_button("CSV", safe_csv(lines), f"purchase_order_{po['id']}.csv", "text/csv",
                               key=f"dl_{po['id']}", icon=":material/download:")
        with st.expander("Lines"):
            st.dataframe(lines, hide_index=True, width="stretch")
