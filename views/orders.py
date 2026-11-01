import pandas as pd
import streamlit as st

from demandino import account, db, suppliers, ui
from demandino.security import safe_csv
from demandino.state import app

A = app()


def _days(start: float, end: float) -> int:
    return (pd.Timestamp(end, unit="s").normalize() - pd.Timestamp(start, unit="s").normalize()).days


ui.page_header("Purchase orders", "Review purchase lists sent from Demand planning, approve them, and record when "
                                  "they arrive — Demandino learns each supplier's real delivery times from that.",
               eyebrow="Workspace")
uid = account.current_user()
if uid is None:
    st.info("Sign in (sidebar) to create and approve purchase orders.", icon=":material/lock:")
    st.stop()

pos = db.list_pos(uid)
if not pos:
    st.caption("No purchase orders yet. On **Demand planning**, use *Send purchase list for approval*.")
    st.stop()

status_chip = {"Draft": "warn", "Approved": "good", "Ordered": "acc", "Received": "good", "Rejected": "bad"}
ui.stat_strip([(s, str(sum(p["status"] == s for p in pos))) for s in ("Draft", "Approved", "Ordered", "Received",
                                                                       "Rejected")])

st.subheader("Supplier delivery times")
sup = suppliers.stats(suppliers.deliveries(pos))
if sup.empty:
    st.caption("Mark orders as **received** below (with the date they arrived) and each supplier's real delivery "
               "time appears here. Demand planning then uses it instead of the lead time you typed in.")
else:
    slower = sup[sup["trend"] == "Getting slower"]
    if len(slower):
        st.warning("Getting slower: " + ", ".join(slower["supplier"]) + ". Order earlier until it settles.",
                   icon=":material/hourglass_bottom:")
    st.dataframe(sup, hide_index=True, width="stretch", column_config={
        "supplier": "Supplier", "deliveries": "Deliveries",
        "avg_days": st.column_config.NumberColumn("Average days", format="%.1f"),
        "sd_days": st.column_config.NumberColumn("Varies by ± days", format="%.1f"),
        "late_share": st.column_config.NumberColumn("Late vs plan", format="percent"),
        "planning_lead": st.column_config.NumberColumn("Plan with (days)", format="%d",
                                                       help="Average plus a margin for the variation, at 95%."),
        "trend": "Trend", "advice": "What to do"})
st.subheader("Orders")
for po in pos:
    lines = pd.DataFrame(po["lines"])
    total = float(lines["order_cost"].sum()) if "order_cost" in lines else 0.0
    with ui.card(f"po-{po['id']}"):
        c1, c2 = st.columns([3, 2], vertical_alignment="center")
        c1.html(f'<b>PO #{po["id"]}</b> · {pd.Timestamp(po["created"], unit="s"):%d %b %Y %H:%M} '
                f'<span class="dm-chip {status_chip.get(po["status"], "")}">{po["status"]}</span>')
        took = f" · arrived in {_days(po['ordered_at'], po['received_at'])} days" if po.get("received_at") else ""
        c1.caption(f"{len(lines)} lines · {ui.money(total)}"
                   + (f" · {po['status'].lower()} by {po['decided_by']}" if po.get("decided_by") else "") + took)
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
            if po["status"] in ("Approved", "Ordered", "Received"):
                with st.popover("Received" if po["status"] == "Received" else "Mark as received",
                                icon=":material/inventory_2:"):
                    placed = pd.Timestamp(po.get("ordered_at") or po.get("decided_at") or po["created"], unit="s")
                    arrived = pd.Timestamp(po["received_at"], unit="s") if po.get("received_at") else pd.Timestamp.now()
                    d1 = st.date_input("Ordered on", placed.date(), key=f"od_{po['id']}")
                    d2 = st.date_input("Arrived on", arrived.date(), key=f"rd_{po['id']}", min_value=d1)
                    if st.button("Save", key=f"rc_{po['id']}", type="primary", icon=":material/save:"):
                        db.receive_po(uid, po["id"], pd.Timestamp(d1).timestamp(), pd.Timestamp(d2).timestamp(),
                                      account.current_name() or "")
                        st.rerun()
            st.download_button("CSV", safe_csv(lines), f"purchase_order_{po['id']}.csv", "text/csv",
                               key=f"dl_{po['id']}", icon=":material/download:")
        with st.expander("Lines"):
            st.dataframe(lines, hide_index=True, width="stretch")
