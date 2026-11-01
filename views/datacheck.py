import streamlit as st

from demandino import quality, ui

ui.page_header("Data check",
               "Before the model learns from your file, Demandino checks it for common problems and asks how to "
               "fix each one — nothing is changed silently.", eyebrow="Workspace")

report = st.session_state.get("_dq")
if report is None:
    st.info("Upload your own sales file in the sidebar (**Data → My CSV**) to check it. The built-in sample and "
            "real-store data are already clean.", icon=":material/upload_file:")
    st.download_button("Download a messy example file to try", quality.messy_example(), "messy_sales_example.csv",
                       "text/csv", icon=":material/bug_report:")
    st.stop()

issues = report["issues"]
todo = [i for i in issues if i.severity == "warn"]
choices = st.session_state.setdefault("dq_choices", {})
ui.spotlight(f"{report['name']}", str(len(todo)), "thing" + ("" if len(todo) == 1 else "s") + " to review",
             f"After your fixes: <b>{report['products']}</b> products, {report['days']:,} product-days, "
             f"{report['start']:%d %b %Y} → {report['end']:%d %b %Y}",
             [("Ready to forecast", "good")] if not todo else
             [(f"{sum(i.count for i in todo):,} rows or days affected", "warn")])

if not issues:
    st.success("No problems found. Your data is ready.", icon=":material/check_circle:")

for issue in issues:
    with ui.card(f"dq-{issue.code}"):
        icon = ":material/info:" if issue.severity == "info" else ":material/warning:"
        st.markdown(f"**{icon} {issue.title}** · {issue.count:,}")
        st.caption(issue.explain)
        if issue.fixes:
            options = list(issue.fixes)
            current = choices.get(issue.code, issue.default)
            picked = st.radio("Fix", options, index=options.index(current) if current in options else 0,
                              format_func=issue.fixes.get, key=f"dq_{issue.code}", horizontal=True)
            if picked != current:
                choices[issue.code] = picked
                st.rerun()  # reloads the data with the new fix (the model retrains for the new version)
        if issue.examples is not None and len(issue.examples):
            with st.expander("Examples", icon=":material/table_rows:"):
                st.dataframe(issue.examples, hide_index=True, width="stretch")

if choices and st.button("Reset all fixes to the defaults", icon=":material/restart_alt:"):
    st.session_state["dq_choices"] = {}
    for issue in issues:
        st.session_state.pop(f"dq_{issue.code}", None)
    st.rerun()
st.caption("Fixes apply only inside Demandino; your original file isn't changed. Save the cleaned data to your "
           "workspace from the sidebar to keep it.")
