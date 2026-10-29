import pandas as pd
import streamlit as st

from demandino import account, db, integrations as ig, ui
from demandino.planning import default_stock, portfolio_plan
from demandino.security import clean_text
from demandino.state import app

A = app()
ui.page_header("Connections & alerts",
               "Pull sales straight from your store, and get told when something needs attention — in Slack or by "
               "email.", eyebrow="Workspace")

uid = account.current_user()
if uid is None:
    st.info("Sign in (sidebar) to connect a store or set up alerts. Keys you enter are stored encrypted.",
            icon=":material/lock:")
    st.stop()


def save_import(sales: pd.DataFrame, name: str, source: str) -> None:
    db.save_dataset(uid, clean_text(name) or source, sales, source)
    st.success(f"Imported {sales['product_id'].nunique()} products, {sales['date'].min():%d %b %Y} → "
               f"{sales['date'].max():%d %b %Y}. Choose **Workspace** under Data in the sidebar to use it.",
               icon=":material/check_circle:")


since = pd.Timestamp.today().normalize() - pd.Timedelta(days=730)
t1, t2, t3, t4, t5 = st.tabs([":material/table_chart: Google Sheets", ":material/storefront: Shopify",
                              ":material/shopping_bag: WooCommerce", ":material/package_2: Amazon",
                              ":material/notifications: Alerts & digest"])
with t1:
    st.caption("Share the sheet as *Anyone with the link can view*. Columns as in the CSV template.")
    url = st.text_input("Sheet link", placeholder="https://docs.google.com/spreadsheets/d/…",
                        value=db.get_setting(uid, "gsheet_url", ""), max_chars=300)
    name = st.text_input("Save as", "Google Sheet sales", max_chars=60, key="gs_name")
    if st.button("Import sheet", type="primary", icon=":material/download:", disabled=not url):
        try:
            with st.spinner("Reading the sheet…"):
                sales = ig.google_sheet(url)
            db.set_setting(uid, "gsheet_url", url)
            save_import(sales, name, "google-sheets")
        except (ig.ConnectorError, ValueError) as exc:
            st.error(str(exc)[:300])

with t2:
    st.caption("In Shopify admin: Settings → Apps → Develop apps → create an app with the **read_orders** scope, then "
               "copy its Admin API access token. Imports the last 2 years of orders.")
    domain = st.text_input("Store address", value=db.get_setting(uid, "shopify_domain", ""),
                           placeholder="my-shop.myshopify.com", max_chars=80)
    have = db.get_secret(uid, "shopify_token") is not None
    token = st.text_input("Admin API access token" + (" (saved — leave blank to keep)" if have else ""),
                          type="password", max_chars=200)
    version = st.text_input("API version", db.get_setting(uid, "shopify_version", "2026-07"), max_chars=7)
    if st.button("Import from Shopify", type="primary", icon=":material/download:", disabled=not domain):
        try:
            if token:
                db.set_secret(uid, "shopify_token", token)
            db.set_setting(uid, "shopify_domain", domain)
            db.set_setting(uid, "shopify_version", version)
            tok = token or db.get_secret(uid, "shopify_token")
            if not tok:
                raise ig.ConnectorError("Enter the access token.")
            with st.spinner("Fetching orders from Shopify…"):
                sales = ig.shopify(domain, tok, since, version)
            save_import(sales, f"Shopify · {domain}", "shopify")
        except (ig.ConnectorError, ValueError) as exc:
            st.error(str(exc)[:300])

with t3:
    st.caption("In WordPress: WooCommerce → Settings → Advanced → REST API → add a **Read** key.")
    site = st.text_input("Store URL", value=db.get_setting(uid, "woo_site", ""), placeholder="https://shop.example.com",
                         max_chars=200)
    have = db.get_secret(uid, "woo_key") is not None
    ck = st.text_input("Consumer key" + (" (saved)" if have else ""), type="password", max_chars=200)
    cs = st.text_input("Consumer secret" + (" (saved)" if have else ""), type="password", max_chars=200)
    if st.button("Import from WooCommerce", type="primary", icon=":material/download:", disabled=not site):
        try:
            if ck and cs:
                db.set_secret(uid, "woo_key", ck)
                db.set_secret(uid, "woo_secret", cs)
            db.set_setting(uid, "woo_site", site)
            k, s_ = ck or db.get_secret(uid, "woo_key"), cs or db.get_secret(uid, "woo_secret")
            if not (k and s_):
                raise ig.ConnectorError("Enter the consumer key and secret.")
            with st.spinner("Fetching orders from WooCommerce…"):
                sales = ig.woocommerce(site, k, s_, since)
            save_import(sales, f"WooCommerce · {site}", "woocommerce")
        except (ig.ConnectorError, ValueError) as exc:
            st.error(str(exc)[:300])

with t4:
    st.caption("Amazon doesn't offer a simple key for this, so upload a report instead: Seller Central → Reports → "
               "Fulfillment (or Order Reports) → **All Orders**, as a tab-delimited or CSV file.")
    rep = st.file_uploader("All Orders report", type=["txt", "csv", "tsv"])
    if rep is not None and st.button("Import report", type="primary", icon=":material/upload:"):
        try:
            if len(rep.getvalue()) > 25 * 1024 * 1024:
                raise ig.ConnectorError("File is larger than 25 MB.")
            save_import(ig.amazon_report(rep.getvalue()), "Amazon orders", "amazon")
        except (ig.ConnectorError, ValueError) as exc:
            st.error(str(exc)[:300])

with t5:
    cfg = db.get_setting(uid, "alerts", {}) or {}
    c1, c2, c3 = st.columns(3)
    so_days = c1.number_input("Warn when stock runs out within (days)", 1, 90, int(cfg.get("stockout_days", 14)))
    change = c2.number_input("Warn on demand change of (%)", 5, 500, int(cfg.get("change_pct", 30)))
    horizon = c3.select_slider("Digest horizon (days)", [30, 60, 90], int(cfg.get("horizon", 30)))
    st.markdown("##### Slack")
    has_slack = db.get_secret(uid, "slack_webhook") is not None
    hook = st.text_input("Incoming webhook URL" + (" (saved)" if has_slack else ""), type="password", max_chars=300,
                         help="Slack → Apps → Incoming Webhooks → Add to a channel.")
    st.markdown("##### Email")
    e1, e2 = st.columns(2)
    smtp_host = e1.text_input("SMTP server", cfg.get("smtp_host", ""), placeholder="smtp.gmail.com", max_chars=190)
    smtp_port = e2.number_input("Port", 1, 65535, int(cfg.get("smtp_port", 587)), help="587 (STARTTLS) or 465 (SSL).")
    smtp_user = e1.text_input("SMTP username", cfg.get("smtp_user", ""), max_chars=190)
    has_pw = db.get_secret(uid, "smtp_password") is not None
    smtp_pw = e2.text_input("SMTP password / app password" + (" (saved)" if has_pw else ""), type="password",
                            max_chars=200)
    sender = e1.text_input("From address", cfg.get("sender", ""), max_chars=190)
    to = e2.text_input("Send to (comma-separated)", cfg.get("to", ""), max_chars=400)

    if st.button("Save alert settings", icon=":material/save:"):
        cfg = {"stockout_days": int(so_days), "change_pct": int(change), "horizon": int(horizon),
               "smtp_host": smtp_host.strip(), "smtp_port": int(smtp_port), "smtp_user": smtp_user.strip(),
               "sender": sender.strip(), "to": to.strip(), "dataset_key": A.data_key}
        db.set_setting(uid, "alerts", cfg)
        if hook:
            db.set_secret(uid, "slack_webhook", hook.strip())
        if smtp_pw:
            db.set_secret(uid, "smtp_password", smtp_pw)
        st.toast("Alert settings saved.", icon=":material/check:")

    stock = db.get_setting(uid, f"stock:{A.data_key}")
    stock = pd.DataFrame(stock) if stock else default_stock(A.sales)
    plan, _ = portfolio_plan(A.fc, A.level_uncertainty, stock, int(horizon), 14, 0.95)
    lines = ig.alert_lines(plan, int(so_days), float(change))
    digest = ig.digest_text(plan, int(horizon), ui.currency())
    st.markdown(f"##### Preview · {len(lines)} alerts for the current data")
    st.code(digest, language=None)
    if not db.get_setting(uid, f"stock:{A.data_key}"):
        st.caption("Using illustrative stock levels — save your real ones on the Demand planning page.")
    b1, b2 = st.columns(2)
    if b1.button("Send to Slack now", icon=":material/send:", width="stretch"):
        try:
            url = hook.strip() or db.get_secret(uid, "slack_webhook")
            if not url:
                raise ig.ConnectorError("Add a Slack webhook first.")
            ig.send_slack(url, digest)
            st.success("Sent to Slack.")
        except ig.ConnectorError as exc:
            st.error(str(exc))
    if b2.button("Send email now", icon=":material/mail:", width="stretch"):
        try:
            pw = smtp_pw or db.get_secret(uid, "smtp_password")
            if not (smtp_host and smtp_user and pw and sender and to):
                raise ig.ConnectorError("Fill in the email settings first.")
            ig.send_email(smtp_host, int(smtp_port), smtp_user, pw, sender, to, "Demandino digest", digest)
            st.success("Email sent.")
        except ig.ConnectorError as exc:
            st.error(str(exc))
    with st.expander("Send automatically every week", icon=":material/schedule:"):
        st.markdown("Run the digest script on a schedule on the machine hosting Demandino, e.g. every Monday 8:00 "
                    "with cron:")
        st.code("0 8 * * 1  cd /path/to/Demandino && .venv/bin/python scripts/send_digests.py", language="bash")
        st.caption("The script sends each user's digest to the Slack channel and/or email saved here.")
