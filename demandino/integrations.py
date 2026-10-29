"""Store connectors (Google Sheets, Shopify, WooCommerce, Amazon order reports) and alerts (Slack, email).

Every outbound request is HTTPS-only, restricted to the service's own hosts, refuses redirects to
other hosts, has a timeout and a response-size cap. Credentials are passed in by the caller (the app
stores them encrypted) and are never logged.
"""
from __future__ import annotations

import base64
import io
import json
import re
import smtplib
import ssl
import urllib.parse
import urllib.request
from email.message import EmailMessage

import pandas as pd

from .data import prepare_sales
from .security import UploadError, check_sales_limits, read_csv_upload

TIMEOUT = 30
MAX_BYTES = 25 * 1024 * 1024
SHOPIFY_DOMAIN = re.compile(r"^[a-z0-9][a-z0-9-]{0,60}\.myshopify\.com$")
EMAIL = re.compile(r"^[^@\s]{1,64}@[A-Za-z0-9.-]{1,190}\.[A-Za-z]{2,}$")


class ConnectorError(ValueError):
    """Message is safe to show to the user."""


# ------------------------------------------------------------------ HTTP
class _SameHostRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if urllib.parse.urlparse(newurl).hostname != urllib.parse.urlparse(req.full_url).hostname:
            raise ConnectorError("The service redirected to a different site; refusing to follow.")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = urllib.request.build_opener(_SameHostRedirect)


def _check_url(url: str, hosts: tuple[str, ...] | None = None, pattern: re.Pattern | None = None) -> str:
    u = urllib.parse.urlparse(url)
    host = (u.hostname or "").lower()
    if u.scheme != "https" or not host or u.username or u.password:
        raise ConnectorError("Only https:// links are allowed.")
    if hosts and host not in hosts:
        raise ConnectorError(f"Only links on {', '.join(hosts)} are allowed.")
    if pattern and not pattern.match(host):
        raise ConnectorError("That address is not allowed for this connector.")
    if re.match(r"^\d+\.\d+\.\d+\.\d+$", host) or host in ("localhost",) or host.endswith((".local", ".internal")):
        raise ConnectorError("Private or numeric addresses are not allowed.")
    return url


def _get(url: str, headers: dict | None = None) -> tuple[bytes, dict]:
    req = urllib.request.Request(url, headers={"User-Agent": "Demandino/1.0", **(headers or {})})
    try:
        with _opener.open(req, timeout=TIMEOUT) as resp:
            data = resp.read(MAX_BYTES + 1)
            if len(data) > MAX_BYTES:
                raise ConnectorError("The response is larger than 25 MB.")
            return data, dict(resp.headers)
    except ConnectorError:
        raise
    except urllib.error.HTTPError as exc:
        hint = {401: "check the access key", 403: "the key lacks permission", 404: "check the address"}.get(exc.code, "")
        raise ConnectorError(f"The service answered {exc.code}" + (f" — {hint}." if hint else ".")) from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ConnectorError("Could not reach the service. Check the address and your connection.") from None


def _post_json(url: str, payload: dict) -> None:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "User-Agent": "Demandino/1.0"})
    try:
        with _opener.open(req, timeout=TIMEOUT) as resp:
            resp.read(10_000)
    except urllib.error.HTTPError as exc:
        raise ConnectorError(f"The service answered {exc.code}.") from None
    except (urllib.error.URLError, TimeoutError, OSError):
        raise ConnectorError("Could not reach the service.") from None


def _finish(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        raise ConnectorError("No sales found for that period.")
    df = pd.DataFrame(rows)
    df = df.groupby(["date", "product_id"], as_index=False).agg(
        units_sold=("units_sold", "sum"), price=("price", "mean"), product_name=("product_name", "first"),
        category=("category", "first"))
    sales = prepare_sales(df)
    check_sales_limits(sales)
    return sales


# ------------------------------------------------------------ connectors
def google_sheet(url: str) -> pd.DataFrame:
    """A Google Sheet shared as 'anyone with the link' (or published), in the template's columns."""
    _check_url(url, hosts=("docs.google.com",))
    m = re.search(r"/spreadsheets/d/([A-Za-z0-9_-]{20,})", url)
    if not m:
        raise ConnectorError("That doesn't look like a Google Sheets link.")
    gid = re.search(r"[#&?]gid=(\d+)", url)
    export = f"https://docs.google.com/spreadsheets/d/{m.group(1)}/export?format=csv" + (f"&gid={gid.group(1)}" if gid else "")
    data, _ = _get(export)
    if data.lstrip()[:15].lower().startswith((b"<!doctype", b"<html")):
        raise ConnectorError("The sheet isn't public. Share it as 'Anyone with the link can view'.")
    try:
        sales = prepare_sales(read_csv_upload(data))
    except UploadError as exc:
        raise ConnectorError(str(exc)) from None
    check_sales_limits(sales)
    return sales


def shopify(domain: str, token: str, since: pd.Timestamp, api_version: str = "2026-07") -> pd.DataFrame:
    """Orders from the Shopify Admin API (needs an Admin API access token with read_orders)."""
    domain = domain.strip().lower().removeprefix("https://").strip("/")
    if not SHOPIFY_DOMAIN.match(domain):
        raise ConnectorError("Enter your store's myshopify.com address, e.g. my-shop.myshopify.com.")
    if not re.match(r"^\d{4}-\d{2}$", api_version):
        raise ConnectorError("API version looks like 2026-07.")
    params = urllib.parse.urlencode({"status": "any", "limit": 250, "created_at_min": since.isoformat(),
                                     "fields": "created_at,cancelled_at,line_items"})
    url = f"https://{domain}/admin/api/{api_version}/orders.json?{params}"
    rows, pages = [], 0
    while url and pages < 400:
        _check_url(url, pattern=SHOPIFY_DOMAIN)
        data, headers = _get(url, {"X-Shopify-Access-Token": token})
        for order in json.loads(data).get("orders", []):
            if order.get("cancelled_at"):
                continue
            day = pd.Timestamp(order["created_at"]).tz_localize(None).normalize()
            for li in order.get("line_items", []):
                pid = li.get("sku") or str(li.get("product_id") or li.get("title"))
                rows.append({"date": day, "product_id": pid, "product_name": li.get("title") or pid,
                             "category": li.get("vendor") or "General", "units_sold": int(li.get("quantity") or 0),
                             "price": float(li.get("price") or 0)})
        nxt = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("Link", "") or headers.get("link", ""))
        url = nxt.group(1) if nxt else None
        pages += 1
    return _finish(rows)


def woocommerce(site: str, key: str, secret: str, since: pd.Timestamp) -> pd.DataFrame:
    """Orders from the WooCommerce REST API (read-only consumer key and secret)."""
    site = site.strip().rstrip("/")
    _check_url(site)
    host = urllib.parse.urlparse(site).hostname
    auth = base64.b64encode(f"{key}:{secret}".encode()).decode()
    rows, page = [], 1
    while page <= 400:
        params = urllib.parse.urlencode({"per_page": 100, "page": page, "after": since.isoformat(),
                                         "status": "completed,processing"})
        url = f"{site}/wp-json/wc/v3/orders?{params}"
        _check_url(url, hosts=(host,))
        data, _ = _get(url, {"Authorization": f"Basic {auth}"})
        orders = json.loads(data)
        if not orders:
            break
        for order in orders:
            day = pd.Timestamp(order["date_created"]).normalize()
            for li in order.get("line_items", []):
                pid = li.get("sku") or str(li.get("product_id"))
                qty = int(li.get("quantity") or 0)
                rows.append({"date": day, "product_id": pid, "product_name": li.get("name") or pid,
                             "category": "General", "units_sold": qty,
                             "price": float(li.get("total") or 0) / max(qty, 1)})
        page += 1
    return _finish(rows)


def amazon_report(content: bytes) -> pd.DataFrame:
    """An Amazon Seller Central 'All Orders' report (tab- or comma-separated)."""
    text = content[:4096].decode("utf-8", "replace")
    sep = "\t" if text.count("\t") > text.count(",") else ","
    try:
        raw = pd.read_csv(io.BytesIO(content), sep=sep, nrows=2_000_001, low_memory=False)
    except Exception:  # noqa: BLE001 - user file
        raise ConnectorError("Could not read the report.") from None
    cols = {c.lower().strip(): c for c in raw.columns}
    need = ["purchase-date", "sku", "quantity", "item-price"]
    if not all(n in cols for n in need):
        raise ConnectorError("Expected an 'All Orders' report with purchase-date, sku, quantity and item-price.")
    df = pd.DataFrame({
        "date": pd.to_datetime(raw[cols["purchase-date"]], errors="coerce", utc=True).dt.tz_localize(None).dt.normalize(),
        "product_id": raw[cols["sku"]].astype(str),
        "product_name": raw[cols["product-name"]].astype(str) if "product-name" in cols else raw[cols["sku"]].astype(str),
        "category": "Amazon", "units_sold": pd.to_numeric(raw[cols["quantity"]], errors="coerce").fillna(0),
        "price": pd.to_numeric(raw[cols["item-price"]], errors="coerce"),
    })
    if "order-status" in cols:
        df = df[~raw[cols["order-status"]].astype(str).str.contains("Cancel", case=False, na=False)]
    df = df.dropna(subset=["date"])
    df["price"] = df["price"] / df["units_sold"].where(df["units_sold"] > 0)
    return _finish(df.dropna(subset=["price"]).to_dict(orient="records"))


# ---------------------------------------------------------------- alerts
def alert_lines(plan: pd.DataFrame, stockout_days: int = 14, change_pct: float = 30.0) -> list[str]:
    """Plain-text alerts from a portfolio plan."""
    out = []
    for r in plan[plan["status"].astype(str) == "Order now"].itertuples():
        out.append(f"ORDER NOW · {r.product_name} ({r.warehouse}): {r.stock:,.0f} in stock, "
                   f"{r.days_of_cover} days of cover — order {r.order_qty:,.0f} units.")
    soon = plan[plan["stockout_date"].notna()]
    for r in soon.itertuples():
        days = (pd.Timestamp(r.stockout_date) - plan.attrs.get("today", pd.Timestamp(r.stockout_date))).days
        if 0 <= days <= stockout_days and str(r.status) != "Order now":
            out.append(f"STOCK-OUT SOON · {r.product_name}: runs out around {pd.Timestamp(r.stockout_date):%d %b}.")
    for r in plan.itertuples():
        if r.recent_daily > 0:
            ch = (r.forecast_daily / r.recent_daily - 1) * 100
            if abs(ch) >= change_pct:
                word = "SPIKE" if ch > 0 else "DROP"
                out.append(f"DEMAND {word} · {r.product_name}: forecast {r.forecast_daily:,.1f}/day vs "
                           f"{r.recent_daily:,.1f}/day recently ({ch:+.0f}%).")
    return out


def digest_text(plan: pd.DataFrame, horizon: int, currency: str = "$") -> str:
    lines = alert_lines(plan)
    total = plan["forecast_units"].sum()
    rev = plan["revenue"].sum()
    spend = plan.loc[plan["status"].astype(str).isin(["Order now", "Order this week"]), "order_cost"].sum()
    head = [f"Demandino weekly digest — next {horizon} days",
            f"Forecast: {total:,.0f} units · revenue {currency}{rev:,.0f}",
            f"To order now/this week: {int(plan['status'].astype(str).isin(['Order now', 'Order this week']).sum())} "
            f"lines · {currency}{spend:,.0f}", ""]
    return "\n".join(head + (lines or ["No alerts — stock and demand look healthy."]))


def send_slack(webhook: str, text: str) -> None:
    _check_url(webhook, hosts=("hooks.slack.com",))
    if not webhook.startswith("https://hooks.slack.com/services/"):
        raise ConnectorError("Use a Slack incoming-webhook URL (https://hooks.slack.com/services/…).")
    _post_json(webhook, {"text": text[:35_000]})


def send_email(host: str, port: int, user: str, password: str, sender: str, to: str, subject: str, body: str) -> None:
    for addr in (sender, *[a.strip() for a in to.split(",")]):
        if not EMAIL.match(addr):
            raise ConnectorError(f"'{addr[:60]}' is not a valid email address.")
    if not re.match(r"^[A-Za-z0-9.-]{3,190}$", host) or not (0 < int(port) < 65536):
        raise ConnectorError("Check the SMTP server and port.")
    msg = EmailMessage()
    msg["From"], msg["To"], msg["Subject"] = sender, to, subject[:200]
    msg.set_content(body[:100_000])
    ctx = ssl.create_default_context()
    try:
        if int(port) == 465:
            with smtplib.SMTP_SSL(host, int(port), context=ctx, timeout=TIMEOUT) as s:
                s.login(user, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, int(port), timeout=TIMEOUT) as s:
                s.starttls(context=ctx)  # never send credentials in clear text
                s.login(user, password)
                s.send_message(msg)
    except smtplib.SMTPAuthenticationError:
        raise ConnectorError("The email server rejected the username or password.") from None
    except (smtplib.SMTPException, OSError):
        raise ConnectorError("Could not send the email. Check the server, port and connection.") from None
