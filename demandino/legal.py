"""Terms & conditions text and the acceptance gate shown before first use."""
from __future__ import annotations

import streamlit as st

TERMS_VERSION = "1.2 · 13 November 2026"
COOKIE = "dm_terms"
COOKIE_VALUE = "v1.2-2026-11-13"  # remembers acceptance in this browser (e.g. across a light/dark reload)

# For whoever runs Demandino: these terms are a general starting point, not legal advice. Have a lawyer review them
# before offering the app to customers, and add your own privacy policy if you host it for other people.
TERMS_MD = """
**Please read these terms before using Demandino.** By ticking *I accept* you agree to them.

#### 1. What Demandino does
Demandino estimates how many of each product you'll sell, from your past sales and the details you enter. It then
suggests reorder dates, order quantities, prices and promotions. You decide what to do with them.

#### 2. Forecasts are estimates
- Every forecast is uncertain. The range shows where sales will most likely land, but real sales will sometimes fall
  outside it. **Model accuracy** shows how close the forecasts were on past sales they hadn't seen, which isn't a
  promise about the future.
- Results are only as good as the data and details you give. Numbers you haven't entered yet, such as stock or unit
  cost, are filled in as examples and labelled that way, so replace them before acting on the advice.
- Suggestions are not financial, tax or legal advice. Check big decisions yourself or with a professional.

#### 3. Your data
- Files you upload are used only to make your forecasts. Without an account they stay in memory for your visit and
  aren't saved.
- With an account, data is saved only when you choose to save it: datasets, scenarios, product edits, stock tables,
  targets, tracked forecasts and purchase orders. You can delete any of it, or your whole account, on the Workspace
  page. Deleting your account needs your password and can't be undone.
- Passwords are stored as one-way hashes, and keys you save for connected services are encrypted and never shown
  again.
- Each account can store a limited amount, such as 10 datasets and 200 scenarios.
- Editing or removing a product changes only how Demandino shows it. Your original sales data stays as it was, and
  you can undo the changes.
- Only upload sales data you have the right to use. Demandino needs dates, products, quantities and prices, so leave
  out customer details such as names, emails, addresses and payment details.

#### 4. Cookies and what your browser remembers
Demandino uses three cookies: one to remember that you accepted these terms, one to keep you signed in for up to 14
days, and one to remember that you closed the getting-started guide. Your browser also remembers whether you chose
light or dark mode. Demandino uses no advertising or tracking cookies.

To protect accounts, the server briefly records the network address of failed sign-ins and new accounts, and keeps a
history of your sign-ins and changes, which you can see on the Security and Change history pages.

#### 5. Services you choose to connect
These only send or fetch data when you turn them on:
- **Store imports** (Google Sheets, Shopify, WooCommerce, Amazon reports) fetch your sales from those services.
- **Alerts** send summaries to the Slack channel or email addresses you enter.
- **Ask Demandino** sends a summary of the forecast you're viewing to Anthropic, which makes the AI model behind the
  assistant. That covers the product name, numbers, dates and findings, not your sales files. Answers can be wrong,
  so check them against the forecast.
- **Weather** sends a place and dates to Open-Meteo.
- **Share links** let anyone who has the link see that forecast summary for 7 days, or until you revoke it.

Each service's own terms apply.

#### 6. Fair use
Don't upload harmful files, try to get into other people's accounts or data, overload the app or use it for anything
unlawful. To keep the app working for everyone, there are limits on file size, new forecasts per hour and assistant
questions.

#### 7. Data included with Demandino
The *Sample* shop is made up, for trying the app. The *Real store* data comes from UCI Online Retail II (Chen, 2012),
licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

#### 8. No warranty
Demandino is provided **"as is"**, without any warranty. As far as the law allows, its makers aren't responsible
for losses from using it or relying on its results, such as lost sales, too much or too little stock, or lost profit.

#### 9. Changes to these terms
If these terms change, you'll be asked to accept the new version before continuing.

*Version {version}.*
"""


def terms_markdown() -> str:
    return TERMS_MD.replace("{version}", TERMS_VERSION)


def _cookie_ok() -> bool:
    try:
        return st.context.cookies.get(COOKIE) == COOKIE_VALUE
    except Exception:  # noqa: BLE001 - no browser context (tests)
        return False


def accepted() -> bool:
    if st.session_state.get("terms_accepted") == TERMS_VERSION:
        return True
    if _cookie_ok() and not st.session_state.get("terms_withdrawn"):
        st.session_state["terms_accepted"] = TERMS_VERSION
        return True
    return False


def sync_cookie() -> None:
    """Write or clear the acceptance cookie after the user accepts or withdraws."""
    action = st.session_state.pop("_terms_cookie", None)
    if action == "set":
        st.html(f"<script>document.cookie = '{COOKIE}={COOKIE_VALUE}; Max-Age=31536000; Path=/; SameSite=Strict';"
                "</script>", unsafe_allow_javascript=True)
    elif action == "clear":
        st.html(f"<script>document.cookie = '{COOKIE}=; Max-Age=0; Path=/; SameSite=Strict';</script>",
                unsafe_allow_javascript=True)


def gate() -> bool:
    """Show the terms with an accept button until they are accepted. Returns True once accepted."""
    if accepted():
        return True
    st.html('<div class="dm-hero"><div class="dm-eyebrow">Before you start</div><h1>Terms &amp; conditions</h1>'
            '<p>A short read: what Demandino does, how your data is handled, and the limits of any forecast.</p></div>')
    with st.container(border=True, key="card-terms", height=460):
        st.markdown(terms_markdown())
    agree = st.checkbox("I have read and accept the terms & conditions", key="terms_checkbox")
    if st.button("Continue to Demandino", type="primary", disabled=not agree, icon=":material/arrow_forward:"):
        st.session_state["terms_accepted"] = TERMS_VERSION
        st.session_state["terms_withdrawn"] = False
        st.session_state["_terms_cookie"] = "set"
        st.rerun()
    return False
