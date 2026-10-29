"""Terms & conditions text and the acceptance gate shown before first use."""
from __future__ import annotations

import streamlit as st

TERMS_VERSION = "1.0 · 29 October 2026"

TERMS_MD = """
**Please read these terms before using Demandino.** By ticking *I accept* you agree to them.

#### 1. What Demandino is
Demandino is a decision-support tool. It uses machine-learning models to *estimate* future product demand from
past sales and the details you provide, and suggests actions such as prices, promotions and reorder quantities.

#### 2. Forecasts are estimates, not guarantees
- Every forecast is uncertain. Ranges show likely outcomes, but real sales can and sometimes will fall outside them.
  The accuracy figures shown in the app describe past tests and do not guarantee future results.
- Results depend on the quality and completeness of the data and details you enter.
- Suggestions (pricing, promotions, orders, growth ideas) are informational only. They are **not** financial,
  investment, legal, tax or accounting advice. Check important decisions with a qualified professional and your
  own judgement.

#### 3. Your data
- Files you upload are processed **in memory on the server running this app** to produce your forecasts. They are
  not sold, shared or sent to third parties by Demandino.
- Uploaded data is **not written to disk** by Demandino. It is kept only for your session and is discarded when the
  server restarts or the cached results expire.
- Only upload data you have the right to use. Do not upload personal data about customers (names, emails,
  addresses, payment details). Demandino only needs dates, products, quantities and prices.
- If someone else hosts this app for you, their own privacy policy and security practices also apply.

#### 4. Acceptable use
You agree not to upload malicious files, try to break or overload the service, access other users' data, or use the
app for anything unlawful.

#### 5. Sample and third-party data
- The *Sample* dataset is simulated and for demonstration only.
- The *Real store* dataset is derived from **UCI Online Retail II** (Chen, D., 2012), licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is used here for testing and demonstration.

#### 6. No warranty and limitation of liability
Demandino is provided **"as is"**, without warranties of any kind, express or implied, including accuracy, fitness for
a particular purpose and non-infringement. To the fullest extent permitted by law, the authors are not liable for
any loss or damage (including lost profits, excess or insufficient stock, or business interruption) arising from
use of, or reliance on, the app or its results.

#### 7. Changes
These terms may be updated. If they change, you will be asked to accept the new version.

*Version {version}. This is a general template, not legal advice. Have it reviewed by a lawyer before offering
Demandino to customers.*
"""


def terms_markdown() -> str:
    return TERMS_MD.replace("{version}", TERMS_VERSION)


def accepted() -> bool:
    return st.session_state.get("terms_accepted") == TERMS_VERSION


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
        st.rerun()
    return False
