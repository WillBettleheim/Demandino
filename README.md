# Demandino: ML demand forecasting

Demandino predicts how many units a product will sell, day by day. It learns sales patterns from your history
(trend, weekday and yearly seasonality, holidays, price and promotion effects) and gives a detailed analysis
of each forecast.

## Run it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

The app opens with a realistic **sample dataset** (30 products, 6 categories, 3 years of daily sales). To use
your own data, choose **Upload my CSV** in the sidebar. You can download a template from the same place.

| column | required | notes |
|---|---|---|
| `date` | ✓ | one row per product per day (missing days count as 0 sales) |
| `product_id` | ✓ | or `product_name` / `sku` |
| `units_sold` | ✓ | also accepts `quantity`, `units`, `qty`, `sales` |
| `price` | ✓ | actual selling price that day |
| `product_name`, `category` | | |
| `regular_price` | | inferred from recent price ceiling if missing |
| `promo` | | 0/1; inferred from price < regular price if missing |

## Pages

- **Forecast a product.** Inputs sit in a panel on the left and results update live:
  - **Product:** an existing product, or a new one estimated from its category and price, a similar product, or
    a short sales history you upload. New launches can have a ramp-up period.
  - **Plans:** price, cost, horizon, promotions (recurring, specific dates, Black Friday) and demand events such
    as campaigns or closures.
  - **Inventory:** stock on hand, lead time, days between orders and service level.

  Results include expected units with an 80% range, revenue, profit, a confidence label and key findings. Tabs
  cover the forecast (with last year overlaid), a drivers waterfall, weekly and monthly breakdowns, price and
  promotion what-ifs, an inventory plan, sales patterns, saved scenarios compared side by side, and an Excel or
  CSV export.
- **Demand planning.** Forecasts every product at once from an editable table of stock and lead times. Each
  product gets a status, a run-out date, an order-by date and an order quantity, and you can download a
  purchase list.
- **Grow demand.** Tests the levers you control by re-running the model with one change at a time: regular price,
  promotion depth and frequency, the best week to promote, and staying in stock. It ranks the options by profit
  and finds the most profitable tested plan that reaches a sales goal. General good-practice ideas the model
  can't measure (listings, reviews, bundles, channels) are listed separately.
- **Explore sales patterns.** Trend, seasonality, weekday × month heatmap, year-over-year, promo lift, price
  elasticity, unusual-day detection and product comparison.
- **Portfolio overview.** Revenue by category, category growth, ABC × XYZ segmentation and momentum.
- **Model accuracy.** A back-test on the last 90 days compared with simple baselines, plus error by weeks ahead
  and by category, and feature importance.

**Light and dark mode.** Use the sun/moon switch in the sidebar. The "Lagoon" theme (teal and coral) has a
deep-sea dark mode and an airy light mode, and every chart and card follows the active mode.

The first run trains the model (about 10 seconds) and saves it in `.cache/`, so later runs start instantly.

- **Store & category totals.** Product forecasts rolled up into categories and the whole store, with ranges that
  account for shared uncertainty. The totals always add up.
- **Ask Demandino.** An AI assistant (Claude Opus 5.5) that answers plain-language questions about the forecast on
  screen. It needs an Anthropic API key: set `ANTHROPIC_API_KEY` on the server, or save a key in your workspace.
  Only a summary of the forecast is sent, never your raw files.
- **Workspace** (sign in required).
  - Accounts: local accounts with scrypt-hashed passwords, a lockout after repeated failures, and sessions that
    survive reloads.
  - Saved data: datasets, scenarios (open any saved scenario back on the Forecast page) and share links (read-only,
    7 days, revocable).
  - Account settings: change password, or delete your account with all its data.
- **Connections & alerts.**
  - Imports: Google Sheets, the Shopify Admin API, the WooCommerce REST API and Amazon "All Orders" reports.
  - Alerts: order-now, stock-out-soon and demand-spike/drop alerts, plus a weekly digest, sent to Slack or email.
  - Automatic sending: schedule `scripts/send_digests.py` with cron.
- **Purchase orders.** A purchase list sent from Demand planning becomes a draft that you approve, reject or mark
  as ordered.
- **Forecast tracking.** Track a forecast and compare it with the real sales as they arrive.

**More forecasting features.**
- **Model inputs:**
  - **Country holidays:** public holidays for the country you pick in the sidebar.
  - **Stock-out correction:** days that look sold out are filled with estimated demand, and you can add an
    `on_hand` column to mark them exactly.
  - **Outside factors:** upload a CSV of daily values such as ad spend or temperature.
- **Day explainer:** "Why this day?" on the forecast's Drivers tab.
- **Demand types:** each product is classed as smooth, erratic, intermittent or lumpy.
- **Cross-product effects:** the Grow demand page counts promotions net of sales taken from sibling products.
- **Ordering rules:** minimum order quantities (MOQ), case packs, unit costs, several warehouses, a purchase budget
  and a weekly cash-flow chart.
- **Display:** currency conversion with your own exchange rate, PDF reports and a phone-friendly layout.

## Terms & data handling

The app asks you to accept terms & conditions before first use (see `demandino/legal.py`, and the Terms page
in the app). They are a general template, not legal advice; have them reviewed before offering the app to
customers. Uploaded files are processed in memory and never written to disk. See [SECURITY.md](SECURITY.md).

## Accuracy on real data

The model is tested on real sales from the [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)
dataset: a UK online gift wholesaler, Dec 2009 – Dec 2011, using its 40 most regularly sold products (CC BY 4.0).
Pick **Real store** in the sidebar to use it in the app.

A "miss" is a day (or a product-week) where actual sales landed outside the predicted range. Each test window
is 90 days the model never saw during training or calibration. Days the whole shop was closed are excluded.

| Data | Unseen test window | Daily misses | Weekly misses | Daily WAPE (simple average) |
|---|---|---|---|---|
| Real · UCI | 10 Sep – 9 Dec 2011 | 3.6% | 2.3% | 84% (90%) |
| Real · UCI | 12 Jun – 10 Sep 2011 | 4.8% | 2.5% | 92% (101%) |
| Simulated | 30 Jul – 28 Oct 2026 | 3.9% | 1.7% | 36% (46%) |
| Simulated | 1 May – 30 Jul 2026 | 5.6% | 1.4% | 38% (45%) |

Point error (WAPE) is a separate question from range reliability. A single product's daily sales are noisy,
and on the wholesaler's lumpy orders the point error stays high. The ranges are honest about that rather than
precise.

To reproduce:

```bash
.venv/bin/python -I scripts/prepare_uci.py data/raw/online_retail_ii/online_retail_II.xlsx data/uci_online_retail_daily.csv 40
.venv/bin/python scripts/benchmark.py
```

The first command needs the dataset zip from the UCI link above, unzipped into `data/raw/online_retail_ii/`.

## How the model works (`demandino/model.py`)

- **One global model for all products.** A scikit-learn `HistGradientBoostingRegressor` is trained on every
  product at once. Each training row asks: "standing at day *o* with this history, what sells on day *o + gap*?"
  This is a direct multi-horizon forecast with no recursive error build-up.
- **Features:**
  - the recent demand level and momentum
  - demand in the same season last year, for both the product and its category
  - discount depth, price compared with the recent average and with category peers, and the promo flag
  - calendar features: weekday, annual cycle, Black Friday weekend, days to Christmas
- **Scale-free target.** The model predicts units ÷ recent level, using a Poisson loss weighted by level. This
  lets a product selling 3 units a day and one selling 300 share patterns.
- **Uncertainty.** Quantile models (p5 and p95) give the daily range. The range is then conformally calibrated:
  a model trained without the latest 180 days forecasts them from several start dates, and the range is widened
  or narrowed until 95% of real outcomes fall inside. Multi-day totals include a level error calibrated the same
  way.
- **Price changes.** Promotional discounts are learned by the ML model. Permanent regular-price changes use each
  product's own price elasticity (a log-log regression controlling for promo, weekday, month and trend), shrunk
  toward its category. Tree models can't extrapolate prices reliably.
- **New products.** A cold-start model (category and price positioning to demand level), a similar product
  rescaled for price, or your expected daily sales. A launch ramp-up curve is optional.

## Tests

```bash
.venv/bin/pip install pytest
.venv/bin/python -m pytest -q tests
```
