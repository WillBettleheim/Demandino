"""Bought together: which products end up in the same order more often than chance.

Input is one row per product per order (order_id, product_id). For every pair of products:
  together    orders that contain both
  confidence  share of A's orders that also contain B (and the other way round)
  lift        how many times more often they're bought together than if customers picked them
              independently (1 = no link, 2 = twice as often)
High-lift pairs with enough orders are candidates for bundles, "goes well with" suggestions and joint
promotions. It also warns when a product you might discount or drop is often bought with a best-seller.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
from scipy import sparse

from .security import UploadError, clean_text, read_csv_upload

MIN_TOGETHER = 30   # pairs bought together fewer times than this are too thin to act on
MAX_PRODUCTS = 400  # keeps the pair matrix small


def matrix(baskets: pd.DataFrame) -> tuple[sparse.csr_matrix, list[str], int]:
    """Orders × products as 0/1 (only orders and products present), the product ids, and the order count."""
    b = baskets[["order_id", "product_id"]].astype(str).drop_duplicates()
    top = b["product_id"].value_counts().head(MAX_PRODUCTS).index
    b = b[b["product_id"].isin(top)]
    orders = pd.Index(b["order_id"].unique())
    products = sorted(b["product_id"].unique())
    # Sparse: most orders hold few products, so a very large file stays small in memory.
    m = sparse.csr_matrix((np.ones(len(b), dtype=np.float32),
                           (orders.get_indexer(b["order_id"]), pd.Index(products).get_indexer(b["product_id"]))),
                          shape=(len(orders), len(products)))
    return m, products, len(orders)


def pairs(baskets: pd.DataFrame, names: dict[str, str] | None = None, min_together: int = MIN_TOGETHER
          ) -> pd.DataFrame:
    """Every product pair bought together at least `min_together` times, strongest links first."""
    m, products, n = matrix(baskets)
    if n == 0 or len(products) < 2:
        return pd.DataFrame(columns=["a", "b", "together", "share_of_orders", "a_to_b", "b_to_a", "lift"])
    co = (m.T @ m).toarray()
    count = np.diag(co).copy()
    i, j = np.triu_indices(len(products), k=1)
    together = co[i, j]
    keep = together >= min_together
    i, j, together = i[keep], j[keep], together[keep]
    names = names or {}
    out = pd.DataFrame({
        "a": [products[k] for k in i], "b": [products[k] for k in j],
        "together": together.astype(int), "share_of_orders": together / n,
        "a_to_b": together / count[i], "b_to_a": together / count[j],
        "lift": together * n / (count[i] * count[j]),
    })
    out.insert(1, "a_name", out["a"].map(lambda p: names.get(p, p)))
    out.insert(3, "b_name", out["b"].map(lambda p: names.get(p, p)))
    return out.sort_values(["lift", "together"], ascending=False).reset_index(drop=True)


def partners(pair_table: pd.DataFrame, product: str, top: int = 8) -> pd.DataFrame:
    """Products most often bought alongside one product: 'customers who bought this also bought'."""
    a = pair_table[pair_table["a"] == product].rename(columns={"b": "other", "b_name": "other_name", "a_to_b": "chance"})
    b = pair_table[pair_table["b"] == product].rename(columns={"a": "other", "a_name": "other_name", "b_to_a": "chance"})
    both = pd.concat([a, b])[["other", "other_name", "together", "chance", "lift"]]
    return both.sort_values(["chance", "lift"], ascending=False).head(top).reset_index(drop=True)


def ideas(pair_table: pd.DataFrame, limit: int = 4) -> list[str]:
    """Plain-language suggestions from the strongest pairs."""
    out, used = [], set()
    strong = pair_table[(pair_table["lift"] >= 2) & (pair_table[["a_to_b", "b_to_a"]].max(axis=1) >= 0.25)]
    for r in strong.itertuples():
        if len(out) >= limit:
            break
        if r.a in used or r.b in used:  # one idea per product, so the list covers different products
            continue
        used |= {r.a, r.b}
        lead, follow, conf = ((r.a_name, r.b_name, r.a_to_b) if r.a_to_b >= r.b_to_a else (r.b_name, r.a_name, r.b_to_a))
        out.append(f"**Bundle or suggest:** {conf:.0%} of orders with {lead} also include {follow} "
                   f"({r.lift:.1f}× more than chance, {r.together:,} orders). Offer them as a set or show "
                   f"{follow} next to {lead}.")
    if not out and len(pair_table):
        out.append("No strong pairs yet: products are mostly bought on their own. Bundles are unlikely to help much.")
    return out


def read_order_lines(content: bytes) -> pd.DataFrame:
    """Parse an uploaded order-lines CSV: needs an order column and a product column (common names accepted)."""
    raw = read_csv_upload(content)
    cols = {str(c).strip().lower().replace(" ", "_"): c for c in raw.columns}
    order = next((cols[c] for c in ("order_id", "order", "invoice", "invoice_no", "invoiceno", "order_number",
                                    "transaction_id", "basket_id") if c in cols), None)
    product = next((cols[c] for c in ("product_id", "sku", "stockcode", "stock_code", "item_id", "product",
                                      "product_name") if c in cols), None)
    if order is None or product is None:
        raise UploadError("Need an order column (e.g. order_id or invoice) and a product column (e.g. product_id "
                          "or sku), one row per product per order.")
    out = pd.DataFrame({"order_id": raw[order].astype(str).map(clean_text),
                        "product_id": raw[product].astype(str).map(clean_text)})
    out = out[(out["order_id"] != "") & (out["product_id"] != "")].drop_duplicates()
    if out["order_id"].nunique() < 50:
        raise UploadError("Need at least 50 orders to find products bought together.")
    return out


def template() -> str:
    buf = io.StringIO()
    pd.DataFrame({"order_id": ["1001", "1001", "1002"], "product_id": ["P001", "P011", "P001"]}).to_csv(buf, index=False)
    return buf.getvalue()


def drop_risks(pair_table: pd.DataFrame, stage: dict[str, str], min_chance: float = 0.25, limit: int = 5
               ) -> list[str]:
    """Declining or end-of-life products that customers often buy with products still selling well."""
    fading = {p for p, st in stage.items() if st in ("Declining", "End of life")}
    rows = []
    for r in pair_table[pair_table["lift"] >= 1.5].itertuples():
        for weak, weak_name, strong, strong_name, chance in ((r.a, r.a_name, r.b, r.b_name, r.b_to_a),
                                                              (r.b, r.b_name, r.a, r.a_name, r.a_to_b)):
            if weak in fading and strong not in fading and chance >= min_chance:
                rows.append((weak, weak_name, strong_name, chance))
    if not rows:
        return []
    df = pd.DataFrame(rows, columns=["weak", "weak_name", "strong_name", "chance"]).sort_values("chance", ascending=False)
    out = []
    for weak, g in list(df.groupby("weak", sort=False))[:limit]:
        in_ = ", ".join(f"{r.chance:.0%} of {r.strong_name} orders" for r in g.head(3).itertuples())
        out.append(f"**Careful before dropping:** {g['weak_name'].iloc[0]} ({stage[weak].lower()}) is in {in_}. "
                   "Offer a replacement so those orders don't shrink.")
    return out
