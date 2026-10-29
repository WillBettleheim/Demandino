"""Convert UCI Online Retail II transactions into Demandino's daily sales format.

Usage: python -I scripts/prepare_uci.py <online_retail_II.xlsx> <out.csv> [n_products]

Real data from a UK online gift retailer (Dec 2009 – Dec 2011):
https://archive.ics.uci.edu/dataset/502/online+retail+ii
Cancellations, returns, postage/fees and zero-price lines are removed; the N products
sold on the most days are kept. Categories are assigned from description keywords.
"""
import re
import sys

import pandas as pd

CATEGORIES = [
    ("Christmas", r"CHRISTMAS|XMAS|ADVENT|SANTA|REINDEER|SNOWFLAKE|BAUBLE"),
    ("Bags & Storage", r"\bBAG\b|BAGS|STORAGE|BOX|TIN|JAR|BASKET|HOLDER"),
    ("Lighting & Candles", r"LIGHT|LANTERN|CANDLE|T-LIGHT|TEALIGHT|LAMP"),
    ("Kitchen & Dining", r"MUG|CUP|PLATE|BOWL|CAKE|TEA|NAPKIN|BAKING|LUNCH|KITCHEN|SPOON|BOTTLE|TRAY"),
    ("Stationery & Party", r"CARD|PAPER|WRAP|PARTY|BUNTING|PEN|NOTEBOOK|STICKER|GIFT|CHALK"),
    ("Home Decor", r"HEART|SIGN|FRAME|DECORATION|HANGING|CLOCK|MIRROR|CUSHION|DOORMAT|WALL|BIRD|GARLAND"),
]
NON_PRODUCT = {"POST", "D", "M", "C2", "DOT", "BANK CHARGES", "AMAZONFEE", "CRUK", "PADS", "S", "B", "ADJUST",
               "ADJUST2", "TEST001", "TEST002", "SP1002", "GIFT"}


def category(desc: str) -> str:
    d = str(desc).upper()
    for name, pat in CATEGORIES:
        if re.search(pat, d):
            return name
    return "Gifts & Other"


def main(src: str, out: str, n_products: int = 40) -> None:
    sheets = pd.read_excel(src, sheet_name=None, dtype={"StockCode": str, "Invoice": str})
    df = pd.concat(sheets.values(), ignore_index=True)
    df = df[~df["Invoice"].str.startswith(("C", "A"))]
    df["StockCode"] = df["StockCode"].str.upper().str.strip()
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0) & ~df["StockCode"].isin(NON_PRODUCT)]
    df = df[df["StockCode"].str.match(r"^\d{5}")]
    df["date"] = pd.to_datetime(df["InvoiceDate"]).dt.normalize()

    # Use the catalogue (single-unit) price, not the realised average: wholesale buyers get volume
    # discounts, so the average price falls on big-order days and would leak the answer to the model.
    daily = df.groupby(["StockCode", "date"]).agg(units_sold=("Quantity", "sum"), list_price=("Price", "max")
                                                  ).reset_index()
    daily["price"] = daily["list_price"].round(2)

    active = daily.groupby("StockCode")["date"].nunique().sort_values(ascending=False)
    keep = active.head(n_products).index
    names = (df[df["StockCode"].isin(keep)].groupby("StockCode")["Description"]
             .agg(lambda s: s.dropna().astype(str).str.strip().mode().iloc[0]).str.title())
    daily = daily[daily["StockCode"].isin(keep)].copy()
    daily["product_name"] = daily["StockCode"].map(names)
    daily["category"] = daily["product_name"].map(category)
    daily = daily.sort_values(["StockCode", "date"])
    daily["regular_price"] = daily.groupby("StockCode")["list_price"].transform(
        lambda s: s.rolling(20, min_periods=1).median())
    daily["promo"] = 0  # the shop's data has no promotion flag
    daily = daily.rename(columns={"StockCode": "product_id"})
    cols = ["date", "product_id", "product_name", "category", "price", "regular_price", "promo", "units_sold"]
    daily[cols].to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"{len(keep)} products, {daily['date'].min():%Y-%m-%d} → {daily['date'].max():%Y-%m-%d}, {len(daily)} rows")
    print(daily.groupby("category")["product_id"].nunique())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 40)
