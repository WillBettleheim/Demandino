"""Honest out-of-sample benchmark on real (UCI Online Retail II) and simulated sales.

Usage: python scripts/benchmark.py [data/uci_online_retail_daily.csv]
The last 90 days are held out; training and range calibration only use earlier data.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from demandino.data import generate_sample_data, prepare_sales  # noqa: E402
from demandino.model import backtest  # noqa: E402


def report(name: str, sales: pd.DataFrame, country: str | None) -> dict:
    t = time.time()
    bt = backtest(sales, country=country)
    m = bt["metrics"]
    print(f"\n== {name}: {sales['product_id'].nunique()} products, test {bt['cutoff']:%Y-%m-%d} +90d "
          f"({time.time() - t:.0f}s)")
    for k, v in m.items():
        print(f"  {k:36s} {v * 100:6.1f}%")
    print(f"  calibration: {bt['calibration']}")
    return {"dataset": name, "products": int(sales["product_id"].nunique()),
            "test_start": f"{bt['cutoff'] + pd.Timedelta(days=1):%Y-%m-%d}", "test_end": f"{sales['date'].max():%Y-%m-%d}",
            **{k: float(v) for k, v in m.items()}}


if __name__ == "__main__":
    real = sys.argv[1] if len(sys.argv) > 1 else "data/uci_online_retail_daily.csv"
    datasets = {"REAL · UCI Online Retail II": (prepare_sales(pd.read_csv(real)), "GB"),
                "SIMULATED sample": (generate_sample_data(end_date="2026-10-28"), "US")}
    # Two separate test windows per dataset: the last 90 days, and the 90 days before that.
    results = []
    for name, (sales, country) in datasets.items():
        for shift in (0, 90):
            data = sales[sales["date"] <= sales["date"].max() - pd.Timedelta(days=shift)]
            results.append(report(name, data, country))
    out = Path(__file__).resolve().parent.parent / "data" / "benchmark_results.json"
    out.write_text(json.dumps({"generated": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"), "results": results},
                              indent=2))
    print(f"\nSaved {out}")
