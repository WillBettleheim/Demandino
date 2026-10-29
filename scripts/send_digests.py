"""Send each user's weekly digest to their saved Slack webhook and/or email (run from cron).

Usage: python scripts/send_digests.py
Uses each user's saved alert settings and their saved stock table for the dataset the alerts were set up on.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from demandino import db, integrations as ig  # noqa: E402
from demandino.data import generate_sample_data, prepare_sales  # noqa: E402
from demandino.model import DemandForecaster  # noqa: E402
from demandino.planning import default_stock, portfolio_plan  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def dataset_for(uid: int, key: str) -> tuple[pd.DataFrame, str | None] | None:
    parts = key.split("-")
    country = parts[-1].upper() if parts[-1] != "none" else None
    if key.startswith("ws-"):
        sales = db.load_dataset(uid, int(parts[2]))
    elif key.startswith("real-"):
        sales = prepare_sales(pd.read_csv(ROOT / "data" / "uci_online_retail_daily.csv"))
    elif key.startswith("sample-"):
        sales = generate_sample_data()
    else:
        return None  # one-off uploads are never stored, so they can't be re-run later
    return (sales, country) if sales is not None else None


def main() -> None:
    with db.connect() as con:
        users = [dict(r) for r in con.execute("SELECT id, username FROM users").fetchall()]
    for u in users:
        cfg = db.get_setting(u["id"], "alerts")
        if not cfg or not cfg.get("dataset_key"):
            continue
        found = dataset_for(u["id"], cfg["dataset_key"])
        if found is None:
            print(f"{u['username']}: dataset not available, skipped")
            continue
        sales, country = found
        fc = DemandForecaster(sales, country=country).fit()
        stock = db.get_setting(u["id"], f"stock:{cfg['dataset_key']}")
        stock = pd.DataFrame(stock) if stock else default_stock(sales)
        plan, _ = portfolio_plan(fc, fc.level_uncertainty, stock, int(cfg.get("horizon", 30)), 14, 0.95)
        text = ig.digest_text(plan, int(cfg.get("horizon", 30)), db.get_setting(u["id"], "currency", "$"))
        sent = []
        hook = db.get_secret(u["id"], "slack_webhook")
        if hook:
            try:
                ig.send_slack(hook, text)
                sent.append("slack")
            except ig.ConnectorError as exc:
                print(f"{u['username']}: slack failed: {exc}")
        pw = db.get_secret(u["id"], "smtp_password")
        if pw and cfg.get("smtp_host") and cfg.get("to"):
            try:
                ig.send_email(cfg["smtp_host"], cfg.get("smtp_port", 587), cfg.get("smtp_user", ""), pw,
                              cfg.get("sender", ""), cfg["to"], "Demandino weekly digest", text)
                sent.append("email")
            except ig.ConnectorError as exc:
                print(f"{u['username']}: email failed: {exc}")
        print(f"{u['username']}: sent via {', '.join(sent) or 'nothing (no channel configured)'}")


if __name__ == "__main__":
    main()
