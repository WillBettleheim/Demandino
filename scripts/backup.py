"""Back up the workspace database safely, even while the app is running.

Usage: python scripts/backup.py [folder] [--keep N]
Keeps the newest N copies (default 14) in a folder only this account can read (default .data/backups). Backups hold
accounts and sales data, so keep any copy you move off the server encrypted.
"""
import argparse
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demandino.db import DATA_DIR, DB_PATH  # noqa: E402
from demandino.security import private_dir  # noqa: E402


def backup(folder: Path, keep: int = 14, source: Path = DB_PATH) -> Path:
    if not source.exists():
        raise FileNotFoundError(f"No database at {source}.")
    private_dir(folder)
    target = folder / f"demandino-{time.strftime('%Y%m%d-%H%M%S')}.db"
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)  # private from the moment it exists
    os.close(fd)
    src, dst = sqlite3.connect(source), sqlite3.connect(target)
    try:
        src.backup(dst)
        if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("The backup failed its integrity check.")
    finally:
        src.close()
        dst.close()
    for old in sorted(folder.glob("demandino-*.db"), reverse=True)[max(keep, 1):]:
        old.unlink()
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", nargs="?", default=str(DATA_DIR / "backups"))
    parser.add_argument("--keep", type=int, default=14)
    args = parser.parse_args()
    print(f"Saved {backup(Path(args.folder), args.keep)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
