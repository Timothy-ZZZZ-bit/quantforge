"""Pre-populate the Parquet cache.

Run via ``python scripts/download_data.py``. Skipped in CI; intended for
local research after fresh clones.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from quantforge.data import default_etf_basket, load_equity_panel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2005-01-03")
    parser.add_argument("--end", default="2024-12-31")
    args = parser.parse_args()
    tickers = default_etf_basket() + ["SPY"]
    panel = load_equity_panel(tickers, args.start, args.end)
    print(f"downloaded {len(panel)} rows for {len(set(panel['ticker']))} tickers")


if __name__ == "__main__":
    main()
