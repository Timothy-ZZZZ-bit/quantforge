"""Investable universe construction.

We avoid survivorship-biased universes wherever possible. The
``SP500History`` class supports a point-in-time membership snapshot from
locally cached membership history. Where membership history is unavailable,
we document the residual bias clearly.

References
----------
- Brown, Goetzmann, Ibbotson, Ross (1992), "Survivorship Bias in Performance
  Studies", *Review of Financial Studies* 5, 553-580.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import pandas as pd

# A liquid 20-ETF basket spanning equities, fixed income, commodities, and FX.
# Selected to maximize cross-asset diversity for cross-sectional and pairs work
# while remaining freely tradable on yfinance.
_DEFAULT_ETF_BASKET: Final[tuple[str, ...]] = (
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "MDY",  # US equity
    "EFA",
    "EEM",
    "EWJ",
    "FXI",  # Intl equity
    "TLT",
    "IEF",
    "SHY",
    "LQD",
    "HYG",
    "TIP",  # Fixed income
    "GLD",
    "SLV",
    "USO",
    "DBC",  # Commodities
    "UUP",  # USD
)


def default_etf_basket() -> list[str]:
    """Return the default cross-asset ETF basket used in baseline backtests."""
    return list(_DEFAULT_ETF_BASKET)


@dataclass(frozen=True)
class _Membership:
    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp


class SP500History:
    """Point-in-time S&P 500 membership.

    The class reads a CSV at ``data/universe/sp500_membership.csv`` (if
    present) with columns ``[ticker, start_date, end_date]``. When the file
    is absent, the class falls back to the current Wikipedia snapshot, which
    introduces survivorship bias. Tests should always provide a synthetic or
    test-controlled membership file.

    Parameters
    ----------
    membership_path : Path | None
        Path to a CSV with columns ``[ticker, start_date, end_date]``.
        ``end_date`` may be empty for currently-listed names.
    """

    def __init__(self, membership_path: Path | None = None) -> None:
        self._members: list[_Membership] = []
        self._loaded_from: str = "none"
        if membership_path is not None and membership_path.exists():
            self._load_from_csv(membership_path)
            self._loaded_from = str(membership_path)

    def _load_from_csv(self, path: Path) -> None:
        df = pd.read_csv(path)
        for row in df.itertuples(index=False):
            start = pd.Timestamp(str(row.start_date))
            end_raw = getattr(row, "end_date", None)
            end = (
                pd.Timestamp(str(end_raw))
                if pd.notna(end_raw) and end_raw
                else pd.Timestamp.max
            )
            self._members.append(
                _Membership(ticker=str(row.ticker), start=start, end=end)
            )

    def constituents_on(self, asof: str | date | pd.Timestamp) -> list[str]:
        """Return the list of tickers that were members on ``asof``.

        If no membership data is loaded, returns an empty list and callers
        should treat this as a hard error in production code paths.
        """
        ts = pd.Timestamp(asof)
        return sorted({m.ticker for m in self._members if m.start <= ts <= m.end})

    @property
    def has_data(self) -> bool:
        """True iff a real membership file has been loaded."""
        return bool(self._members)


__all__ = ["SP500History", "default_etf_basket"]
