"""Portfolio state object: positions, cash, equity, accruals."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class PortfolioState:
    """Mutable portfolio state advanced one bar at a time by the engine.

    The state is intentionally minimal: a cash balance, a per-ticker share
    book, and a running record of equity. All higher-level metrics
    (Sharpe, drawdown, attribution) are computed from the equity series in
    post-processing.

    Attributes
    ----------
    cash : float
        Settled cash balance.
    shares : dict[str, float]
        Signed shares held per ticker.
    equity_curve : list[tuple[pd.Timestamp, float]]
        Pairs of (date, total equity) for every processed bar.
    """

    cash: float = 1_000_000.0
    shares: dict[str, float] = field(default_factory=dict)
    equity_curve: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    turnover: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    gross_leverage: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    weight_history: list[tuple[pd.Timestamp, dict[str, float]]] = field(default_factory=list)

    def mark_to_market(self, date: pd.Timestamp, prices: dict[str, float]) -> float:
        """Mark to market: equity = cash + sum(shares * price)."""
        pos_value = sum(self.shares.get(t, 0.0) * prices.get(t, np.nan) for t in self.shares)
        equity = self.cash + (pos_value if np.isfinite(pos_value) else 0.0)
        self.equity_curve.append((date, equity))
        gross = sum(abs(self.shares.get(t, 0.0)) * prices.get(t, 0.0) for t in self.shares)
        self.gross_leverage.append((date, gross / equity if equity > 0 else 0.0))
        return equity

    def apply_fill(self, ticker: str, shares_delta: float, price: float, cost: float) -> None:
        """Apply a single trade fill to the state."""
        notional = shares_delta * price
        self.cash -= notional
        self.cash -= cost
        self.shares[ticker] = self.shares.get(ticker, 0.0) + shares_delta

    def equity_series(self) -> pd.Series:
        """Equity curve as a pandas Series indexed by date."""
        if not self.equity_curve:
            return pd.Series(dtype=float)
        dates, values = zip(*self.equity_curve, strict=False)
        return pd.Series(values, index=pd.DatetimeIndex(dates), name="equity")

    def turnover_series(self) -> pd.Series:
        if not self.turnover:
            return pd.Series(dtype=float)
        dates, values = zip(*self.turnover, strict=False)
        return pd.Series(values, index=pd.DatetimeIndex(dates), name="turnover")

    def gross_leverage_series(self) -> pd.Series:
        if not self.gross_leverage:
            return pd.Series(dtype=float)
        dates, values = zip(*self.gross_leverage, strict=False)
        return pd.Series(values, index=pd.DatetimeIndex(dates), name="gross_leverage")

    def weights_frame(self) -> pd.DataFrame:
        if not self.weight_history:
            return pd.DataFrame()
        rows: list[dict[str, object]] = []
        for d, w in self.weight_history:
            row: dict[str, object] = {"date": d}
            row.update(w)
            rows.append(row)
        return pd.DataFrame(rows).set_index("date").fillna(0.0)


__all__ = ["PortfolioState"]
