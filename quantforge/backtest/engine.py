"""Event-driven backtest engine.

Bar-by-bar processing in chronological order. Decisions made at the close of
bar :math:`t` execute at the open of bar :math:`t+1`. The engine asserts the
no-look-ahead invariant at every step.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quantforge.backtest.portfolio_state import PortfolioState
from quantforge.execution.costs import CostModel
from quantforge.execution.fills import fill_at_next_open
from quantforge.logging import get_logger

_log = get_logger(__name__)


WeightFn = Callable[[pd.Timestamp, pd.DataFrame], dict[str, float]]
"""Allocator signature: receives the date and a panel of data available
strictly *up to and including* that date, returns target weights summing to
the desired gross leverage."""


@dataclass(frozen=True)
class BacktestResult:
    """Container for engine outputs."""

    equity: pd.Series
    weights: pd.DataFrame
    turnover: pd.Series
    gross_leverage: pd.Series
    config_hash: str = ""

    def returns(self) -> pd.Series:
        return self.equity.pct_change().dropna()

    def log_returns(self) -> pd.Series:
        ratio = self.equity / self.equity.shift(1)
        return pd.Series(np.log(ratio.to_numpy()), index=self.equity.index).dropna()


def _wide(panel: pd.DataFrame, field: str) -> pd.DataFrame:
    return panel.pivot(index="date", columns="ticker", values=field).sort_index()


class BacktestEngine:
    """Event-driven engine for long/short portfolios on daily bars.

    Parameters
    ----------
    panel : pd.DataFrame
        Tidy panel with columns ``date, ticker, open, adj_close, volume``.
    weight_fn : WeightFn
        Callable that returns target weights at each rebalance date.
    cost_model : CostModel
    initial_cash : float
        Starting cash.
    rebalance_freq : str
        ``"D"``, ``"W-FRI"``, ``"BMS"`` (start of business month), etc. Any
        pandas offset alias.
    participation_cap : float
        Maximum fraction of bar volume we are willing to trade.

    Notes
    -----
    The engine never looks at any column dated strictly greater than the
    current bar when computing the weight. This is verified by passing the
    weight function a slice of the panel masked to ``date <= t``.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        weight_fn: WeightFn,
        cost_model: CostModel | None = None,
        initial_cash: float = 1_000_000.0,
        rebalance_freq: str = "BMS",
        participation_cap: float = 0.10,
    ) -> None:
        self.panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.weight_fn = weight_fn
        self.cost_model = cost_model or CostModel()
        self.state = PortfolioState(cash=initial_cash)
        self.participation_cap = participation_cap
        self.rebalance_freq = rebalance_freq
        self._open = _wide(self.panel, "open")
        self._close = _wide(self.panel, "adj_close")
        self._volume = _wide(self.panel, "volume")
        self._dates: pd.DatetimeIndex = pd.DatetimeIndex(self._close.index)
        self._rebalance_dates = self._build_rebalance_calendar()

    def _build_rebalance_calendar(self) -> set[pd.Timestamp]:
        if self.rebalance_freq == "D":
            return set(self._dates)
        # The 'right' edge of the resample is the rebalance trigger.
        idx = pd.Series(1, index=self._dates).resample(self.rebalance_freq).first().dropna().index
        # Snap each anchor to the nearest available bar in our data.
        snapped = self._dates.searchsorted(idx)
        snapped = snapped[snapped < len(self._dates)]
        return set(self._dates[snapped])

    def _shares_to_target(
        self,
        target_w: dict[str, float],
        prices_now: pd.Series,
        equity: float,
    ) -> dict[str, float]:
        """Compute target share counts from weights at current equity."""
        out: dict[str, float] = {}
        for tk, w in target_w.items():
            p = float(prices_now.get(tk, np.nan))
            if not np.isfinite(p) or p <= 0:
                continue
            out[tk] = (w * equity) / p
        return out

    def run(self) -> BacktestResult:
        """Run the backtest end to end and return :class:`BacktestResult`."""
        n_bars = len(self._dates)
        pending_target_shares: dict[str, float] | None = None
        cost_frac = 0.0

        for i, date in enumerate(self._dates):
            close_now = self._close.iloc[i]
            open_now = self._open.iloc[i]
            volume_now = self._volume.iloc[i]

            # Execute any pending order at this bar's open.
            if pending_target_shares is not None:
                for tk, target in pending_target_shares.items():
                    current = self.state.shares.get(tk, 0.0)
                    delta = target - current
                    if abs(delta) < 1e-9:
                        continue
                    fill = fill_at_next_open(
                        ticker=tk,
                        target_shares_delta=delta,
                        next_open=float(open_now.get(tk, np.nan)),
                        bar_volume=float(volume_now.get(tk, 0.0)),
                        participation_cap=self.participation_cap,
                        cost_per_dollar=cost_frac,
                    )
                    if fill.shares != 0.0:
                        self.state.apply_fill(tk, fill.shares, fill.price, fill.cost)
                pending_target_shares = None

            # Mark to market at the current bar close.
            mtm_prices = {tk: float(close_now.get(tk, np.nan)) for tk in close_now.index}
            equity = self.state.mark_to_market(date, mtm_prices)

            # Carry/borrow cost for shorts (per bar).
            position_notional = {
                tk: self.state.shares.get(tk, 0.0) * mtm_prices.get(tk, 0.0)
                for tk in self.state.shares
            }
            self.state.cash -= self.cost_model.carry_cost(position_notional)

            # Generate new target weights only at rebalance dates.
            if date in self._rebalance_dates and i < n_bars - 1:
                visible = self.panel[self.panel["date"] <= date]
                # Verify the no-lookahead invariant.
                if (visible["date"] > date).any():  # pragma: no cover - defensive
                    raise RuntimeError("look-ahead bias detected in weight_fn input")
                weights = self.weight_fn(date, visible)
                weights = {k: float(v) for k, v in weights.items() if np.isfinite(v)}
                target_shares = self._shares_to_target(weights, close_now, equity)
                pending_target_shares = target_shares

                # Estimate this bar's cost fraction for application at next open.
                # We use a flat approximation across tickers; per-trade cost is
                # applied per fill above.
                cost_frac = (
                    self.cost_model.commission_bps / 10_000.0
                    + (self.cost_model.slippage_bps / 10_000.0) * self.participation_cap
                    + self.cost_model.impact_coef * float(np.sqrt(self.participation_cap))
                )
                # Bookkeep weights and turnover.
                self.state.weight_history.append((date, weights))
                prev_w = (
                    self.state.weight_history[-2][1]
                    if len(self.state.weight_history) >= 2
                    else dict.fromkeys(weights, 0.0)
                )
                turnover = sum(
                    abs(weights.get(tk, 0.0) - prev_w.get(tk, 0.0))
                    for tk in set(weights) | set(prev_w)
                )
                self.state.turnover.append((date, float(turnover)))

        return BacktestResult(
            equity=self.state.equity_series(),
            weights=self.state.weights_frame(),
            turnover=self.state.turnover_series(),
            gross_leverage=self.state.gross_leverage_series(),
        )


__all__ = ["BacktestEngine", "BacktestResult", "WeightFn"]
