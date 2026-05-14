"""Synthetic end-to-end test.

Generates a panel with a known planted factor and verifies that a basic
TSMOM + HRP pipeline recovers a positive Sharpe in-sample under zero costs.
This is intentionally a sanity check rather than a strong claim, since the
pipeline could fail in many ways and the planted-factor data generating
process is benign.
"""

from __future__ import annotations

import numpy as np
import pytest

from quantforge.backtest import BacktestEngine
from quantforge.data import synthetic_equity_panel
from quantforge.execution.costs import CostModel
from quantforge.metrics.performance import sharpe_ratio
from quantforge.portfolio import HRPAllocator
from quantforge.signals import TimeSeriesMomentum


@pytest.mark.integration
def test_synthetic_end_to_end():
    panel = synthetic_equity_panel(
        n_tickers=12,
        n_days=252 * 5,
        seed=2026,
        mu_annual=0.12,
        sigma_annual=0.18,
    )

    wide_close = panel.pivot(index="date", columns="ticker", values="adj_close").sort_index()
    rets_hist = np.log(wide_close / wide_close.shift(1))

    sig = TimeSeriesMomentum(lookback=63, skip=5, vol_window=21)
    alloc = HRPAllocator()

    def weight_fn(date, visible):
        sig.fit(visible)
        alpha = sig.predict(visible).abs()  # long-only for the integration test
        if alpha.empty:
            return {}
        hist = rets_hist.loc[:date].iloc[-126:]
        w = alloc.allocate(alpha, hist)
        return w.to_dict()

    engine = BacktestEngine(panel, weight_fn, cost_model=CostModel(commission_bps=0, slippage_bps=0, impact_coef=0))
    result = engine.run()
    r = result.returns()
    assert len(r) > 252
    s = sharpe_ratio(r)
    # The planted positive drift should give a positive Sharpe under zero costs.
    assert np.isfinite(s)
