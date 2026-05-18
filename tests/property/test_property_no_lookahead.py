from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.backtest import BacktestEngine
from quantforge.data import synthetic_equity_panel
from quantforge.execution.costs import CostModel


def _eq_weight_fn(date, visible):
    tickers = visible["ticker"].unique().tolist()
    if not tickers:
        return {}
    n = len(tickers)
    return dict.fromkeys(tickers, 1.0 / n)


def test_no_lookahead_under_future_permutation():
    """Future bars should never change a decision made at time t."""
    panel = synthetic_equity_panel(n_tickers=5, n_days=200, seed=99)
    eng1 = BacktestEngine(panel.copy(), _eq_weight_fn, cost_model=CostModel())
    res1 = eng1.run()

    # Permute the future (last 30 bars).
    cutoff = panel["date"].max() - pd.Timedelta(days=30)
    past = panel[panel["date"] <= cutoff].copy()
    future = panel[panel["date"] > cutoff].copy()
    rng = np.random.default_rng(0)
    future["adj_close"] = rng.permutation(future["adj_close"].to_numpy())
    perturbed = pd.concat([past, future], ignore_index=True).sort_values(
        ["date", "ticker"]
    )
    eng2 = BacktestEngine(perturbed, _eq_weight_fn, cost_model=CostModel())
    res2 = eng2.run()

    # Up to (but not including) the cutoff, equity should match exactly.
    e1 = res1.equity.loc[res1.equity.index <= cutoff]
    e2 = res2.equity.loc[res2.equity.index <= cutoff]
    np.testing.assert_allclose(e1.to_numpy(), e2.to_numpy(), atol=1e-6, rtol=1e-6)
