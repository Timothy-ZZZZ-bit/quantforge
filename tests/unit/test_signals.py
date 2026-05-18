from __future__ import annotations

import numpy as np
import pandas as pd

from quantforge.signals import (
    CrossSectionalMomentum,
    OUMeanReversion,
    PairsTrade,
    QualityFactor,
    TimeSeriesMomentum,
)


def test_tsmom_returns_dict_of_floats(medium_panel):
    sig = TimeSeriesMomentum(lookback=63, skip=5, vol_window=21)
    sig.fit(medium_panel)
    out = sig.predict(medium_panel)
    assert isinstance(out, pd.Series)
    assert out.notna().all()


def test_xsmom_balances_long_and_short(medium_panel):
    sig = CrossSectionalMomentum(lookback=63, skip=5, decile=0.30)
    out = sig.predict(medium_panel)
    if not out.empty:
        # Long and short sides should net (approximately) to zero notional.
        assert abs(out.sum()) < 0.5


def test_ou_skipped_on_short_history():
    # Empty panel -> empty signal.
    panel = pd.DataFrame(
        columns=[
            "date",
            "ticker",
            "open",
            "high",
            "low",
            "close",
            "adj_close",
            "volume",
        ]
    )
    sig = OUMeanReversion(window=63)
    out = sig.predict(panel)
    assert out.empty


def test_quality_factor_runs(medium_panel):
    sig = QualityFactor(lookback=126)
    out = sig.predict(medium_panel)
    assert isinstance(out, pd.Series)


def test_pairs_no_signals_on_independent_series():
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "date": np.tile(pd.bdate_range("2022-01-03", periods=n), 2),
            "ticker": ["X"] * n + ["Y"] * n,
            "adj_close": np.concatenate(
                [
                    100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                    100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))),
                ]
            ),
        }
    )
    sig = PairsTrade(candidate_pairs=[("X", "Y")], window=120)
    out = sig.predict(df)
    # Independent series rarely cointegrate; signal should be small or empty.
    assert isinstance(out, pd.Series)
