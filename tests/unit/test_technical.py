from __future__ import annotations

import pandas as pd
import pytest

from quantforge.features.technical import momentum, realized_vol


def test_momentum_simple():
    prices = pd.Series([100.0, 110.0, 121.0, 133.1])
    mom = momentum(prices, lookback=2, skip=0)
    assert mom.iloc[-1] == pytest.approx(133.1 / 110.0 - 1.0)


def test_momentum_invalid_lookback():
    with pytest.raises(ValueError):
        momentum(pd.Series([1.0]), lookback=0)


def test_momentum_with_skip():
    prices = pd.Series([100.0, 110.0, 121.0, 133.1])
    mom = momentum(prices, lookback=1, skip=1)
    # skip=1 means measure mom over the bar 1..2 not 2..3
    assert mom.iloc[-1] == pytest.approx(121.0 / 110.0 - 1.0)


def test_realized_vol_close_to_close(small_panel):
    sub = small_panel[small_panel["ticker"] == "SYN000"].set_index("date")
    v = realized_vol(sub, window=21, method="close_to_close")
    assert v.notna().sum() > 0
    assert (v.dropna() > 0).all()


@pytest.mark.parametrize("method", ["parkinson", "garman_klass", "yang_zhang"])
def test_realized_vol_other_estimators(small_panel, method):
    sub = small_panel[small_panel["ticker"] == "SYN000"].set_index("date")
    v = realized_vol(sub, window=21, method=method)
    assert v.notna().sum() > 0
    assert (v.dropna() >= 0).all()


def test_realized_vol_unknown_method(small_panel):
    sub = small_panel[small_panel["ticker"] == "SYN000"].set_index("date")
    with pytest.raises(ValueError):
        realized_vol(sub, window=21, method="invalid")  # type: ignore[arg-type]
