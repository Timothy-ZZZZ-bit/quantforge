from __future__ import annotations

import pandas as pd
import pytest

from quantforge.features.cross_sectional import cs_rank, cs_zscore, sector_neutralize


def _panel():
    dates = pd.to_datetime(["2024-01-02"] * 4 + ["2024-01-03"] * 4)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": ["A", "B", "C", "D"] * 2,
            "value": [1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0],
            "sector": ["X", "X", "Y", "Y"] * 2,
        }
    )


def test_cs_rank_bounds():
    p = _panel()
    r = cs_rank(p, "value")
    assert r.min() >= -0.5
    assert r.max() <= 0.5


def test_cs_rank_monotone():
    p = _panel()
    p_first = p[p["date"] == "2024-01-02"]
    r = cs_rank(p_first, "value")
    # Ranks should monotonically increase with the value column.
    assert (r.diff().dropna() > 0).all()


def test_cs_zscore_mean_zero():
    p = _panel()
    z = cs_zscore(p, "value")
    for _, g in p.groupby("date"):
        g_z = z.loc[g.index]
        assert g_z.mean() == pytest.approx(0.0, abs=1e-10)


def test_sector_neutralize_sums_to_zero_within_sector():
    p = _panel()
    s = sector_neutralize(p, "value", "sector")
    by = p.groupby(["date", "sector"]).indices
    for (_, _), idx in by.items():
        assert s.loc[idx].sum() == pytest.approx(0.0, abs=1e-10)


def test_sector_neutralize_missing_sector_col():
    p = _panel().drop(columns=["sector"])
    with pytest.raises(KeyError):
        sector_neutralize(p, "value", "sector")
