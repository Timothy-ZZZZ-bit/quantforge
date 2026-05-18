from __future__ import annotations

from pathlib import Path

import pandas as pd

from quantforge.data.storage import ParquetCache
from quantforge.data.universe import SP500History, default_etf_basket


def test_parquet_cache_roundtrip(tmp_path: Path):
    cache = ParquetCache(root=tmp_path)
    calls = {"n": 0}

    def compute() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"a": [1, 2, 3]})

    df1 = cache.get_or_compute("ns", {"k": 1}, compute)
    df2 = cache.get_or_compute("ns", {"k": 1}, compute)
    assert calls["n"] == 1  # second call is a cache hit
    pd.testing.assert_frame_equal(df1, df2)


def test_parquet_cache_force_recompute(tmp_path: Path):
    cache = ParquetCache(root=tmp_path)
    calls = {"n": 0}

    def compute() -> pd.DataFrame:
        calls["n"] += 1
        return pd.DataFrame({"a": [1]})

    cache.get_or_compute("ns", {"k": 1}, compute)
    cache.get_or_compute("ns", {"k": 1}, compute, force=True)
    assert calls["n"] == 2


def test_parquet_cache_distinct_payloads(tmp_path: Path):
    cache = ParquetCache(root=tmp_path)
    p1 = cache.path_for("ns", {"k": 1})
    p2 = cache.path_for("ns", {"k": 2})
    assert p1 != p2


def test_default_etf_basket_non_empty():
    basket = default_etf_basket()
    assert len(basket) >= 15
    assert "SPY" in basket


def test_sp500_history_empty_without_data():
    hist = SP500History()
    assert not hist.has_data
    assert hist.constituents_on("2020-01-02") == []


def test_sp500_history_from_csv(tmp_path: Path):
    csv = tmp_path / "membership.csv"
    csv.write_text(
        "ticker,start_date,end_date\n" "AAA,2010-01-01,2015-01-01\n" "BBB,2012-01-01,\n"
    )
    hist = SP500History(membership_path=csv)
    assert hist.has_data
    assert hist.constituents_on("2013-06-01") == ["AAA", "BBB"]
    assert hist.constituents_on("2016-01-01") == ["BBB"]
    assert hist.constituents_on("2009-01-01") == []
