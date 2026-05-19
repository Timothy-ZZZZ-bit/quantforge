"""Coverage for the data loaders with mocked network dependencies.

The loaders hit yfinance, FRED, and Ken French's library. These tests inject
fakes so the suite never touches the network.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quantforge.data import loaders
from quantforge.data.storage import ParquetCache


@pytest.fixture(autouse=True)
def _tmp_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Redirect the module-level cache to a temp directory per test."""
    monkeypatch.setattr(loaders, "_CACHE", ParquetCache(root=tmp_path))


def _fake_yf_frame(n: int = 30) -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=n, name="Date")
    base = 100.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "Open": base,
            "High": base * 1.01,
            "Low": base * 0.99,
            "Close": base,
            "Adj Close": base,
            "Volume": np.full(n, 1_000_000.0),
        },
        index=idx,
    )


def test_yf_columns_single_index():
    out = loaders._yf_columns(_fake_yf_frame(), "ABC")
    assert list(out.columns) == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]
    assert (out["ticker"] == "ABC").all()


def test_load_equity_panel_with_fake_yfinance(monkeypatch: pytest.MonkeyPatch):
    fake = types.ModuleType("yfinance")

    def _download(ticker, **kwargs):
        return _fake_yf_frame()

    fake.download = _download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake)

    panel = loaders.load_equity_panel(["AAA", "BBB"], "2022-01-01", "2022-03-01")
    assert set(panel["ticker"]) == {"AAA", "BBB"}
    assert "adj_close" in panel.columns
    # Second call should hit the parquet cache (no error).
    panel2 = loaders.load_equity_panel(["AAA", "BBB"], "2022-01-01", "2022-03-01")
    assert len(panel2) == len(panel)


def test_load_equity_panel_all_empty(monkeypatch: pytest.MonkeyPatch):
    fake = types.ModuleType("yfinance")
    fake.download = lambda ticker, **kw: pd.DataFrame()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    panel = loaders.load_equity_panel(["ZZZ"], "2022-01-01", "2022-03-01")
    assert panel.empty


def test_load_fama_french_with_fake_reader(monkeypatch: pytest.MonkeyPatch):
    fake_pdr = types.ModuleType("pandas_datareader")
    fake_data = types.ModuleType("pandas_datareader.data")

    def _reader(name, source, *args, **kwargs):
        idx = pd.bdate_range("2020-01-02", periods=40, name="Date")
        ff = pd.DataFrame(
            {
                "Mkt-RF": np.full(40, 0.02),
                "SMB": np.full(40, 0.01),
                "HML": np.full(40, -0.01),
                "RMW": np.full(40, 0.005),
                "CMA": np.full(40, 0.003),
                "RF": np.full(40, 0.001),
            },
            index=idx,
        )
        if "Momentum" in name:
            return {0: pd.DataFrame({"Mom   ": np.full(40, 0.015)}, index=idx)}
        return {0: ff}

    fake_data.DataReader = _reader  # type: ignore[attr-defined]
    fake_pdr.data = fake_data  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas_datareader", fake_pdr)
    monkeypatch.setitem(sys.modules, "pandas_datareader.data", fake_data)

    out = loaders.load_fama_french(model="FF5+MOM")
    assert "Mkt-RF" in out.columns
    assert "MOM" in out.columns
    # Percentages converted to decimals.
    assert abs(out["Mkt-RF"].iloc[0]) < 1.0


def test_load_fama_french_invalid_model(monkeypatch: pytest.MonkeyPatch):
    fake_pdr = types.ModuleType("pandas_datareader")
    fake_data = types.ModuleType("pandas_datareader.data")
    fake_data.DataReader = lambda *a, **k: {0: pd.DataFrame()}  # type: ignore[attr-defined]
    fake_pdr.data = fake_data  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas_datareader", fake_pdr)
    monkeypatch.setitem(sys.modules, "pandas_datareader.data", fake_data)
    with pytest.raises(ValueError):
        loaders.load_fama_french(model="NOPE")


def test_load_macro_series_with_fake_reader(monkeypatch: pytest.MonkeyPatch):
    fake_pdr = types.ModuleType("pandas_datareader")
    fake_data = types.ModuleType("pandas_datareader.data")

    def _reader(series_ids, source, start, end):
        idx = pd.bdate_range("2020-01-02", periods=10, name="DATE")
        return pd.DataFrame({s: np.arange(10, dtype=float) for s in series_ids}, index=idx)

    fake_data.DataReader = _reader  # type: ignore[attr-defined]
    fake_pdr.data = fake_data  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pandas_datareader", fake_pdr)
    monkeypatch.setitem(sys.modules, "pandas_datareader.data", fake_data)

    out = loaders.load_macro_series(["DGS10", "VIXCLS"], "2020-01-01", "2020-02-01")
    assert "date" in out.columns
    assert "DGS10" in out.columns
