"""Data ingestion from free sources: yfinance, FRED, Ken French's library.

The synthetic generator at the bottom of this module produces a panel with a
known factor structure; it is used in unit and integration tests so the test
suite never depends on a network connection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from quantforge.constants import TRADING_DAYS_PER_YEAR
from quantforge.data.storage import ParquetCache
from quantforge.logging import get_logger

if TYPE_CHECKING:
    pass

_log = get_logger(__name__)
_CACHE = ParquetCache()


def _yf_columns(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize yfinance multi-index columns to lowercase singles."""
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance returns (field, ticker) when multiple tickers, single level otherwise.
        df = pd.DataFrame(df.xs(ticker, axis=1, level=1, drop_level=True))
    rename = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    }
    out = df.rename(columns=rename)
    keep = [c for c in ["open", "high", "low", "close", "adj_close", "volume"] if c in out.columns]
    out = out[keep].reset_index().rename(columns={"Date": "date", "index": "date"})
    out["ticker"] = ticker
    return out[["date", "ticker", *keep]]


def load_equity_panel(
    tickers: list[str],
    start: str,
    end: str,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download adjusted OHLCV for a list of tickers.

    Returns a tidy long-format DataFrame with columns
    ``[date, ticker, open, high, low, close, adj_close, volume]``.
    Splits and dividends are reflected in ``adj_close``; ``close`` is the raw
    close. Returns should be computed off ``adj_close``.

    Notes
    -----
    The function caches under ``data/cache/equity_panel-<hash>.parquet``.
    """
    payload = {"tickers": sorted(tickers), "start": start, "end": end, "v": 1}

    def _compute() -> pd.DataFrame:
        import yfinance as yf  # local import: optional dep when only using synthetic

        frames: list[pd.DataFrame] = []
        for tk in sorted(tickers):
            _log.info("yf.download", ticker=tk, start=start, end=end)
            df = yf.download(
                tk,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            if df.empty:
                _log.warning("yf.empty", ticker=tk)
                continue
            frames.append(_yf_columns(df, tk))
        if not frames:
            return pd.DataFrame(
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
        out = pd.concat(frames, ignore_index=True)
        out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None).dt.normalize()
        return out.sort_values(["ticker", "date"]).reset_index(drop=True)

    return _CACHE.get_or_compute("equity_panel", payload, _compute, force=force_refresh)


def load_fama_french(
    model: str = "FF5+MOM",
    frequency: str = "daily",
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download Fama-French factor returns from Ken French's library.

    Parameters
    ----------
    model : str
        ``"FF3"``, ``"FF5"`` or ``"FF5+MOM"`` (default).
    frequency : str
        ``"daily"`` (default) or ``"monthly"``.
    """
    payload = {"model": model, "frequency": frequency, "v": 1}

    def _compute() -> pd.DataFrame:
        from pandas_datareader import data as pdr  # local import

        suffix = "_daily" if frequency == "daily" else ""
        if model == "FF3":
            ds = pdr.DataReader(f"F-F_Research_Data_Factors{suffix}", "famafrench")
            ff = ds[0]
        elif model in ("FF5", "FF5+MOM"):
            ds5 = pdr.DataReader(f"F-F_Research_Data_5_Factors_2x3{suffix}", "famafrench")
            ff = ds5[0]
            if model == "FF5+MOM":
                mom = pdr.DataReader(f"F-F_Momentum_Factor{suffix}", "famafrench")[0]
                mom.columns = ["MOM"]
                ff = ff.join(mom, how="inner")
        else:
            raise ValueError(f"unsupported model: {model!r}")
        # Ken French publishes in percentages; convert to decimal.
        out = (ff / 100.0).reset_index().rename(columns={"Date": "date"})
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        return out

    return _CACHE.get_or_compute("fama_french", payload, _compute, force=force_refresh)


def load_macro_series(
    series_ids: list[str],
    start: str,
    end: str,
    *,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download macro series from FRED via pandas_datareader."""
    payload = {"series_ids": sorted(series_ids), "start": start, "end": end, "v": 1}

    def _compute() -> pd.DataFrame:
        from pandas_datareader import data as pdr

        df = pdr.DataReader(series_ids, "fred", start, end)
        return df.reset_index().rename(columns={"DATE": "date"})

    return _CACHE.get_or_compute("macro", payload, _compute, force=force_refresh)


def synthetic_equity_panel(
    n_tickers: int = 10,
    n_days: int = 252 * 5,
    start: str = "2018-01-02",
    *,
    seed: int = 1729,
    mu_annual: float = 0.08,
    sigma_annual: float = 0.20,
    factor_loading_dispersion: float = 0.30,
    factor_sigma_annual: float = 0.15,
) -> pd.DataFrame:
    """Generate a deterministic synthetic equity panel with a single common factor.

    Used by tests and notebooks where a network call is undesirable. Returns the
    same tidy schema as :func:`load_equity_panel`.

    Mathematical Definition
    -----------------------
    For ticker :math:`i` and day :math:`t`,

    .. math::
        r_{i,t} = \\beta_i f_t + \\epsilon_{i,t}, \\quad
        f_t \\sim \\mathcal{N}(\\mu_f, \\sigma_f^2), \\quad
        \\epsilon_{i,t} \\sim \\mathcal{N}(0, \\sigma_i^2).

    Prices are obtained by exponential cumulative sum of returns.

    References
    ----------
    Standard single-factor model. See Bodie/Kane/Marcus Ch. 7.
    """
    rng = np.random.default_rng(seed)

    dt = 1.0 / TRADING_DAYS_PER_YEAR
    mu_d = mu_annual * dt
    sigma_d = sigma_annual * np.sqrt(dt)
    factor_sigma_d = factor_sigma_annual * np.sqrt(dt)

    betas = 1.0 + rng.normal(0.0, factor_loading_dispersion, size=n_tickers)
    factor = rng.normal(mu_d, factor_sigma_d, size=n_days)
    idio = rng.normal(0.0, sigma_d, size=(n_days, n_tickers))
    returns = factor[:, None] * betas[None, :] + idio

    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    dates = pd.bdate_range(start=start, periods=n_days)

    tickers = [f"SYN{i:03d}" for i in range(n_tickers)]
    out_frames: list[pd.DataFrame] = []
    for j, tk in enumerate(tickers):
        close = prices[:, j]
        # Build synthetic OHLCV around close with realistic intraday spread.
        intraday_sd = sigma_d * close
        high = close + np.abs(rng.normal(0, intraday_sd))
        low = close - np.abs(rng.normal(0, intraday_sd))
        open_ = close * (1.0 + rng.normal(0, sigma_d / 2.0, size=n_days))
        volume = rng.integers(low=500_000, high=2_500_000, size=n_days).astype(float)
        frame = pd.DataFrame(
            {
                "date": dates,
                "ticker": tk,
                "open": open_,
                "high": np.maximum.reduce([high, open_, close]),
                "low": np.minimum.reduce([low, open_, close]),
                "close": close,
                "adj_close": close,
                "volume": volume,
            }
        )
        out_frames.append(frame)
    return (
        pd.concat(out_frames, ignore_index=True)
        .sort_values(["ticker", "date"])
        .reset_index(drop=True)
    )


__all__ = [
    "load_equity_panel",
    "load_fama_french",
    "load_macro_series",
    "synthetic_equity_panel",
]
