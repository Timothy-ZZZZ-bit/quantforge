"""Data ingestion, universe construction, caching, and quality auditing."""

from __future__ import annotations

from quantforge.data.loaders import (
    load_equity_panel,
    load_fama_french,
    load_macro_series,
    synthetic_equity_panel,
)
from quantforge.data.quality import QualityReport, audit_panel
from quantforge.data.storage import ParquetCache
from quantforge.data.universe import SP500History, default_etf_basket

__all__ = [
    "ParquetCache",
    "QualityReport",
    "SP500History",
    "audit_panel",
    "default_etf_basket",
    "load_equity_panel",
    "load_fama_french",
    "load_macro_series",
    "synthetic_equity_panel",
]
