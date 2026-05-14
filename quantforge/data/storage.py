"""Local Parquet cache used to memoize expensive data loads.

The cache is keyed by a stable hash of the loader's arguments. Loaders should
call ``cache.get_or_compute(key, fn)`` rather than calling the network every
time. Cached files live under ``data/cache/`` by default.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from quantforge.constants import DATA_CACHE_DIR
from quantforge.logging import get_logger

_log = get_logger(__name__)


def _stable_key(payload: dict[str, object]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


class ParquetCache:
    """Tiny Parquet-backed memoization layer.

    Parameters
    ----------
    root : Path | None
        Directory in which to store cached files. Created if it does not exist.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root: Path = Path(root) if root is not None else DATA_CACHE_DIR
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, namespace: str, payload: dict[str, object]) -> Path:
        """Return the Parquet path for a namespace/payload combination."""
        return self.root / f"{namespace}-{_stable_key(payload)}.parquet"

    def get_or_compute(
        self,
        namespace: str,
        payload: dict[str, object],
        compute: Callable[[], pd.DataFrame],
        force: bool = False,
    ) -> pd.DataFrame:
        """Return cached DataFrame or compute, persist, and return.

        Parameters
        ----------
        namespace : str
            Logical grouping, e.g. ``"equity_panel"`` or ``"fama_french"``.
        payload : dict
            Arguments that uniquely determine the result.
        compute : callable
            Zero-argument callable invoked on cache miss.
        force : bool
            If True, ignore an existing cache entry and recompute.
        """
        path = self.path_for(namespace, payload)
        if path.exists() and not force:
            _log.debug("cache.hit", namespace=namespace, path=str(path))
            return pd.read_parquet(path)
        _log.info("cache.compute", namespace=namespace, path=str(path))
        df = compute()
        try:
            df.to_parquet(path, index=False)
        except (ValueError, TypeError) as exc:
            # Some DataFrames are not Parquet-friendly; fall through gracefully.
            _log.warning("cache.write_failed", path=str(path), error=str(exc))
        return df


__all__ = ["ParquetCache"]
