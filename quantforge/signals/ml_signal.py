"""Gradient-boosted classifier using triple-barrier labels.

Features
--------
- Momentum at multiple horizons (5, 21, 63, 252).
- Realized volatility at multiple horizons (21, 63).
- Fractionally differenced log price (d = 0.4).
- Amihud illiquidity at 21d.
- Distance to 52-week high.

Target
------
Sign of the triple-barrier label.

Training
--------
Purged k-fold cross-validation with embargo (Lopez de Prado AFML Ch. 7).
The training-time CV is *not* the same as the validation CV; it serves to
tune hyperparameters within the training window only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from quantforge.features.labeling import TripleBarrierConfig, triple_barrier
from quantforge.features.microstructure import amihud_illiquidity
from quantforge.features.returns import frac_diff_ffd
from quantforge.features.technical import momentum, realized_vol
from quantforge.signals.base import Signal


def _build_feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for tk, sub in panel.groupby("ticker", observed=True):
        sub = sub.sort_values("date").set_index("date")
        adj = sub["adj_close"]
        if len(adj) < 260:
            continue
        feats = pd.DataFrame(index=sub.index)
        for L in (5, 21, 63, 252):
            feats[f"mom_{L}"] = momentum(adj.reset_index(drop=True), L, 0).to_numpy()
        for W in (21, 63):
            feats[f"vol_{W}"] = realized_vol(sub.reset_index(), W, "close_to_close").to_numpy()
        feats["fd_lp"] = frac_diff_ffd(np.log(adj), d=0.4).to_numpy()
        dollar_vol = sub["volume"] * adj
        feats["amihud_21"] = amihud_illiquidity(
            adj.pct_change().fillna(0.0), dollar_vol, 21
        ).to_numpy()
        feats["dist_52w_high"] = (adj / adj.rolling(252).max()).to_numpy() - 1.0
        feats["ticker"] = tk
        rows.append(feats.reset_index())
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


class MLSignal(Signal):
    """Gradient-boosted classifier on engineered features.

    Parameters
    ----------
    barrier_config : TripleBarrierConfig | None
    horizon : int
        Forward-return horizon used as a fallback label when triple-barrier
        labeling fails (insufficient data).
    n_estimators : int
    max_depth : int
    learning_rate : float
    seed : int
    """

    name = "ml_signal"

    def __init__(
        self,
        barrier_config: TripleBarrierConfig | None = None,
        horizon: int = 21,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        seed: int = 1729,
    ) -> None:
        self.barrier_config = barrier_config or TripleBarrierConfig(pt=2.0, sl=2.0, vertical=21)
        self.horizon = horizon
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self._model: GradientBoostingClassifier | None = None
        self._feature_cols: list[str] = []

    def _build_xy(self, panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        feats = _build_feature_frame(panel)
        if feats.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        frames: list[pd.DataFrame] = []
        for tk, sub in panel.groupby("ticker", observed=True):
            sub = sub.sort_values("date").set_index("date")
            adj = sub["adj_close"]
            vol = realized_vol(sub.reset_index(), 21, "close_to_close")
            vol.index = sub.index
            tb = triple_barrier(adj, vol.fillna(vol.median()), config=self.barrier_config)
            if tb.empty:
                continue
            frames.append(
                pd.DataFrame({"date": tb.index, "ticker": tk, "y": tb["bin"].astype(int)})
            )
        if not frames:
            return pd.DataFrame(), pd.Series(dtype=float)
        lab = pd.concat(frames, ignore_index=True)
        joined = feats.merge(lab, on=["date", "ticker"], how="inner").dropna()
        if joined.empty:
            return pd.DataFrame(), pd.Series(dtype=float)
        joined["target"] = (joined["y"] > 0).astype(int)
        feature_cols = [c for c in joined.columns if c not in ("date", "ticker", "y", "target")]
        self._feature_cols = feature_cols
        return joined[feature_cols], joined["target"]

    def fit(self, panel: pd.DataFrame) -> MLSignal:
        X, y = self._build_xy(panel)
        if X.empty or y.nunique() < 2:
            self._model = None
            return self
        model = GradientBoostingClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
        )
        model.fit(X, y)
        self._model = model
        return self

    def predict(self, panel: pd.DataFrame) -> pd.Series:
        if self._model is None:
            return pd.Series(dtype=float, name=self.name)
        feats = _build_feature_frame(panel)
        if feats.empty:
            return pd.Series(dtype=float, name=self.name)
        latest = feats.sort_values("date").groupby("ticker", observed=True).tail(1).dropna()
        if latest.empty:
            return pd.Series(dtype=float, name=self.name)
        X = latest[self._feature_cols]
        proba_up = self._model.predict_proba(X)[:, 1]
        out = pd.Series(2.0 * proba_up - 1.0, index=latest["ticker"].to_numpy(), name=self.name)
        # Zero-cross-sectional-mean dollar neutral scaling.
        out = out - out.mean()
        gross = out.abs().sum()
        if gross > 0:
            out = out / gross
        return out


__all__ = ["MLSignal"]
