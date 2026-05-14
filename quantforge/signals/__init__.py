"""Signal strategies producing alpha scores or target weights.

Each strategy implements :class:`base.Signal` and exposes a uniform
``fit(panel) / predict(panel) -> pd.Series`` API. The returned series is
indexed by ``[date, ticker]`` and contains either a continuous alpha score
or a target weight, as documented in each strategy's docstring.
"""

from __future__ import annotations

from quantforge.signals.base import Signal, SignalDiagnostics
from quantforge.signals.mean_reversion import OUMeanReversion
from quantforge.signals.ml_signal import MLSignal
from quantforge.signals.pairs import PairsTrade
from quantforge.signals.quality import QualityFactor
from quantforge.signals.tsmom import TimeSeriesMomentum
from quantforge.signals.xsmom import CrossSectionalMomentum

__all__ = [
    "CrossSectionalMomentum",
    "MLSignal",
    "OUMeanReversion",
    "PairsTrade",
    "QualityFactor",
    "Signal",
    "SignalDiagnostics",
    "TimeSeriesMomentum",
]
