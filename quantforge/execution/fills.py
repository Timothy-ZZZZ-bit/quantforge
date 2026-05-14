"""Fill-price assumptions used by the backtest engine.

QuantForge assumes orders generated on bar ``t`` execute at the open of bar
``t+1``. This is the most defensible assumption for a daily backtest because
it avoids both look-ahead and the artificially favorable practice of filling
at the same bar's close.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fill:
    """A single executed trade for a single asset on a single bar."""

    ticker: str
    shares: float  # signed
    price: float
    notional: float
    participation: float
    cost: float


def fill_at_next_open(
    ticker: str,
    target_shares_delta: float,
    next_open: float,
    bar_volume: float,
    participation_cap: float,
    cost_per_dollar: float,
) -> Fill:
    """Apply the participation cap and compute the executed Fill.

    Parameters
    ----------
    ticker : str
    target_shares_delta : float
        Signed change in share count desired by the allocator.
    next_open : float
        The open price on the bar following the decision bar.
    bar_volume : float
        Total bar volume on the executing bar.
    participation_cap : float
        Maximum fraction of bar volume we will trade.
    cost_per_dollar : float
        Combined commission + slippage + impact cost expressed as a fraction
        of traded notional.
    """
    if next_open <= 0 or bar_volume <= 0:
        return Fill(ticker, 0.0, next_open, 0.0, 0.0, 0.0)
    max_shares = participation_cap * bar_volume
    capped_shares = max(-max_shares, min(target_shares_delta, max_shares))
    notional = capped_shares * next_open
    participation = abs(capped_shares) / bar_volume
    cost = abs(notional) * cost_per_dollar
    return Fill(
        ticker=ticker,
        shares=capped_shares,
        price=next_open,
        notional=notional,
        participation=participation,
        cost=cost,
    )


__all__ = ["Fill", "fill_at_next_open"]
