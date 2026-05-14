# ADR-005: Deflated Sharpe Ratio as the headline statistic

## Status

Accepted.

## Context

Raw Sharpe ratio is the standard headline in retail quant tearsheets. It
ignores skew, kurtosis, and implicit multiple testing. A repository whose
headline is "Sharpe = 2.5" without further qualification will be
correctly dismissed by any sophisticated reader.

## Decision

The tearsheet displays both raw Sharpe and Deflated Sharpe Ratio (DSR);
the DSR is the *headline* and the raw Sharpe is reported alongside for
context. The DSR `n_trials` argument is read from the on-disk research
log, not chosen by the author.

## Rationale

Bailey and Lopez de Prado (2014) showed that the expected maximum Sharpe
across `N` independent zero-skill strategies scales as $\sqrt{2 \log N}$.
A backtest from a search over many configurations therefore has an
implicit benchmark Sharpe equal to that expected maximum, not zero. DSR
makes this adjustment explicit.

## Consequences

Strategies that look impressive under raw Sharpe but fail DSR are
documented as failures rather than shipped as findings. This is the
desired behavior; the validation protocol is doing its job.
