# Methodology: validation and overfitting

The single largest risk in retail quantitative research is implicit
multiple testing: iterating on backtest configurations until the curve
looks good and publishing that result without recording how many other
configurations were tried. The validation protocol below exists to
neutralize that risk. Every strategy in QuantForge must clear all nine
items before it can be merged into `main` as a validated result.

## 1. No look-ahead

A property test permutes the future portion of the panel and verifies that
no decision made before the cutoff changes. The backtest engine enforces
the invariant in its inner loop. See
[`tests/property/test_property_no_lookahead.py`](../../tests/property/test_property_no_lookahead.py).

## 2. Purged k-fold cross-validation

For ML signals, training-time CV uses purged k-fold with embargo equal to
the maximum label horizon. Fold-level performance is reported separately;
the aggregate is *not* the headline statistic.

## 3. Walk-forward analysis

Both rolling and expanding-window walk-forward are produced for every
strategy. The tearsheet is generated on out-of-sample segments only.

## 4. Combinatorial purged cross-validation

CPCV with at least ten groups and two test groups, yielding
$2 \cdot \binom{10}{2} / 10 = 9$ paths per asset; aggregating across the
combination set produces $\binom{10}{2} = 45$ OOS path estimates. We
report the 5th percentile, median, and 95th percentile Sharpe across the
distribution.

## 5. Deflated Sharpe Ratio

Bailey and Lopez de Prado (2014). Given a sample Sharpe $\hat{SR}$, the
*Probabilistic* Sharpe Ratio is

$$
\text{PSR}(SR^*) = \Phi\!\left(
    \frac{(\hat{SR} - SR^*) \sqrt{n-1}}
         {\sqrt{1 - \gamma_3 \hat{SR} + \frac{\gamma_4 - 1}{4} \hat{SR}^2}}
\right),
$$

where $\gamma_3$ is sample skew and $\gamma_4$ is sample excess kurtosis.
The *Deflated* Sharpe Ratio sets the benchmark $SR^*$ equal to the expected
maximum of $N$ independent standard-normal Sharpe estimates, which scales
roughly as $\sqrt{2 \log N}$. Therefore the more strategies one has tried,
the higher the bar that any single strategy must clear.

We do not estimate `n_trials` by hand. The CLI appends every backtest run
to `research_log.jsonl` with its config hash, OOS Sharpe, and timestamp.
The DSR computation reads from that log; the number is enforced, not
chosen.

## 6. Transaction cost sensitivity

Every strategy is run at four cost multipliers: 0x, 1x, 2x, 5x. The
tearsheet reports the multiplier at which Sharpe halves and the
multiplier at which Sharpe becomes negative.

## 7. Regime robustness

Sharpe split by VIX regime (low, mid, high terciles) and by yield-curve
regime (steepening, flattening, inverted). A strategy that lives entirely
in one regime is suspect.

## 8. Bootstrap CI on Sharpe

Stationary bootstrap (Politis-Romano 1994) with mean block length matched
to the autocorrelation horizon of the returns; default 21 bars (one
trading month). We report a 95% confidence interval.

## 9. Multiple-testing audit

The on-disk research log is the source of truth. Any strategy variant
that has been run is recorded; any deletion of the log to inflate DSR is
a research-integrity violation.

## A strategy that fails any one of these does not get merged

Failures are documented in `docs/failed_strategies/` with the diagnosis.
This is a feature: a quant researcher who never published failed
strategies has a publication bias problem.

## References

- Bailey, D.H., Lopez de Prado, M. (2012). The Sharpe Ratio efficient frontier. *Journal of Risk* 15, 3-44.
- Bailey, D.H., Lopez de Prado, M. (2014). The Deflated Sharpe Ratio. *Journal of Portfolio Management* 40, 94-107.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chs. 7 and 12.
- Politis, D.N., Romano, J.P. (1994). The stationary bootstrap. *Journal of the American Statistical Association* 89, 1303-1313.
