# Methodology: backtesting

## Engine

The QuantForge backtest engine is *event-driven* and processes one bar at
a time in chronological order. At each bar `t`:

1. Mark to market at the close. Compute equity.
2. Apply per-bar carry costs (borrow fees on shorts).
3. If `t` is a rebalance date and not the last bar, call the allocator
   with the panel masked to `date <= t`. Record target weights.
4. Convert target weights to target shares using the current close.
5. The pending order set is executed at the *open of bar* `t+1`, with the
   participation cap and transaction-cost model applied.

The invariant the engine asserts is that the weight function never
receives data dated after `t`. The corresponding property test in
`tests/property/test_property_no_lookahead.py` permutes future bars and
verifies that decisions made before the cutoff are unaffected.

## Transaction costs

Per-trade cost is decomposed into three components, all in basis points of
traded notional:

- **Commission:** flat `bps`. Default 1 bp.
- **Linear slippage:** `participation * bps`. Default `5 bps` at full participation.
- **Almgren-Chriss impact:** $\eta \cdot \sqrt{\text{participation}}$ with $\eta = 0.10$. See `constants.DEFAULT_IMPACT_COEF` for the source.

Carry cost on shorts is 50 bps per annum on the absolute short notional,
applied per bar.

## Rebalance frequency

Daily, weekly (Fridays), and monthly (business-month start) are
supported via the pandas offset alias. Higher rebalance frequencies
produce more decisions and therefore raise the multiple-testing bar
implicitly; we record the chosen frequency on every run.

## Walk-forward

Two modes. Rolling: the training window slides forward by `step_years`,
maintaining a fixed width. Expanding: the training window grows by
`step_years`, starting at the earliest available bar. Each yields a
sequence of disjoint test segments; the tearsheet is generated on the
concatenation of test segments only.

## Combinatorial purged cross-validation

Lopez de Prado AFML Ch. 12. Partition observations into `N` contiguous
groups, choose `k` to be the test set, run train-test on each
$\binom{N}{k}$ combination, and aggregate. Purging removes any training
observations whose labels overlap with test observations; embargo extends
the purge by a fixed number of bars on either side of every test
fragment. The number of out-of-sample paths produced is

$$
\frac{k \cdot \binom{N}{k}}{N}.
$$

CPCV produces a *distribution* of OOS Sharpe ratios, not a single point
estimate; the 5th, median, and 95th percentiles are reported on the
tearsheet alongside the histogram.

## References

- Almgren, R., Chriss, N. (2000). Optimal execution of portfolio transactions. *Journal of Risk* 3, 5-39.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Ch. 12.
