# Methodology: signal construction

Each signal implements a uniform `fit / predict` API and produces an alpha
score per asset for the latest bar. Five concrete strategies ship in the
library; each is described below with its statistical content.

## Time-series momentum (TSMOM)

Moskowitz, Ooi, and Pedersen (2012) document a robust positive return to
positions sized by the sign of trailing returns across asset classes. The
position is scaled by inverse realized volatility so each asset
contributes the same ex-ante risk.

$$
w_{i,t} = \operatorname{sign}\left(\frac{p_{i,t-s}}{p_{i,t-s-L}} - 1\right) \cdot \frac{\sigma^*}{\hat{\sigma}_{i,t}^{\text{annual}}}.
$$

With $L = 252$ (twelve months) and $s = 21$ (one-month recency skip), this
captures the canonical Moskowitz et al. construction. The skip avoids the
well-documented one-month reversal.

## Cross-sectional momentum (XSMOM)

Jegadeesh and Titman (1993) showed that going long the top-decile and
short the bottom-decile by trailing 12-1 return earns a persistent
premium. We rank assets by

$$
m_{i,t}(L, s) = \frac{p_{i,t-s}}{p_{i,t-s-L}} - 1,
$$

then form a dollar-neutral portfolio: $+1/n_{\text{long}}$ on the top
decile and $-1/n_{\text{short}}$ on the bottom decile.

## Engle-Granger cointegration pairs

For a candidate pair $(i, j)$:

1. Estimate $\log p^i_t = \alpha + \beta \log p^j_t + \varepsilon_t$ by OLS.
2. Test the residual $\varepsilon_t$ with an Augmented Dickey-Fuller test; keep pairs with p-value below 0.05.
3. Fit an OU process to the residual; require the half-life to be no greater than 30 trading days.
4. Trade the standardized residual: short the spread at $z = +2$, long at $z = -2$, exit at $|z| < 0.5$, stop at $|z| > 4$.

The half-life filter is critical: a pair with a 200-day half-life is
either not a real pair or one whose mean-reversion is too slow to be
exploited within realistic holding periods.

## OU mean reversion (single name)

After removing market beta, the residual log-price should be stationary
under the mean-reversion hypothesis. We fit

$$
dx_t = -\theta (x_t - \mu) dt + \sigma dW_t
$$

via OLS regression of $x_{t+1}$ on $x_t$ (the discrete analogue), recover
$\theta$ and $\mu$, and trade $z = (x_t - \mu) / s_x$ with the same entry
and exit logic as the pairs strategy. Half-life $\log 2 / \theta$ must be
less than 30 days.

## ML signal

A gradient-boosted classifier trained on triple-barrier labels. Features:

- Momentum at horizons (5, 21, 63, 252).
- Realized volatility at (21, 63).
- Fractionally differenced log price with $d = 0.4$.
- Amihud illiquidity over 21 days.
- Distance to the 52-week high.

The target is the sign of the triple-barrier label (binary classifier
between up and not-up). The model is trained on the in-sample portion of
each walk-forward fold; we deliberately do not run a hyperparameter sweep
inside the validation harness because that would inflate the implicit
multiple-testing count we charge to the deflated Sharpe.

## References

- Avellaneda, M., Lee, J.H. (2010). Statistical arbitrage in the US equities market. *Quantitative Finance* 10, 761-782.
- Engle, R.F., Granger, C.W.J. (1987). Co-integration and error correction. *Econometrica* 55, 251-276.
- Jegadeesh, N., Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance* 48, 65-91.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chs 3, 6, 7.
- Moskowitz, T., Ooi, Y.H., Pedersen, L.H. (2012). Time series momentum. *Journal of Financial Economics* 104, 228-250.
