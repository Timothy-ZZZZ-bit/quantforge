# Methodology: portfolio construction

Five allocators ship with the library. Each is described mathematically and
in implementation terms.

## Equal weight

Trivial baseline. Two modes:

- Signless: $w_i = 1/n$ for every name with a finite alpha.
- Signed: $w_i = \operatorname{sign}(\alpha_i) / n_{\text{nonzero}}$.

## Mean-variance with Ledoit-Wolf shrinkage

The textbook Markowitz problem is

$$
\max_w \; w^\top \mu - \frac{\lambda}{2} w^\top \Sigma w
\quad \text{s.t.} \quad |w_i| \le c, \; \sum_i |w_i| \le L.
$$

The sample covariance $\hat{\Sigma}$ is dominated under expected-Frobenius
loss by the Ledoit-Wolf shrinkage estimator

$$
\Sigma^{\text{LW}} = (1 - \delta) \hat{\Sigma} + \delta \, \mu_F I,
$$

where $\mu_F$ is the average sample variance and $\delta$ is the shrinkage
intensity computed in closed form from the data. QuantForge uses
`sklearn.covariance.LedoitWolf` as the estimator and `scipy.optimize.minimize`
with SLSQP to solve the QP.

## Equal-risk contribution (ERC)

Maillard, Roncalli, and Teiletche (2010) define the ERC portfolio as the
positive weight vector for which every name contributes the same marginal
risk:

$$
\text{RC}_i(w) := w_i \cdot (\Sigma w)_i = \frac{1}{n} \cdot w^\top \Sigma w.
$$

We solve via Spinu (2013)'s cyclic coordinate descent: each coordinate
update is the positive root of the quadratic $a_i w_i^2 + b_i w_i - c = 0$
with $a_i = \Sigma_{ii}$, $b_i = (\Sigma w)_i - \Sigma_{ii} w_i$, $c = 1/n$.
The library also exposes a SLSQP reference solver in
`solve_erc_slsqp`; the unit tests confirm the two implementations agree
to within $10^{-3}$.

## Hierarchical Risk Parity (HRP)

Lopez de Prado (2016). Three steps:

1. Compute the correlation distance $d_{ij} = \sqrt{(1 - \rho_{ij}) / 2}$ and build a single-linkage cluster tree.
2. Quasi-diagonalize the covariance: reorder so correlated assets are adjacent.
3. Recursive bisection: at each level, split the ordered list, compute inverse-variance weights for each half, and allocate between the halves in proportion to the inverse of each half's variance.

HRP avoids the well-known instability of mean-variance optimization under
near-singular covariance matrices, and its weights are typically more
diversified across clusters.

## Black-Litterman with sample views

Black and Litterman (1992) construct a Bayesian posterior over expected
returns that blends a market-implied prior $\Pi$ with caller-provided views
$Q$:

$$
\hat{\mu} = \left[(\tau \Sigma)^{-1} + P^\top \Omega^{-1} P\right]^{-1}
            \left[(\tau \Sigma)^{-1} \Pi + P^\top \Omega^{-1} Q\right].
$$

QuantForge treats each signal score as an absolute view ($P = I$, $Q$ is
the vector of alpha scores). The diagonal entries of $\Omega$ are set
proportional to the variance of each name, so noisier signals get
downweighted. The posterior mean is then fed into the MVO solver.

## Constraints

Per-name caps (`max_weight`, `min_weight`), gross-leverage cap, and an
optional turnover cap are applied post-optimization in
`portfolio.constraints.apply_constraints`. When the caps and gross-leverage
target are mutually infeasible, the caps take precedence. The turnover
cap is enforced by pulling the new weights toward the prior portfolio
along the connecting line until the cap binds.

## References

- Black, F., Litterman, R. (1992). Global Portfolio Optimization. *Financial Analysts Journal* 48, 28-43.
- Ledoit, O., Wolf, M. (2004). Honey, I shrunk the sample covariance matrix. *Journal of Portfolio Management* 30, 110-119.
- Lopez de Prado, M. (2016). Building diversified portfolios that outperform out-of-sample. *Journal of Portfolio Management* 42, 59-69.
- Maillard, S., Roncalli, T., Teiletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management* 36, 60-70.
- Markowitz, H. (1952). Portfolio selection. *Journal of Finance* 7, 77-91.
- Spinu, F. (2013). An algorithm for computing risk parity weights. SSRN 2297383.
