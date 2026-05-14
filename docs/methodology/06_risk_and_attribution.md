# Methodology: risk and attribution

## Value at risk

Three flavors of VaR are provided.

- **Parametric.** Assumes returns are normal. $\text{VaR}_\alpha = -(\mu + z_\alpha \sigma)$, where $z_\alpha$ is the lower-tail normal quantile. Fast but optimistic; ignores fat tails.

- **Historical.** $\text{VaR}_\alpha = -\hat{Q}(1 - \alpha)$, the empirical quantile of past returns. Captures fat tails but not the tail beyond the empirical maximum loss.

- **Cornish-Fisher.** Adjusts the normal quantile for sample skew $S$ and excess kurtosis $K$:

  $$
  z^* = z + \tfrac{1}{6}(z^2 - 1) S + \tfrac{1}{24}(z^3 - 3z) K - \tfrac{1}{36}(2 z^3 - 5 z) S^2.
  $$

Each VaR function returns a point estimate alongside a bootstrap 95% confidence interval, the conservative choice for any risk-management workflow.

## Expected shortfall (CVaR)

$$
\text{ES}_\alpha = -\mathbb{E}[r \mid r \le -\text{VaR}_\alpha].
$$

ES is coherent in the sense of Artzner et al. (1999), unlike VaR; for risk management it is the right object to look at.

## Drawdown

Per-bar drawdown is $W_t / \max_{s \le t} W_s - 1$. We report maximum drawdown, time underwater, and recovery time from the worst trough.

## Position sizing

- **Fractional Kelly.** $f^* = \mu / \sigma^2$ on annualized inputs; scale by a fraction (0.25 by default) to reduce drawdowns.

- **Volatility targeting.** Scale gross exposure by $\sigma^* / \hat{\sigma}_{\text{annual}}$ to hit a target annualized volatility.

Both are implemented in `quantforge.risk.sizing`. Volatility targeting is the more common production choice because Kelly is acutely sensitive to estimation error in $\mu$.

## Stress tests

Historical scenario replay against three canonical windows:

- 2008 Global Financial Crisis (2008-09 to 2009-03).
- COVID-19 crash (2020-02 to 2020-04).
- 2022 inflation-driven drawdown (2022-01 to 2022-10).

Each window reports total return, Sharpe, max drawdown, and 95% ES.

## Factor attribution

We regress portfolio excess returns onto the Fama-French five-factor model plus Carhart momentum:

$$
r_{p,t} - r_{f,t} = \alpha + \beta_{\text{MKT}} \text{MKT}_t + \beta_{\text{SMB}} \text{SMB}_t + \beta_{\text{HML}} \text{HML}_t + \beta_{\text{RMW}} \text{RMW}_t + \beta_{\text{CMA}} \text{CMA}_t + \beta_{\text{MOM}} \text{MOM}_t + \varepsilon_t.
$$

Standard errors are Newey-West HAC adjusted with the Andrews (1991) optimal lag, $L^* = \lfloor 4 (n/100)^{2/9} \rfloor$. A statistically significant $\alpha$ after factor adjustment is what one is actually looking for in a research result.

## References

- Andrews, D.W.K. (1991). Heteroskedasticity and autocorrelation consistent covariance matrix estimation. *Econometrica* 59, 817-858.
- Artzner, P., Delbaen, F., Eber, J.-M., Heath, D. (1999). Coherent measures of risk. *Mathematical Finance* 9, 203-228.
- Carhart, M.M. (1997). On persistence in mutual fund performance. *Journal of Finance* 52, 57-82.
- Fama, E.F., French, K.R. (2015). A five-factor asset pricing model. *Journal of Financial Economics* 116, 1-22.
- Newey, W.K., West, K.D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica* 55, 703-708.
