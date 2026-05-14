# Methodology: returns and features

## Returns

Two conventions, both supported throughout the library.

**Log returns.**

$$
r_t = \log\frac{p_t}{p_{t-1}}.
$$

Log returns add across time, which makes them ideal for cumulative computations and statistical work. They are the input to almost every downstream module in this library.

**Simple returns.**

$$
r_t = \frac{p_t}{p_{t-1}} - 1.
$$

Simple returns aggregate across assets in a weighted portfolio, which makes them the natural unit for performance attribution. The library converts between log and simple as needed.

## Fractional differencing

Many price series are nearly integrated of order 1, so differencing them once yields a stationary returns series at the cost of erasing the memory in the levels. Hosking (1981) showed how to take a *fractional* difference: an order `d` in `(0, 1)` that preserves long memory while reducing the order of integration.

The fixed-width-window FFD operator (Lopez de Prado AFML Ch. 5) is

$$
(\nabla^d X)_t = \sum_{k=0}^{K} \omega_k X_{t-k}, \quad \omega_k = (-1)^k \binom{d}{k},
$$

where the window is truncated at the smallest `K` such that $|\omega_K| < \tau$ for a threshold `tau`. The recursion

$$
\omega_k = -\omega_{k-1} \cdot \frac{d - k + 1}{k}
$$

makes the weight computation trivial.

QuantForge implements this in `quantforge.features.returns.frac_diff_ffd`. A typical workflow on a log-price series picks `d` as the smallest order at which the ADF p-value falls below 0.05; usually `d` in `(0.2, 0.5)` suffices.

## Volatility estimators

Four estimators are supported, ranked roughly by efficiency given complete OHLC data.

- **Close to close.**

  $$
  \hat{\sigma}_t = \sqrt{\frac{1}{n-1} \sum_{i=t-n+1}^{t} (r_i - \bar{r})^2}.
  $$

- **Parkinson.** Uses only high and low.

  $$
  \hat{\sigma}^2 = \frac{1}{4 n \log 2} \sum_i \left[\log\frac{h_i}{l_i}\right]^2.
  $$

  Biased downward when overnight returns are non-trivial.

- **Garman-Klass.** Combines high, low, open, and close.

  $$
  \hat{\sigma}^2 = \frac{1}{n}\sum_i \left[\frac{1}{2}\left(\log\frac{h_i}{l_i}\right)^2 - (2 \log 2 - 1)\left(\log\frac{c_i}{o_i}\right)^2\right].
  $$

- **Yang-Zhang.** Drift independent; combines the overnight, intraday-close-to-open, and Rogers-Satchell pieces. The weight $k = 0.34 / (1.34 + (n+1)/(n-1))$ minimizes asymptotic variance.

The library returns *per-bar* standard deviations. Annualize by multiplying by $\sqrt{252}$ via `constants.VOL_ANNUALIZATION_FACTOR`.

## Microstructure proxies

Three classic daily-bar proxies.

- **Amihud illiquidity:** $\text{ILLIQ}_t = \frac{1}{n}\sum |r_i| / V_i$.
- **Roll spread:** $\hat{s}_t = 2 \sqrt{-\text{Cov}(\Delta p_t, \Delta p_{t-1})}$ if the covariance is negative.
- **Kyle's lambda:** OLS slope of returns on signed dollar volume over a rolling window.

These are *proxies*: tick data is the right input for true microstructure work. On daily bars they nonetheless correlate well with intraday measures and serve as useful conditioning variables.

## Triple-barrier method

Lopez de Prado AFML Ch. 3. For each event `t_0` we define three barriers:

- An upper barrier at $+\pi \cdot \sigma_{t_0}$, where $\pi$ is the profit-take multiplier.
- A lower barrier at $-\lambda \cdot \sigma_{t_0}$, where $\lambda$ is the stop-loss multiplier.
- A vertical barrier at $t_0 + h$ for horizon $h$.

The label is the sign of the first-touched barrier. Triple-barrier labels are far more informative than fixed-horizon labels because they incorporate path-dependent stops and take-profits that real strategies impose.

## References

- Asness, C.S., Frazzini, A., Pedersen, L.H. (2019). Quality minus junk. *Review of Accounting Studies* 24, 34-112.
- Hosking, J.R.M. (1981). Fractional differencing. *Biometrika* 68, 165-176.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chs 3 and 5.
- Yang, D., Zhang, Q. (2000). Drift-independent volatility estimation. *Journal of Business* 73, 477-491.
