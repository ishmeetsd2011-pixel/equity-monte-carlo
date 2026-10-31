# Equity Monte Carlo Lab

A runnable Streamlit research application for single-stock and long-only portfolios spanning NSE, BSE, NYSE and NASDAQ. Supports mixed INR/USD holdings, four simulation models, contributions, rebalancing, benchmark/alternative allocation comparisons, risk diagnostics, and downloadable reports.

**Educational tool, not investment advice. Historical data does not guarantee future results.**

## Setup and run

Python 3.11+ is required (validated here with Python 3.13).

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app/main.py
```

Run commands from the repository root. Yahoo downloads require internet access. The UI includes an explicitly labelled **Synthetic offline demo data** option so all models and exports work without a data provider. No API keys are needed. Never interpret demo results as forecasts of actual market data.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python work/benchmark.py
```

The benchmark measures simulation plus portfolio accounting, excluding data download and Plotly rendering. A 10,000-path, five-asset, 252-day run used 20.2 MB for stored portfolio values. Initial measured times on this machine: GBM 0.19s, Student-t 0.32s, bootstrap 0.14s, GARCH 0.33s. Hardware, fitting and history length affect results. There are no Python loops over simulation paths in the core; loops over dates and fitted assets are intentional. The chart displays at most 25 individual paths, plus percentile bands computed from all paths.

`requirements.txt` pins the tested runtime dependencies for deployment. `requirements-dev.txt` adds pytest for development. `requirements.lock.txt` is a historical local environment snapshot and is not used for Cloud deployment. Use Python 3.13 with these exact deployment pins; the source targets Python 3.11+ but other versions may require different compatible dependency versions.

## Community Cloud deployment

Follow DEPLOYMENT.md for the complete GitHub push and Cloud deployment guide. Deploy the repository root with entrypoint `app/main.py`, root `requirements.txt`, and **Python 3.13** selected in Advanced settings. No secrets or system packages are required. The root `.streamlit/config.toml` contains the deployment configuration.

## Workflow

1. Select ticker market. US symbols remain unchanged (`AAPL`, `MSFT`). NSE/BSE selection appends `.NS`/`.BO` to bare symbols. In Mixed / US mode enter full Indian symbols, e.g. `AAPL, RELIANCE.NS, TCS.NS`. Supported benchmark indices include `^GSPC`, `^NSEI`, `^BSESN`.
2. Enter weights summing to one, or share counts. In share mode, current *adjusted-share equivalents* multiplied by the latest converted adjusted closes determine initial capital, superseding the investment field. Fractional holdings are supported. These are total-return research holdings, not broker-reconciled live share quantities.
3. Choose INR or USD, model, horizon, paths (1,000–100,000), annualisation (252 default US/mixed, 248 default Indian), seed, and lookback. Horizons expressed in months/years map to rounded annualisation-based simulation steps.
4. Set rebalancing, monthly contributions, target and risk-free rate. Initial government-yield proxies are **indicative, editable defaults**, not current market quotes. “Load 10Y government yield” requests the latest available FRED India benchmark monthly or US daily Treasury observation and displays its date. India series can lag; verify the date and supply a current rate if needed. A retrieval failure retains a labelled indicative fallback. A 10Y yield is a conventional Sharpe proxy, not a riskless rate for every horizon.
5. Optionally choose Nifty 50 / S&P 500 and/or enter an alternative vector of weights for the same assets. Comparisons use identical capital and contributions in the same currency and on the common data window.
6. Run. Changing inputs does not replace the completed result until another run succeeds. Refresh data queues a forced download for the next run. Failed runs retain the last successful result with its settings.
7. Download terminal outcomes, daily bands, risk and comparison CSVs, or a self-contained interactive HTML report. Open the HTML in a browser and use Print → Save as PDF for a PDF copy.

## Models in plain language

All models work on **daily log returns of base-currency adjusted prices**. Let `m` be their mean and `C` their sample daily covariance. Cholesky factors generate dependent shocks. Singular/indefinite covariance matrices are symmetrized and eigenvalues clipped to a small positive floor before refactoring. Optional scikit-learn Ledoit–Wolf shrinkage reduces sample covariance estimation noise. Bootstrap resamples observations and does not use the covariance toggle.

- **GBM:** `S[t+1] = S[t] * exp(m + L z)`, `L L' = C`. With empirical log mean `m`, arithmetic SDE drift is `m + diag(C)/2`; this avoids subtracting a variance correction twice. This gives a lognormal distribution of prices, constant parameters and Gaussian log returns.
- **Student-t:** fit a common degrees-of-freedom parameter to pooled standardized asset returns with SciPy; constrain it to [2.1, 200]. Use correlated Gaussian draws and a shared chi-square scale, standardized by `sqrt((df-2)/chi2(df))`. This is an elliptical multivariate t with finite log-return variance and shared tail events. It preserves the fitted covariance in log returns. **Student-t log returns have no finite exponential moment: theoretical expected prices are infinite.** Displayed expected values are finite-sample estimates that can be unstable; medians/quantiles and explicit stress analysis are preferable. No hidden tail clipping is applied.
- **Historical / circular block bootstrap:** resample entire multivariate rows together, preserving observed equity/FX dependence. Block length 1 is ordinary historical bootstrap. Longer circular blocks preserve within-block serial dependence and volatility clusters; dependence is interrupted at block boundaries. Blocks can wrap from the end of history to the start. Covariance shrinkage and antithetic sampling do not apply to this model.
- **GARCH(1,1):** use `arch` to fit each asset's constant-mean volatility model on percentage log returns. Retain its final conditional variance and residual. Simulate `h[t+1] = omega + alpha*epsilon[t]^2 + beta*h[t]`, with jointly Gaussian standardized shocks using the correlation of fitted standardized residuals. Variance state is specific to each path. This is constant conditional correlation GARCH, not DCC or a full multivariate GARCH fit. Fits that fail convergence or are effectively nonstationary are rejected explicitly. A constant asset is deterministic.

Antithetic variates pair innovations with their negatives (and share the t scale); an odd path count leaves one unpaired draw. This can reduce variance for some estimators, not every payoff or tail metric. GARCH pairs subsequently develop symmetric residuals and matching variance states. Seeds reproduce a run with the same inputs and library environment. Comparisons with differing numbers of assets do not constitute paired pathwise relative-performance experiments.

The drift override is an annual compounded **log-growth** scenario: an input `g` becomes daily log drift `log(1+g)/annualisation`, applied to each base-currency asset. It replaces empirical drift, including when bootstrapping centered observations. It is not a calibration to expected arithmetic terminal return. Historical drift is noisy and can overwhelm forecasts over long horizons.

## Currency, calendars and accounting

Yahoo's `auto_adjust=True` Close adjusts for splits and dividends. US assets are USD; `.NS`, `.BO`, `^NSEI`, `^BSESN` are INR. `INR=X` supplies INR per USD. Convert every historical asset observation **before** estimating returns: US→INR multiplies by FX; India→USD divides. Thus FX movements and their empirical equity dependence enter the joint simulation directly. FX is not frozen at today's spot and not independently double-counted. Only the documented INR/USD equity universes are supported; other exchanges or currencies would require metadata-aware mapping.

Use the **union of observed dates**, including FX dates, forward-filling holidays and removing initial rows without complete histories. Never backward-fill pre-listing observations. This adds zero local-price returns on closed-market sessions; converted asset values may still change with FX. It can damp estimated volatility and distort contemporaneous correlations. US/India close times are asynchronous, not simultaneous snapshots. Long gaps (>10 calendar days) and stale tails (>10 days relative to requested end) fail explicitly. Unlisted/invalid symbols, empty/short histories, corrupt caches and API failures produce actionable errors. A missing ticker is never silently removed or weights renormalized.

CSV caches live in `.cache/prices`, keyed by symbol, requested dates and adjustment version; writes are atomic. Streamlit adds a one-hour in-memory frame cache; end-date keys rotate daily, while same-date local files remain until Refresh data. Neither layer silently substitutes stale data on provider failure. Yahoo's end date is exclusive, so the current incomplete session is excluded.

Future dates are **weekday approximations**, not exact NSE/BSE/US holiday calendars. Contributions arrive after that day's return on the first future weekday of a new calendar month; no initial-day contribution occurs. They buy target weights, even in buy-and-hold mode. Rebalancing occurs after the contribution every 1/3/12 calendar months measured from the starting month. All holdings are long-only with fractional shares and no leverage, tax, spread, slippage or transaction fees. Adjusted-price growth implicitly reinvests distributions.

The application streams one paths × assets day at a time and retains only paths × dates portfolio values and path diagnostics. A 600 MB output budget rejects excessive horizon/path/comparison combinations before allocation. This is an output budget, not a guarantee on total process RAM: chart quantiles, temporary arrays and Streamlit serialization require additional memory. The UI updates progress during simulation and caches histories; downloads and simulations still occupy the active Streamlit script until complete.

## Risk definitions

- Expected/median terminal wealth and terminal sample standard deviation are in the selected currency.
- Probability of loss compares terminal wealth with **initial capital plus every contribution**. Target probability compares final wealth with the chosen target, including contributed capital.
- Loss = total paid in − terminal wealth. 95%/99% VaR is the interpolated empirical loss quantile; CVaR is the mean of sampled losses at or above that threshold. Negative VaR is allowed and means the relevant scenario still gains money. Discrete/tied empirical distributions need not have exactly 5%/1% of paths in their CVaR tail. Contributions are not discounted; these are not IRR or opportunity-cost losses.
- Daily time-weighted return = pre-contribution wealth / previous post-contribution wealth − 1. Chaining these returns removes external cash flows. Maximum drawdown uses this unit NAV, so an SIP cannot erase an investment drawdown. Both its mean and full empirical distribution are displayed.
- Annual return per path = chained growth^(annualisation / simulated days) − 1. Annual volatility uses sample daily simple-return standard deviation × sqrt(annualisation). Sharpe uses annualized mean daily excess return divided by annualized volatility, with daily rate `(1 + annual_rf)^(1/annualisation) − 1`. Undefined Sharpe for zero volatility is NaN. Metrics table reports means across paths for these three measures.
- Fan charts show 5/25/50/75/95 percentiles, which are pointwise quantiles, not individual investable paths or confidence intervals for the model's true parameters.

## Structure and checks

```text
app/main.py          Streamlit inputs, five tabs, comparisons, downloads
app/charts.py        Plotly fan charts and terminal histograms
core/data.py         adjusted downloads, ticker validation, calendar alignment, CSV cache
core/currency.py     historical INR/USD conversion
core/models.py       estimation, PSD repair, GBM/t/bootstrap/GARCH shocks
core/portfolio.py    streaming holdings, SIP, rebalancing, unit-NAV diagnostics
core/metrics.py      risk metrics and CSV frames
core/report.py       self-contained HTML report
tests/              convergence, dependence, determinism, data, accounting, UI tests
work/benchmark.py    reproducible four-model performance measurement
```

Tests cover GBM mean/variance against lognormal theory, Cholesky correlation reproduction, PSD fallback, reproducible seeds for all models, antithetic centering, block integrity, shrinkage, VaR/CVaR on known arrays, INR/USD conversion, ticker/API errors, cache refresh and holiday alignment, stale symbols, contributions and drawdown neutrality, buy-and-hold versus rebalancing, memory validation, and an end-to-end offline Streamlit run. Tests do not rely on external market services. Live Yahoo availability is separate from deterministic correctness.

## Limitations and future improvements

Yahoo is an unofficial, rate-limited, sometimes incomplete source: corporate-action corrections, stale quotes and provider outages remain possible. Adjusted history may change. Neither downloads nor simulations resolve survivorship bias or make historical estimates reliable forecasts. Fixed means, fitted correlations and volatility specifications can break in new regimes. No model parameter uncertainty, contagion/DCC, market closures, liquidity events, bankruptcies, discrete dividend cash accounting or trade execution constraints are modelled. Index adjusted closes may be price indices rather than total-return benchmarks. Multi-year extrapolation and fitted t tails can be extremely unstable.

Useful extensions: regime switching, parameter uncertainty, DCC/FX factors, exact exchange calendars, stress scenarios, walk-forward forecast validation, options overlays, tax lots and Indian LTCG/STCG and US capital gains, and broker-grade corporate-action handling.

## References

- [yfinance download API](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html)
- [arch volatility forecasting](https://arch.readthedocs.io/en/latest/univariate/forecasting.html)
- [FRED US 10Y Treasury yield](https://fred.stlouisfed.org/series/DGS10)
- [FRED India 10Y government benchmark](https://fred.stlouisfed.org/series/INDIRLTLT01STM)
