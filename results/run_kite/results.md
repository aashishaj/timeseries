# rv_ladder (real:kite, dev)

- experiment: rv_ladder
- data: real:kite (data/kite)
- split: dev: scored sessions 2026-02-06 -> 2026-09-11; last 40 sessions held out
- symbols: RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, BHARTIARTL, SBIN, ITC, LT, KOTAKBANK
- models: naive = random_walk; hist_mean = historical_mean; seasonal_mean_5d = seasonal_naive, n_sessions=5; ewma_hl1 = ewma, halflife_sessions=1.0; ewma_hl5 = ewma, halflife_sessions=5.0; har = har, pooled=False; har_pooled = har, pooled=True; har_mix = har, intraday_beta=0.5, pooled=True; arma_logrv_mix = arma_logrv, intraday_beta=0.5, order=[1, 1]; garch = garch, asymmetric=False; gjr_garch = garch, asymmetric=True; lgbm = lightgbm; nbeats = nbeats, intraday_beta=0.5, lookback_days=30, n_blocks=4, n_epochs=100, width=64; nbeats_deep = nbeats, intraday_beta=0.5, lookback_days=60, n_blocks=6, n_epochs=150, width=128; chronos_tiny = chronos, intraday_beta=0.5, model_id=amazon/chronos-t5-tiny, num_samples=100; combo_equal = combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=equal; combo_opt = combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=qlike_opt
- git: 9f5e84b-dirty
- walk-forward: expanding window, first 60 sessions for training, refit every 5 sessions

## Results vs naive

### 30min - RV over next 6 bars, origins every 1 bar(s) intraday

88,698 scored forecasts per model over 8,870 origins; mean target RV 0.1009 %^2; 14,902 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) | QLIKE / best lower rung | p vs best lower rung |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 10.0053 | 1.000 | 0.1603 | 1.000 | 1.000 | - | - | - | - | - |
| hist_mean | 0.6475 | 0.065 | 0.05548 | 0.346 | 0.847 | -1.10 | 0.273 | 10/10 | - | - |
| seasonal_mean_5d | 0.5928 | 0.059 | 0.05553 | 0.346 | 0.687 | -1.10 | 0.27 | 10/10 | - | - |
| ewma_hl1 | 0.4404 | 0.044 | 0.04817 | 0.300 | 0.611 | -1.12 | 0.262 | 10/10 | 0.743 | 1.75e-33 |
| ewma_hl5 | 0.4650 | 0.046 | 0.04809 | 0.300 | 0.624 | -1.12 | 0.263 | 10/10 | 0.784 | 1.64e-24 |
| har | 0.4924 | 0.049 | 0.04847 | 0.302 | 0.699 | -1.12 | 0.265 | 10/10 | 0.831 | 6.7e-13 |
| har_pooled | 0.4940 | 0.049 | 0.04851 | 0.303 | 0.700 | -1.11 | 0.265 | 10/10 | 0.833 | 4.66e-13 |
| har_mix | 0.4652 | 0.046 | 0.04795 | 0.299 | 0.654 | -1.12 | 0.263 | 10/10 | 0.785 | 5.19e-17 |
| arma_logrv_mix | 0.4635 | 0.046 | 0.04769 | 0.298 | 0.632 | -1.12 | 0.263 | 10/10 | 0.782 | 4.43e-18 |
| garch | 0.4415 | 0.044 | 0.05213 | 0.325 | 0.632 | -1.12 | 0.262 | 10/10 | 0.745 | 4.7e-24 |
| gjr_garch | 0.4417 | 0.044 | 0.05282 | 0.329 | 0.633 | -1.12 | 0.262 | 10/10 | 0.745 | 3.57e-24 |
| lgbm | 0.4180 | 0.042 | 0.04668 | 0.291 | 0.588 | -1.12 | 0.261 | 10/10 | 0.949 | 1.65e-08 |
| nbeats | 0.5027 | 0.050 | 0.05074 | 0.317 | 0.660 | -1.11 | 0.265 | 10/10 | 1.203 | 1.07e-39 |
| nbeats_deep | 0.4769 | 0.048 | 0.04822 | 0.301 | 0.628 | -1.12 | 0.264 | 10/10 | 1.141 | 1.25e-35 |
| combo_equal | 0.4279 | 0.043 | 0.04633 | 0.289 | 0.600 | -1.12 | 0.262 | 10/10 | 1.024 | 1.96e-06 |
| combo_opt | **0.4175** | 0.042 | 0.04662 | 0.291 | 0.602 | -1.12 | 0.261 | 10/10 | 0.999 | 0.707 |
| chronos_tiny | - | - | - | - | - | - | - | nan | - | - |

Forecast coverage below 100%: nbeats_deep 86.4%, chronos_tiny 0.0%
Forecasts floored before QLIKE: naive 2.0

### next_session - RV over next 75 bars, origins at each session open

980 scored forecasts per model over 98 origins; mean target RV 1.807 %^2; 500 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) | QLIKE / best lower rung | p vs best lower rung |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.2784 | 1.000 | 3.776 | 1.000 | 1.000 | - | - | - | - | - |
| hist_mean | 0.2219 | 0.797 | 2.444 | 0.647 | 0.851 | -2.48 | 0.0132 | 8/10 | - | - |
| seasonal_mean_5d | 0.1859 | 0.668 | 2.411 | 0.638 | 0.850 | -5.34 | 9.25e-08 | 10/10 | - | - |
| ewma_hl1 | 0.1890 | 0.679 | 2.371 | 0.628 | 0.786 | -6.86 | 6.65e-12 | 10/10 | 1.017 | 0.73 |
| ewma_hl5 | 0.1803 | 0.648 | 2.247 | 0.595 | 0.785 | -5.71 | 1.16e-08 | 10/10 | 0.970 | 0.445 |
| har | 0.1774 | 0.637 | 2.279 | 0.604 | 0.862 | -5.78 | 7.55e-09 | 10/10 | 0.954 | 0.329 |
| har_pooled | 0.1781 | 0.640 | 2.289 | 0.606 | 0.862 | -5.94 | 2.91e-09 | 10/10 | 0.958 | 0.325 |
| har_mix | 0.1781 | 0.640 | 2.289 | 0.606 | 0.862 | -5.94 | 2.91e-09 | 10/10 | 0.958 | 0.325 |
| arma_logrv_mix | 0.1700 | 0.611 | 2.176 | 0.576 | 0.791 | -6.86 | 6.8e-12 | 10/10 | 0.915 | 0.022 |
| garch | 0.2173 | 0.780 | 3.745 | 0.992 | 0.890 | -3.22 | 0.00129 | 9/10 | 1.169 | 0.14 |
| gjr_garch | 0.2163 | 0.777 | 3.999 | 1.059 | 0.892 | -3.31 | 0.000935 | 9/10 | 1.163 | 0.149 |
| lgbm | 0.1875 | 0.674 | 2.436 | 0.645 | 0.863 | -6.15 | 7.91e-10 | 10/10 | 1.103 | 0.0071 |
| nbeats | 0.2850 | 1.024 | 3.51 | 0.930 | 1.032 | +0.33 | 0.744 | 4/10 | 1.676 | 1.33e-08 |
| nbeats_deep | 0.2136 | 0.767 | 2.402 | 0.636 | 0.866 | -3.63 | 0.000281 | 10/10 | 1.256 | 1.11e-05 |
| combo_equal | **0.1689** | 0.607 | 2.172 | 0.575 | 0.786 | -7.29 | 3.01e-13 | 10/10 | 0.994 | 0.767 |
| combo_opt | 0.1702 | 0.612 | 2.401 | 0.636 | 0.843 | -6.68 | 2.46e-11 | 10/10 | 1.001 | 0.948 |
| chronos_tiny | - | - | - | - | - | - | - | nan | - | - |

Forecast coverage below 100%: nbeats_deep 83.1%, chronos_tiny 0.0%

How to read this:
* Target: realized variance (sum of squared 5-min percent log returns) over the horizon;
  the first bar of a session uses open->close, so overnight gaps are excluded.
* QLIKE and MSE are on variance; lower is better. `/ naive` < 1 means better than the
  random walk on the same origins. MASE = MAE / naive MAE on the same origins.
* DM stat: Diebold-Mariano on QLIKE vs naive (negative = lower loss than naive), using
  the cross-sectional mean loss per origin and Newey-West errors. Vol forecast errors are
  persistent, so treat borderline p-values with caution.
* beats naive (symbols): symbols where the model's mean QLIKE is below naive's.
* QLIKE / best lower rung: against the lowest-QLIKE model of all lower rungs, on the
  same sample, with its DM p-value. The ladder rule is that this must be < 1.
* Best QLIKE per horizon in bold.


## Data quality

- sessions: 248 (2025-11-12 -> 2026-11-12)
- source rows read: 920,038; used: 919,488
- duplicate rows dropped: 0
- invalid price rows dropped: 0
- rows on non-session dates dropped: 550
- rows outside 09:15-15:30 dropped: 0
- known special sessions ignored: 2026-11-08 (Muhurat trading (Sunday))

| symbol | full_sessions | partial_sessions | missing_sessions | bars_missing | pct_bars_missing |
|---|---|---|---|---|---|
| BHARTIARTL | 178 | 70 | 0 | 210 | 1.13 |
| HDFCBANK | 178 | 70 | 0 | 210 | 1.13 |
| ICICIBANK | 178 | 70 | 0 | 210 | 1.13 |
| INFY | 178 | 70 | 0 | 210 | 1.13 |
| ITC | 178 | 70 | 0 | 210 | 1.13 |
| KOTAKBANK | 178 | 70 | 0 | 210 | 1.13 |
| LT | 178 | 70 | 0 | 210 | 1.13 |
| RELIANCE | 178 | 70 | 0 | 210 | 1.13 |
| SBIN | 178 | 70 | 0 | 210 | 1.13 |
| TCS | 178 | 70 | 0 | 210 | 1.13 |
