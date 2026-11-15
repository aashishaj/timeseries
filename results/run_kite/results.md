# rv_ladder (real:kite, dev)

- experiment: rv_ladder
- data: real:kite (data/kite)
- split: dev: scored sessions 2026-02-06 -> 2026-09-11; last 40 sessions held out
- symbols: RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, BHARTIARTL, SBIN, ITC, LT, KOTAKBANK
- models: naive = random_walk; hist_mean = historical_mean; seasonal_mean_5d = seasonal_naive, n_sessions=5; ewma_hl1 = ewma, halflife_sessions=1.0; ewma_hl5 = ewma, halflife_sessions=5.0; har = har, pooled=False; har_pooled = har, pooled=True; har_mix = har, intraday_beta=0.5, pooled=True; arma_logrv_mix = arma_logrv, intraday_beta=0.5, order=[1, 1]; garch = garch, asymmetric=False; gjr_garch = garch, asymmetric=True; lgbm = lightgbm; nbeats = nbeats, intraday_beta=0.5, lookback_days=30, n_blocks=4, n_epochs=100, width=64; nbeats_deep = nbeats, intraday_beta=0.5, lookback_days=60, n_blocks=6, n_epochs=150, width=128; chronos_tiny = chronos, intraday_beta=0.5, model_id=amazon/chronos-t5-tiny, num_samples=100; combo_equal = combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=equal; combo_opt = combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=qlike_opt
- git: 643374a-dirty
- walk-forward: expanding window, first 60 sessions for training, refit every 5 sessions

## Results vs naive

### 30min - RV over next 6 bars, origins every 1 bar(s) intraday

85,758 scored forecasts per model over 8,576 origins; mean target RV 0.09927 %^2; 13,402 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) | QLIKE / best lower rung | p vs best lower rung |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 1.2516 | 1.000 | 0.1649 | 1.000 | 1.000 | - | - | - | - | - |
| hist_mean | 0.6497 | 0.519 | 0.05241 | 0.318 | 0.843 | -10.94 | 7.53e-28 | 10/10 | - | - |
| seasonal_mean_5d | 0.5843 | 0.467 | 0.0514 | 0.312 | 0.666 | -12.06 | 1.65e-33 | 10/10 | - | - |
| ewma_hl1 | 0.4341 | 0.347 | 0.04482 | 0.272 | 0.595 | -15.01 | 6.44e-51 | 10/10 | 0.743 | 1.86e-37 |
| ewma_hl5 | 0.4598 | 0.367 | 0.04472 | 0.271 | 0.608 | -14.56 | 4.79e-48 | 10/10 | 0.787 | 1.42e-27 |
| har | 0.4892 | 0.391 | 0.0451 | 0.274 | 0.685 | -13.88 | 8.6e-44 | 10/10 | 0.837 | 1.09e-13 |
| har_pooled | 0.4903 | 0.392 | 0.04512 | 0.274 | 0.686 | -13.86 | 1.09e-43 | 10/10 | 0.839 | 1e-13 |
| har_mix | 0.4615 | 0.369 | 0.04464 | 0.271 | 0.639 | -14.52 | 9.53e-48 | 10/10 | 0.790 | 8.72e-21 |
| arma_logrv_mix | 0.4598 | 0.367 | 0.04437 | 0.269 | 0.618 | -14.57 | 4.09e-48 | 10/10 | 0.787 | 2.92e-22 |
| garch | 0.4336 | 0.346 | 0.04873 | 0.296 | 0.615 | -15.21 | 3.01e-52 | 10/10 | 0.742 | 4.93e-27 |
| gjr_garch | 0.4338 | 0.347 | 0.04943 | 0.300 | 0.616 | -15.21 | 3.18e-52 | 10/10 | 0.743 | 3.97e-27 |
| lgbm | 0.4120 | 0.329 | 0.04333 | 0.263 | 0.571 | -15.59 | 8.47e-55 | 10/10 | 0.950 | 2.41e-13 |
| nbeats | 0.4968 | 0.397 | 0.04761 | 0.289 | 0.646 | -13.95 | 3.29e-44 | 10/10 | 1.206 | 1.83e-41 |
| nbeats_deep | 0.4739 | 0.379 | 0.04495 | 0.273 | 0.612 | -14.37 | 7.46e-47 | 10/10 | 1.150 | 1.41e-36 |
| chronos_tiny | 0.4735 | 0.378 | 0.04494 | 0.273 | 0.588 | -14.39 | 6.41e-47 | 10/10 | 1.149 | 7.18e-32 |
| combo_equal | 0.4227 | 0.338 | 0.04297 | 0.261 | 0.585 | -15.31 | 6.32e-53 | 10/10 | 1.026 | 6.89e-08 |
| combo_opt | **0.4114** | 0.329 | 0.04327 | 0.262 | 0.584 | -15.58 | 9.81e-55 | 10/10 | 0.999 | 0.681 |

Forecast coverage below 100%: nbeats_deep 86.5%
Forecasts floored before QLIKE: naive 3

### next_session - RV over next 72 bars, origins at each session open

1,280 scored forecasts per model over 128 origins; mean target RV 1.601 %^2; 200 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) | QLIKE / best lower rung | p vs best lower rung |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.2889 | 1.000 | 3.269 | 1.000 | 1.000 | - | - | - | - | - |
| hist_mean | 0.2398 | 0.830 | 2.126 | 0.650 | 0.904 | -2.27 | 0.0234 | 8/10 | - | - |
| seasonal_mean_5d | 0.2092 | 0.724 | 2.118 | 0.648 | 0.866 | -5.40 | 6.82e-08 | 10/10 | - | - |
| ewma_hl1 | 0.2108 | 0.730 | 2.145 | 0.656 | 0.796 | -7.16 | 8.04e-13 | 10/10 | 1.008 | 0.847 |
| ewma_hl5 | 0.2013 | 0.697 | 1.971 | 0.603 | 0.794 | -6.00 | 2.03e-09 | 9/10 | 0.962 | 0.236 |
| har | 0.1898 | 0.657 | 1.979 | 0.605 | 0.876 | -6.51 | 7.69e-11 | 10/10 | 0.907 | 0.0445 |
| har_pooled | 0.1909 | 0.661 | 1.985 | 0.607 | 0.875 | -6.70 | 2.1e-11 | 10/10 | 0.913 | 0.0448 |
| har_mix | 0.1909 | 0.661 | 1.985 | 0.607 | 0.875 | -6.70 | 2.1e-11 | 10/10 | 0.913 | 0.0448 |
| arma_logrv_mix | 0.1813 | 0.627 | 1.888 | 0.578 | 0.798 | -7.95 | 1.82e-15 | 10/10 | 0.866 | 0.000353 |
| garch | 0.2264 | 0.784 | 3.249 | 0.994 | 0.900 | -3.52 | 0.000431 | 10/10 | 1.082 | 0.37 |
| gjr_garch | 0.2257 | 0.781 | 3.455 | 1.057 | 0.901 | -3.51 | 0.00044 | 10/10 | 1.079 | 0.396 |
| lgbm | 0.1950 | 0.675 | 2.054 | 0.628 | 0.852 | -6.92 | 4.45e-12 | 10/10 | 1.076 | 0.0172 |
| nbeats | 0.2962 | 1.025 | 3.032 | 0.928 | 1.002 | +0.38 | 0.704 | 5/10 | 1.634 | 7.28e-10 |
| nbeats_deep | 0.2393 | 0.828 | 2.077 | 0.635 | 0.855 | -2.98 | 0.00286 | 8/10 | 1.320 | 1.58e-05 |
| chronos_tiny | 0.2281 | 0.790 | 2.056 | 0.629 | 0.777 | -3.72 | 0.000196 | 9/10 | 1.259 | 0.000456 |
| combo_equal | **0.1788** | 0.619 | 1.872 | 0.573 | 0.792 | -8.14 | 3.95e-16 | 10/10 | 0.987 | 0.485 |
| combo_opt | 0.1871 | 0.648 | 2.014 | 0.616 | 0.837 | -7.81 | 5.61e-15 | 10/10 | 1.032 | 0.102 |

Forecast coverage below 100%: nbeats_deep 86.5%

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
- source rows read: 920,038; used: 892,799
- duplicate rows dropped: 0
- invalid price rows dropped: 0
- rows on non-session dates dropped: 550
- rows outside 09:15-15:15 dropped: 26,689
- continuous trading ends at 15:15 from 2026-08-03 (normally 15:30); every session uses the common 09:15-15:15 window so all days are comparable
- known special sessions ignored: 2026-11-08 (Muhurat trading (Sunday))

| symbol | full_sessions | partial_sessions | missing_sessions | bars_missing | pct_bars_missing |
|---|---|---|---|---|---|
| BHARTIARTL | 248 | 0 | 0 | 0 | 0.00 |
| HDFCBANK | 248 | 0 | 0 | 0 | 0.00 |
| ICICIBANK | 248 | 0 | 0 | 0 | 0.00 |
| INFY | 248 | 0 | 0 | 0 | 0.00 |
| ITC | 248 | 0 | 0 | 0 | 0.00 |
| KOTAKBANK | 248 | 0 | 0 | 0 | 0.00 |
| LT | 248 | 0 | 0 | 0 | 0.00 |
| RELIANCE | 248 | 0 | 0 | 0 | 0.00 |
| SBIN | 248 | 0 | 0 | 0 | 0.00 |
| TCS | 248 | 0 | 0 | 0 | 0.00 |
