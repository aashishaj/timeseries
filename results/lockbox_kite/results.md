# rv_ladder (real:kite, final)

- experiment: rv_ladder
- data: real:kite (data/kite)
- split: final: scored sessions 2026-09-15 -> 2026-11-12 (LOCKBOX)
- symbols: RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, BHARTIARTL, SBIN, ITC, LT, KOTAKBANK
- models: naive = random_walk; hist_mean = historical_mean; seasonal_mean_5d = seasonal_naive, n_sessions=5; ewma_hl1 = ewma, halflife_sessions=1.0; ewma_hl5 = ewma, halflife_sessions=5.0; har = har, pooled=False; har_pooled = har, pooled=True; har_mix = har, intraday_beta=0.5, pooled=True; arma_logrv_mix = arma_logrv, intraday_beta=0.5, order=[1, 1]; garch = garch, asymmetric=False; gjr_garch = garch, asymmetric=True; lgbm = lightgbm; nbeats = nbeats, intraday_beta=0.5, lookback_days=30, n_blocks=4, n_epochs=100, width=64; nbeats_deep = nbeats, intraday_beta=0.5, lookback_days=60, n_blocks=6, n_epochs=150, width=128; chronos_tiny = chronos, intraday_beta=0.5, model_id=amazon/chronos-t5-tiny, num_samples=100; combo_equal = combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=equal; combo_opt = combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=qlike_opt
- git: 2bc461f
- walk-forward: expanding window, first 208 sessions for training, refit every 5 sessions

## Results vs naive

### 30min - RV over next 6 bars, origins every 1 bar(s) intraday

26,800 scored forecasts per model over 2,680 origins; mean target RV 0.1141 %^2; 0 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) | QLIKE / best lower rung | p vs best lower rung |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 1.3350 | 1.000 | 0.1789 | 1.000 | 1.000 | - | - | - | - | - |
| hist_mean | 0.5899 | 0.442 | 0.0544 | 0.304 | 0.742 | -3.77 | 0.000166 | 10/10 | - | - |
| seasonal_mean_5d | 0.5618 | 0.421 | 0.04897 | 0.274 | 0.660 | -4.11 | 4e-05 | 10/10 | - | - |
| ewma_hl1 | 0.4197 | 0.314 | 0.04247 | 0.237 | 0.589 | -4.65 | 3.28e-06 | 10/10 | 0.747 | 5.44e-18 |
| ewma_hl5 | 0.4395 | 0.329 | 0.04441 | 0.248 | 0.576 | -4.59 | 4.37e-06 | 10/10 | 0.782 | 1.75e-20 |
| har | 0.4462 | 0.334 | 0.04335 | 0.242 | 0.634 | -4.48 | 7.55e-06 | 10/10 | 0.794 | 4.87e-11 |
| har_pooled | 0.4457 | 0.334 | 0.04316 | 0.241 | 0.636 | -4.46 | 8.05e-06 | 10/10 | 0.793 | 1.28e-10 |
| har_mix | 0.4497 | 0.337 | 0.04368 | 0.244 | 0.633 | -4.47 | 7.84e-06 | 10/10 | 0.801 | 1.89e-10 |
| arma_logrv_mix | 0.4479 | 0.335 | 0.04388 | 0.245 | 0.623 | -4.47 | 7.68e-06 | 10/10 | 0.797 | 8.54e-11 |
| garch | 0.4222 | 0.316 | 0.04089 | 0.229 | 0.571 | -4.76 | 1.95e-06 | 10/10 | 0.752 | 7.82e-21 |
| gjr_garch | 0.4220 | 0.316 | 0.04046 | 0.226 | 0.571 | -4.75 | 2.02e-06 | 10/10 | 0.751 | 8.7e-21 |
| lgbm | **0.3995** | 0.299 | 0.03936 | 0.220 | 0.561 | -4.89 | 1.02e-06 | 10/10 | 0.952 | 0.0048 |
| nbeats | 0.4982 | 0.373 | 0.04733 | 0.265 | 0.642 | -4.22 | 2.4e-05 | 10/10 | 1.247 | 9.11e-16 |
| nbeats_deep | 0.4706 | 0.352 | 0.04544 | 0.254 | 0.590 | -4.38 | 1.21e-05 | 10/10 | 1.178 | 1.69e-11 |
| chronos_tiny | 0.4696 | 0.352 | 0.04489 | 0.251 | 0.588 | -4.37 | 1.24e-05 | 10/10 | 1.176 | 1.47e-10 |
| combo_equal | 0.4092 | 0.307 | 0.04068 | 0.227 | 0.575 | -4.73 | 2.23e-06 | 10/10 | 1.024 | 0.0675 |
| combo_opt | 0.3995 | 0.299 | 0.03927 | 0.220 | 0.575 | -4.88 | 1.06e-06 | 10/10 | 1.000 | 0.969 |

### next_session - RV over next 72 bars, origins at each session open

400 scored forecasts per model over 40 origins; mean target RV 1.85 %^2; 0 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) | QLIKE / best lower rung | p vs best lower rung |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.2305 | 1.000 | 2.945 | 1.000 | 1.000 | - | - | - | - | - |
| hist_mean | 0.2108 | 0.914 | 2.147 | 0.729 | 0.843 | -1.02 | 0.309 | 5/10 | - | - |
| seasonal_mean_5d | 0.1906 | 0.827 | 2.16 | 0.733 | 0.928 | -2.24 | 0.0254 | 8/10 | - | - |
| ewma_hl1 | 0.1677 | 0.728 | 1.811 | 0.615 | 0.785 | -3.32 | 0.000911 | 9/10 | 0.880 | 0.0444 |
| ewma_hl5 | 0.1881 | 0.816 | 2.027 | 0.688 | 0.806 | -1.65 | 0.099 | 9/10 | 0.987 | 0.865 |
| har | 0.1593 | 0.691 | 1.848 | 0.628 | 0.836 | -5.03 | 4.9e-07 | 10/10 | 0.836 | 0.00201 |
| har_pooled | 0.1537 | 0.667 | 1.813 | 0.616 | 0.823 | -5.68 | 1.31e-08 | 10/10 | 0.806 | 0.000183 |
| har_mix | 0.1537 | 0.667 | 1.813 | 0.616 | 0.823 | -5.68 | 1.31e-08 | 10/10 | 0.806 | 0.000183 |
| arma_logrv_mix | 0.1600 | 0.694 | 1.858 | 0.631 | 0.808 | -5.33 | 9.86e-08 | 10/10 | 0.840 | 0.000892 |
| garch | 0.1827 | 0.793 | 1.95 | 0.662 | 0.806 | -2.52 | 0.0117 | 8/10 | 0.959 | 0.587 |
| gjr_garch | 0.1822 | 0.791 | 1.923 | 0.653 | 0.804 | -2.55 | 0.0108 | 8/10 | 0.956 | 0.579 |
| lgbm | 0.1529 | 0.663 | 1.705 | 0.579 | 0.788 | -4.93 | 8.05e-07 | 10/10 | 0.995 | 0.912 |
| nbeats | 0.3282 | 1.424 | 2.979 | 1.012 | 1.067 | +2.64 | 0.00837 | 1/10 | 2.146 | 1.3e-06 |
| nbeats_deep | 0.2704 | 1.173 | 2.254 | 0.765 | 0.863 | +1.13 | 0.259 | 4/10 | 1.768 | 0.000449 |
| chronos_tiny | 0.2291 | 0.994 | 2.114 | 0.718 | 0.834 | -0.07 | 0.943 | 5/10 | 1.498 | 9.63e-05 |
| combo_equal | 0.1452 | 0.630 | 1.664 | 0.565 | 0.763 | -5.90 | 3.62e-09 | 10/10 | 0.949 | 0.063 |
| combo_opt | **0.1419** | 0.616 | 1.654 | 0.562 | 0.800 | -5.06 | 4.12e-07 | 10/10 | 0.928 | 0.179 |

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
