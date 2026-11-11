# rv_ladder (synthetic, dev): combinations vs their members

- experiment: rv_ladder
- data: synthetic (data/synthetic)
- split: dev: scored sessions 2025-03-27 -> 2025-11-03; last 40 sessions held out
- symbols: RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, BHARTIARTL, SBIN, ITC, LT, KOTAKBANK
- models: naive = random_walk; seasonal_mean_5d = seasonal_naive, n_sessions=5; har_mix = har, intraday_beta=0.5, pooled=True; arma_logrv_mix = arma_logrv, intraday_beta=0.5, order=[1, 1]; garch = garch, asymmetric=False; lgbm = lightgbm; combo_equal = combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=equal; combo_opt = combination, members=[{'type': 'lightgbm'}, {'type': 'garch'}, {'type': 'arma_logrv', 'params': {'order': [1, 1], 'intraday_beta': 0.5}}, {'type': 'har', 'params': {'pooled': True, 'intraday_beta': 0.5}}], method=qlike_opt
- git: 0fb5acc-dirty
- walk-forward: expanding window, first 60 sessions for training, refit every 5 sessions

## Results vs naive

### 30min - RV over next 6 bars, origins every 1 bar(s) intraday

101,260 scored forecasts per model over 10,360 origins; mean target RV 0.1492 %^2; 2,340 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) | QLIKE / best lower rung | p vs best lower rung |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.6172 | 1.000 | 0.04285 | 1.000 | 1.000 | - | - | - | - | - |
| seasonal_mean_5d | 0.4077 | 0.661 | 0.03267 | 0.763 | 0.912 | -22.32 | 2.45e-110 | 10/10 | - | - |
| har_mix | 0.2927 | 0.474 | 0.02566 | 0.599 | 0.770 | -41.04 | 0 | 10/10 | 0.718 | 6.62e-168 |
| arma_logrv_mix | 0.2897 | 0.469 | 0.02531 | 0.591 | 0.775 | -41.39 | 0 | 10/10 | 0.711 | 5.55e-169 |
| garch | 0.2860 | 0.463 | 0.02486 | 0.580 | 0.784 | -43.55 | 0 | 10/10 | 0.701 | 3.4e-139 |
| lgbm | 0.2542 | 0.412 | 0.02234 | 0.521 | 0.731 | -46.38 | 0 | 10/10 | 0.889 | 1.15e-100 |
| combo_equal | 0.2688 | 0.436 | 0.02372 | 0.554 | 0.746 | -44.72 | 0 | 10/10 | 1.057 | 1.91e-55 |
| combo_opt | **0.2542** | 0.412 | 0.02225 | 0.519 | 0.742 | -46.17 | 0 | 10/10 | 1.000 | 0.988 |

Forecast coverage below 100%: har_mix 99.1%

### next_session - RV over next 75 bars, origins at each session open

1,416 scored forecasts per model over 148 origins; mean target RV 1.991 %^2; 64 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) | QLIKE / best lower rung | p vs best lower rung |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.1695 | 1.000 | 2.565 | 1.000 | 1.000 | - | - | - | - | - |
| seasonal_mean_5d | 0.1180 | 0.696 | 1.759 | 0.686 | 0.843 | -7.17 | 7.69e-13 | 10/10 | - | - |
| har_mix | 0.1143 | 0.674 | 1.747 | 0.681 | 0.814 | -8.54 | 1.39e-17 | 10/10 | 0.968 | 0.16 |
| arma_logrv_mix | 0.1088 | 0.642 | 1.674 | 0.653 | 0.811 | -9.41 | 4.97e-21 | 10/10 | 0.922 | 0.000148 |
| garch | 0.1379 | 0.813 | 2.021 | 0.788 | 0.909 | -4.43 | 9.32e-06 | 10/10 | 1.168 | 0.000995 |
| lgbm | 0.1122 | 0.662 | 1.68 | 0.655 | 0.813 | -7.64 | 2.22e-14 | 10/10 | 1.031 | 0.203 |
| combo_equal | **0.1081** | 0.638 | 1.667 | 0.650 | 0.793 | -9.42 | 4.68e-21 | 10/10 | 0.993 | 0.648 |
| combo_opt | 0.1082 | 0.638 | 1.655 | 0.645 | 0.823 | -8.74 | 2.29e-18 | 10/10 | 0.994 | 0.872 |

Forecast coverage below 100%: har_mix 99.1%

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

- sessions: 248 (2025-01-01 -> 2025-12-31)
- source rows read: 949,611; used: 919,037
- duplicate rows dropped: 460
- invalid price rows dropped: 0
- rows on non-session dates dropped: 600
- rows outside 09:15-15:30 dropped: 29,514
- known special sessions ignored: 2025-10-21 (Diwali Laxmi Pujan)

| symbol | full_sessions | partial_sessions | missing_sessions | bars_missing | pct_bars_missing |
|---|---|---|---|---|---|
| BHARTIARTL | 241 | 5 | 2 | 169 | 0.91 |
| HDFCBANK | 240 | 6 | 2 | 169 | 0.91 |
| ICICIBANK | 240 | 6 | 2 | 180 | 0.97 |
| INFY | 240 | 6 | 2 | 176 | 0.95 |
| ITC | 241 | 5 | 2 | 177 | 0.95 |
| KOTAKBANK | 240 | 6 | 2 | 186 | 1.00 |
| LT | 240 | 6 | 2 | 177 | 0.95 |
| RELIANCE | 240 | 6 | 2 | 186 | 1.00 |
| SBIN | 240 | 6 | 2 | 183 | 0.98 |
| TCS | 240 | 6 | 2 | 178 | 0.96 |
