# Synthetic data: realized volatility baselines (pipeline check, not market evidence)

- experiment: rv_baselines
- data_root: data/synthetic
- symbols: RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, BHARTIARTL, SBIN, ITC, LT, KOTAKBANK
- sessions: 248 (2025-01-01 -> 2025-12-31)
- models: naive = random_walk; hist_mean = historical_mean; seasonal_naive = seasonal_naive, n_sessions=1; seasonal_mean_5d = seasonal_naive, n_sessions=5
- git: 0d6e71f
- walk-forward: expanding window, first 60 sessions for training, refit every 5 sessions

## Results vs naive

### 30min - RV over next 6 bars, origins every 1 bar(s) intraday

129,050 scored forecasts per model over 13,160 origins; mean target RV 0.1237 %^2; 2,550 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) |
|---|---|---|---|---|---|---|---|---|
| naive | 0.6190 | 1.000 | 0.02482 | 1.000 | 1.000 | - | - | - |
| hist_mean | 0.3632 | 0.587 | 0.0174 | 0.701 | 0.880 | -23.99 | 3.62e-127 | 10/10 |
| seasonal_naive | 0.7777 | 1.256 | 0.02566 | 1.034 | 1.057 | +10.52 | 6.98e-26 | 0/10 |
| seasonal_mean_5d | **0.3557** | 0.575 | 0.01699 | 0.685 | 0.881 | -26.64 | 2.63e-156 | 10/10 |

Forecast coverage below 100%: seasonal_naive 99.1%

### next_session - RV over next 75 bars, origins at each session open

1,811 scored forecasts per model over 188 origins; mean target RV 1.641 %^2; 69 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) |
|---|---|---|---|---|---|---|---|---|
| naive | 0.0896 | 1.000 | 0.6882 | 1.000 | 1.000 | - | - | - |
| hist_mean | 0.1376 | 1.535 | 1.054 | 1.532 | 1.191 | +2.41 | 0.016 | 0/10 |
| seasonal_naive | 0.0896 | 1.000 | 0.6886 | 1.001 | 1.000 | -0.17 | 0.869 | 6/10 |
| seasonal_mean_5d | **0.0769** | 0.858 | 0.6617 | 0.962 | 0.960 | -3.94 | 8.15e-05 | 8/10 |

Forecast coverage below 100%: seasonal_naive 99.3%

How to read this:
* Target: realized variance (sum of squared 5-min percent log returns) over the horizon;
  the first bar of a session uses open->close, so overnight gaps are excluded.
* QLIKE and MSE are on variance; lower is better. `/ naive` < 1 means better than the
  random walk on the same origins. MASE = MAE / naive MAE on the same origins.
* DM stat: Diebold-Mariano on QLIKE vs naive (negative = lower loss than naive), using
  the cross-sectional mean loss per origin and Newey-West errors. Vol forecast errors are
  persistent, so treat borderline p-values with caution.
* beats naive (symbols): symbols where the model's mean QLIKE is below naive's.
* Best QLIKE per horizon in bold.


## Data quality

- sessions: 248 (2025-01-01 -> 2025-12-31)
- source rows read: 949,615; used: 919,045
- duplicate rows dropped: 460
- invalid price rows dropped: 0
- rows on non-session dates dropped: 600
- rows outside 09:15-15:30 dropped: 29,510
- known special sessions ignored: 2025-10-21 (Diwali Laxmi Pujan)

| symbol | full_sessions | partial_sessions | missing_sessions | bars_missing | pct_bars_missing |
|---|---|---|---|---|---|
| BHARTIARTL | 240 | 6 | 2 | 189 | 1.02 |
| HDFCBANK | 240 | 6 | 2 | 181 | 0.97 |
| ICICIBANK | 240 | 6 | 2 | 187 | 1.01 |
| INFY | 240 | 6 | 2 | 172 | 0.92 |
| ITC | 240 | 6 | 2 | 169 | 0.91 |
| KOTAKBANK | 240 | 6 | 2 | 182 | 0.98 |
| LT | 240 | 6 | 2 | 171 | 0.92 |
| RELIANCE | 240 | 6 | 2 | 183 | 0.98 |
| SBIN | 240 | 6 | 2 | 171 | 0.92 |
| TCS | 240 | 6 | 2 | 170 | 0.91 |
