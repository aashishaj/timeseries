# rv_ladder (synthetic, dev)

- experiment: rv_ladder
- data: synthetic (data/synthetic)
- split: dev: scored sessions 2025-03-27 -> 2025-11-03; last 40 sessions held out
- symbols: RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, BHARTIARTL, SBIN, ITC, LT, KOTAKBANK
- models: naive = random_walk; hist_mean = historical_mean; seasonal_mean_5d = seasonal_naive, n_sessions=5; ewma_hl1 = ewma, halflife_sessions=1.0; ewma_hl5 = ewma, halflife_sessions=5.0; har = har, pooled=False; har_pooled = har, pooled=True; har_mix = har, intraday_beta=0.5, pooled=True
- git: 63182ed-dirty
- walk-forward: expanding window, first 60 sessions for training, refit every 5 sessions

## Results vs naive

### 30min - RV over next 6 bars, origins every 1 bar(s) intraday

99,314 scored forecasts per model over 10,360 origins; mean target RV 0.1491 %^2; 4,286 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) |
|---|---|---|---|---|---|---|---|---|
| naive | 0.6171 | 1.000 | 0.04281 | 1.000 | 1.000 | - | - | - |
| hist_mean | 0.3930 | 0.637 | 0.03341 | 0.780 | 0.914 | -22.93 | 2.18e-116 | 10/10 |
| seasonal_mean_5d | 0.4078 | 0.661 | 0.03249 | 0.759 | 0.909 | -21.97 | 5.9e-107 | 10/10 |
| ewma_hl1 | 0.3002 | 0.486 | 0.02608 | 0.609 | 0.805 | -38.23 | 8.58e-320 | 10/10 |
| ewma_hl5 | 0.3296 | 0.534 | 0.02856 | 0.667 | 0.829 | -33.35 | 8.42e-244 | 10/10 |
| har | 0.3431 | 0.556 | 0.02908 | 0.679 | 0.834 | -30.23 | 1.08e-200 | 10/10 |
| har_pooled | 0.3382 | 0.548 | 0.029 | 0.678 | 0.828 | -31.16 | 4.33e-213 | 10/10 |
| har_mix | **0.2926** | 0.474 | 0.02554 | 0.597 | 0.769 | -40.39 | 0 | 10/10 |

Forecast coverage below 100%: ewma_hl1 98.8%, ewma_hl5 99.3%, har 99.1%, har_pooled 99.1%, har_mix 99.1%

### next_session - RV over next 75 bars, origins at each session open

1,389 scored forecasts per model over 148 origins; mean target RV 1.99 %^2; 91 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) |
|---|---|---|---|---|---|---|---|---|
| naive | 0.1686 | 1.000 | 2.525 | 1.000 | 1.000 | - | - | - |
| hist_mean | 0.1635 | 0.970 | 2.41 | 0.955 | 0.978 | -0.20 | 0.841 | 8/10 |
| seasonal_mean_5d | 0.1180 | 0.700 | 1.75 | 0.693 | 0.846 | -6.96 | 3.42e-12 | 10/10 |
| ewma_hl1 | 0.1201 | 0.712 | 1.802 | 0.714 | 0.847 | -10.30 | 7.23e-25 | 10/10 |
| ewma_hl5 | 0.1211 | 0.718 | 1.825 | 0.723 | 0.850 | -5.44 | 5.47e-08 | 10/10 |
| har | 0.1194 | 0.708 | 1.747 | 0.692 | 0.831 | -7.62 | 2.63e-14 | 10/10 |
| har_pooled | **0.1144** | 0.678 | 1.736 | 0.687 | 0.817 | -8.34 | 7.3e-17 | 10/10 |
| har_mix | 0.1144 | 0.678 | 1.736 | 0.687 | 0.817 | -8.34 | 7.3e-17 | 10/10 |

Forecast coverage below 100%: ewma_hl1 98.8%, ewma_hl5 99.3%, har 99.1%, har_pooled 99.1%, har_mix 99.1%

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
