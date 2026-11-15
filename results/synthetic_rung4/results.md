# rv_ladder (synthetic, dev)

- experiment: rv_ladder
- data: synthetic (data/synthetic)
- split: dev: scored sessions 2025-03-27 -> 2025-11-03; last 40 sessions held out
- symbols: RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, BHARTIARTL, SBIN, ITC, LT, KOTAKBANK
- models: naive = random_walk; arma_logrv_mix = arma_logrv, intraday_beta=0.5, order=[1, 1]; lgbm = lightgbm; nbeats = nbeats, intraday_beta=0.5, lookback_days=30, n_blocks=4, n_epochs=100, width=64
- git: 9185f19-dirty
- walk-forward: expanding window, first 60 sessions for training, refit every 5 sessions

## Results vs naive

### 30min - RV over next 6 bars, origins every 1 bar(s) intraday

102,226 scored forecasts per model over 10,360 origins; mean target RV 0.1496 %^2; 1,374 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) | QLIKE / best lower rung | p vs best lower rung |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.6172 | 1.000 | 0.04286 | 1.000 | 1.000 | - | - | - | - | - |
| arma_logrv_mix | 0.2898 | 0.470 | 0.02534 | 0.591 | 0.775 | -41.63 | 0 | 10/10 | 0.470 | 0 |
| lgbm | **0.2543** | 0.412 | 0.02237 | 0.522 | 0.731 | -46.65 | 0 | 10/10 | 0.878 | 4.51e-121 |
| nbeats | 0.3121 | 0.506 | 0.02688 | 0.627 | 0.799 | -38.00 | 5.51e-316 | 10/10 | 1.227 | 2e-137 |

### next_session - RV over next 75 bars, origins at each session open

1,429 scored forecasts per model over 148 origins; mean target RV 1.994 %^2; 51 (origin, symbol) pairs excluded (missing bars in the target window, zero RV, or a model without a forecast).

| model | QLIKE | QLIKE / naive | MSE | MSE / naive | MASE | DM stat | DM p | beats naive (symbols) | QLIKE / best lower rung | p vs best lower rung |
|---|---|---|---|---|---|---|---|---|---|---|
| naive | 0.1685 | 1.000 | 2.546 | 1.000 | 1.000 | - | - | - | - | - |
| arma_logrv_mix | **0.1086** | 0.644 | 1.668 | 0.655 | 0.814 | -9.38 | 6.66e-21 | 10/10 | 0.644 | 6.66e-21 |
| lgbm | 0.1119 | 0.664 | 1.673 | 0.657 | 0.817 | -7.61 | 2.83e-14 | 10/10 | 1.031 | 0.203 |
| nbeats | 0.1747 | 1.036 | 2.465 | 0.968 | 1.044 | +0.84 | 0.403 | 4/10 | 1.609 | 1.5e-21 |

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
