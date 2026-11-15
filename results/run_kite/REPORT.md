# rv_ladder on real:kite

Development split only: the lockbox (last `holdout_sessions`) is still sealed.
Lower QLIKE / naive is better; `vs best lower rung` < 1 with a small p means the
model earned its place on the ladder.

## 30min

| model | rung | QLIKE / naive | vs best lower rung (p) | explained | median vol error | within +-25% |
|---|---|---|---|---|---|---|
| combo_opt | 5 | 0.329 | 0.999 (0.68) | 32% | 32% | 41% |
| lgbm | 3 | 0.329 | 0.950 (2.4e-13) | 32% | 31% | 42% |
| combo_equal | 5 | 0.338 | 1.026 (6.9e-08) | 30% | 33% | 40% |
| garch | 2 | 0.346 | 0.742 (4.9e-27) | 28% | 34% | 39% |
| gjr_garch | 2 | 0.347 | 0.743 (4e-27) | 28% | 34% | 39% |
| ewma_hl1 | 2 | 0.347 | 0.743 (1.9e-37) | 28% | 33% | 40% |
| arma_logrv_mix | 2 | 0.367 | 0.787 (2.9e-22) | 24% | 34% | 39% |
| ewma_hl5 | 2 | 0.367 | 0.787 (1.4e-27) | 24% | 35% | 38% |
| har_mix | 2 | 0.369 | 0.790 (8.7e-21) | 24% | 36% | 37% |
| chronos_tiny | 4 | 0.378 | 1.149 (7.2e-32) | 23% | 33% | 40% |
| nbeats_deep | 4 | 0.379 | 1.150 (1.4e-36) | 23% | 34% | 39% |
| har | 2 | 0.391 | 0.837 (1.1e-13) | 21% | 41% | 34% |
| har_pooled | 2 | 0.392 | 0.839 (1e-13) | 21% | 41% | 33% |
| nbeats | 4 | 0.397 | 1.206 (1.8e-41) | 22% | 34% | 38% |
| seasonal_mean_5d | 1 | 0.467 | - | 17% | 34% | 38% |
| hist_mean | 1 | 0.519 | - | 1% | 56% | 26% |
| naive | 1 | 1.000 | - | 18% | 39% | 33% |

## next_session

| model | rung | QLIKE / naive | vs best lower rung (p) | explained | median vol error | within +-25% |
|---|---|---|---|---|---|---|
| combo_equal | 5 | 0.619 | 0.987 (0.48) | 33% | 20% | 60% |
| arma_logrv_mix | 2 | 0.627 | 0.866 (0.00035) | 32% | 20% | 61% |
| combo_opt | 5 | 0.648 | 1.032 (0.1) | 33% | 21% | 60% |
| har | 2 | 0.657 | 0.907 (0.044) | 29% | 23% | 54% |
| har_pooled | 2 | 0.661 | 0.913 (0.045) | 29% | 23% | 54% |
| har_mix | 2 | 0.661 | 0.913 (0.045) | 29% | 23% | 54% |
| lgbm | 3 | 0.675 | 1.076 (0.017) | 30% | 20% | 59% |
| ewma_hl5 | 2 | 0.697 | 0.962 (0.24) | 27% | 19% | 61% |
| seasonal_mean_5d | 1 | 0.724 | - | 27% | 20% | 60% |
| ewma_hl1 | 2 | 0.730 | 1.008 (0.85) | 30% | 19% | 65% |
| gjr_garch | 2 | 0.781 | 1.079 (0.4) | 19% | 22% | 57% |
| garch | 2 | 0.784 | 1.082 (0.37) | 18% | 21% | 57% |
| chronos_tiny | 4 | 0.790 | 1.259 (0.00046) | 25% | 19% | 63% |
| nbeats_deep | 4 | 0.828 | 1.320 (1.6e-05) | 22% | 20% | 59% |
| hist_mean | 1 | 0.830 | - | 6% | 25% | 50% |
| naive | 1 | 1.000 | - | 23% | 21% | 57% |
| nbeats | 4 | 1.025 | 1.634 (7.3e-10) | 19% | 23% | 55% |

## Paper trading: volatility targeting (simulated, intraday MIS)

Target 15% annualized (intraday), capital Rs 1,000,000, gross exposure capped at 2x, 148 sessions. Long at the open, flat at the close, Zerodha intraday charges + slippage. `constant` = equal notional, no forecast.

| strategy | ann. return % | ann. vol % | vol / target | Sharpe | max DD % | worst day % | std(ret/forecast vol) | p99 abs(ret/forecast vol) | costs %/yr | avg gross lev. |
|---|---|---|---|---|---|---|---|---|---|---|
| constant | -45.86 | 11.22 | 0.75 | -4.09 | -28.89 | -1.86 | - | - | 30.37 | 1.00 |
| naive | -44.71 | 9.86 | 0.66 | -4.54 | -28.24 | -2.24 | 1.21 | 3.72 | 28.10 | 0.90 |
| hist_mean | -38.00 | 8.78 | 0.59 | -4.33 | -23.92 | -1.60 | 1.08 | 3.18 | 26.31 | 0.78 |
| seasonal_mean_5d | -39.22 | 8.95 | 0.60 | -4.38 | -24.91 | -1.83 | 1.10 | 3.35 | 26.81 | 0.83 |
| ewma_hl1 | -44.02 | 9.32 | 0.62 | -4.72 | -27.70 | -1.81 | 1.15 | 3.57 | 27.73 | 0.87 |
| ewma_hl5 | -40.42 | 8.72 | 0.58 | -4.64 | -25.42 | -1.52 | 1.09 | 3.09 | 26.84 | 0.82 |
| har | -37.48 | 8.35 | 0.56 | -4.49 | -23.62 | -1.81 | 1.02 | 3.08 | 25.66 | 0.77 |
| har_pooled | -37.58 | 8.22 | 0.55 | -4.57 | -23.58 | -1.67 | 1.01 | 2.95 | 25.54 | 0.76 |
| har_mix | -37.58 | 8.22 | 0.55 | -4.57 | -23.58 | -1.67 | 1.01 | 2.95 | 25.54 | 0.76 |
| arma_logrv_mix | -39.76 | 8.72 | 0.58 | -4.56 | -25.10 | -1.77 | 1.07 | 3.08 | 26.58 | 0.81 |
| garch | -42.77 | 9.08 | 0.61 | -4.71 | -26.84 | -1.86 | 1.11 | 3.79 | 26.69 | 0.82 |
| gjr_garch | -42.65 | 9.05 | 0.60 | -4.71 | -26.77 | -1.87 | 1.11 | 3.80 | 26.69 | 0.82 |
| lgbm | -40.85 | 8.41 | 0.56 | -4.85 | -25.70 | -1.59 | 1.05 | 3.08 | 26.18 | 0.80 |
| nbeats | -40.58 | 9.58 | 0.64 | -4.24 | -26.12 | -2.28 | 1.19 | 3.76 | 27.30 | 0.86 |
| nbeats_deep | -32.13 | 8.23 | 0.55 | -3.91 | -20.77 | -1.51 | 1.11 | 3.28 | 23.66 | 0.74 |
| chronos_tiny | -43.11 | 9.67 | 0.64 | -4.46 | -27.26 | -1.92 | 1.19 | 3.37 | 28.20 | 0.89 |
| combo_equal | -40.08 | 8.54 | 0.57 | -4.69 | -25.20 | -1.69 | 1.05 | 3.20 | 26.25 | 0.79 |
| combo_opt | -40.07 | 8.39 | 0.56 | -4.77 | -25.03 | -1.61 | 1.04 | 3.18 | 26.10 | 0.79 |

Volatility forecasts say nothing about direction: compare risk columns (vol /
target near 1, std(ret/forecast vol) near 1, smaller tails and worst days),
not returns. On synthetic data the true drift is zero, so returns are noise.


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

