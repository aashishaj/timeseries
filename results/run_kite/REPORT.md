# rv_ladder on real:kite

Development split only: the lockbox (last `holdout_sessions`) is still sealed.
Lower QLIKE / naive is better; `vs best lower rung` < 1 with a small p means the
model earned its place on the ladder.

## 30min

| model | rung | QLIKE / naive | vs best lower rung (p) | explained | median vol error | within +-25% |
|---|---|---|---|---|---|---|
| combo_opt | 5 | 0.042 | 0.999 (0.71) | 32% | 32% | 40% |
| lgbm | 3 | 0.042 | 0.949 (1.7e-08) | 32% | 31% | 42% |
| combo_equal | 5 | 0.043 | 1.024 (2e-06) | 30% | 33% | 40% |
| ewma_hl1 | 2 | 0.044 | 0.743 (1.7e-33) | 28% | 33% | 40% |
| garch | 2 | 0.044 | 0.745 (4.7e-24) | 27% | 34% | 39% |
| gjr_garch | 2 | 0.044 | 0.745 (3.6e-24) | 27% | 34% | 39% |
| arma_logrv_mix | 2 | 0.046 | 0.782 (4.4e-18) | 24% | 34% | 39% |
| ewma_hl5 | 2 | 0.046 | 0.784 (1.6e-24) | 24% | 35% | 38% |
| har_mix | 2 | 0.046 | 0.785 (5.2e-17) | 24% | 36% | 37% |
| nbeats_deep | 4 | 0.048 | 1.141 (1.2e-35) | 23% | 34% | 39% |
| har | 2 | 0.049 | 0.831 (6.7e-13) | 21% | 41% | 34% |
| har_pooled | 2 | 0.049 | 0.833 (4.7e-13) | 21% | 41% | 33% |
| nbeats | 4 | 0.050 | 1.203 (1.1e-39) | 21% | 34% | 38% |
| seasonal_mean_5d | 1 | 0.059 | - | 16% | 35% | 38% |
| hist_mean | 1 | 0.065 | - | 1% | 55% | 26% |
| naive | 1 | 1.000 | - | 17% | 39% | 33% |
| chronos_tiny | 4 | not scored: forecasts only 0% of targets | - | - | - | - |

## next_session

| model | rung | QLIKE / naive | vs best lower rung (p) | explained | median vol error | within +-25% |
|---|---|---|---|---|---|---|
| combo_equal | 5 | 0.607 | 0.994 (0.77) | 26% | 19% | 64% |
| arma_logrv_mix | 2 | 0.611 | 0.915 (0.022) | 25% | 19% | 63% |
| combo_opt | 5 | 0.612 | 1.001 (0.95) | 27% | 20% | 62% |
| har | 2 | 0.637 | 0.954 (0.33) | 22% | 22% | 57% |
| har_pooled | 2 | 0.640 | 0.958 (0.32) | 22% | 22% | 58% |
| har_mix | 2 | 0.640 | 0.958 (0.32) | 22% | 22% | 58% |
| ewma_hl5 | 2 | 0.648 | 0.970 (0.44) | 21% | 18% | 63% |
| seasonal_mean_5d | 1 | 0.668 | - | 23% | 19% | 62% |
| lgbm | 3 | 0.674 | 1.103 (0.0071) | 23% | 19% | 60% |
| ewma_hl1 | 2 | 0.679 | 1.017 (0.73) | 25% | 19% | 65% |
| nbeats_deep | 4 | 0.767 | 1.256 (1.1e-05) | 16% | 21% | 59% |
| gjr_garch | 2 | 0.777 | 1.163 (0.15) | 14% | 19% | 61% |
| garch | 2 | 0.780 | 1.169 (0.14) | 13% | 20% | 61% |
| hist_mean | 1 | 0.797 | - | 4% | 22% | 57% |
| naive | 1 | 1.000 | - | 17% | 21% | 58% |
| nbeats | 4 | 1.024 | 1.676 (1.3e-08) | 14% | 22% | 55% |
| chronos_tiny | 4 | not scored: forecasts only 0% of targets | - | - | - | - |

## Paper trading: volatility targeting (simulated, intraday MIS)

Target 15% annualized (intraday), capital Rs 1,000,000, gross exposure capped at 2x, 148 sessions. Long at the open, flat at the close, Zerodha intraday charges + slippage. `constant` = equal notional, no forecast.

| strategy | ann. return % | ann. vol % | vol / target | Sharpe | max DD % | worst day % | std(ret/forecast vol) | p99 abs(ret/forecast vol) | costs %/yr | avg gross lev. |
|---|---|---|---|---|---|---|---|---|---|---|
| constant | -27.51 | 10.59 | 0.71 | -2.60 | -18.10 | -1.90 | - | - | 24.21 | 0.80 |
| naive | -24.96 | 8.87 | 0.59 | -2.82 | -16.53 | -2.26 | 1.19 | 3.48 | 21.36 | 0.67 |
| hist_mean | -23.23 | 8.18 | 0.55 | -2.84 | -14.93 | -1.61 | 1.11 | 3.21 | 20.81 | 0.62 |
| seasonal_mean_5d | -21.78 | 8.11 | 0.54 | -2.68 | -14.22 | -1.82 | 1.08 | 3.14 | 20.43 | 0.62 |
| ewma_hl1 | -24.72 | 8.37 | 0.56 | -2.95 | -15.97 | -1.82 | 1.13 | 3.39 | 21.07 | 0.64 |
| ewma_hl5 | -23.32 | 7.92 | 0.53 | -2.94 | -14.97 | -1.52 | 1.07 | 3.07 | 20.50 | 0.61 |
| har | -21.67 | 7.67 | 0.51 | -2.83 | -14.06 | -1.82 | 1.01 | 3.12 | 19.74 | 0.58 |
| har_pooled | -21.74 | 7.53 | 0.50 | -2.89 | -14.03 | -1.68 | 1.00 | 2.92 | 19.62 | 0.57 |
| har_mix | -21.74 | 7.53 | 0.50 | -2.89 | -14.03 | -1.68 | 1.00 | 2.92 | 19.62 | 0.57 |
| arma_logrv_mix | -22.96 | 7.97 | 0.53 | -2.88 | -14.91 | -1.78 | 1.07 | 3.06 | 20.46 | 0.61 |
| garch | -25.60 | 8.39 | 0.56 | -3.05 | -16.36 | -1.89 | 1.13 | 3.81 | 20.80 | 0.63 |
| gjr_garch | -25.53 | 8.38 | 0.56 | -3.05 | -16.31 | -1.89 | 1.13 | 3.79 | 20.79 | 0.63 |
| lgbm | -23.47 | 7.72 | 0.51 | -3.04 | -15.20 | -1.72 | 1.03 | 3.07 | 19.90 | 0.59 |
| nbeats | -23.54 | 8.48 | 0.57 | -2.78 | -15.77 | -1.85 | 1.14 | 3.90 | 20.74 | 0.64 |
| nbeats_deep | -15.06 | 7.28 | 0.49 | -2.07 | -10.13 | -1.47 | 1.07 | 3.21 | 17.15 | 0.52 |
| chronos_tiny | 0.00 | 0.00 | 0.00 | - | 0.00 | 0.00 | - | - | 0.00 | 0.00 |
| combo_equal | -23.35 | 7.84 | 0.52 | -2.98 | -15.04 | -1.76 | 1.05 | 3.11 | 20.21 | 0.60 |
| combo_opt | -22.73 | 7.63 | 0.51 | -2.98 | -14.57 | -1.44 | 1.02 | 3.02 | 19.85 | 0.59 |

Volatility forecasts say nothing about direction: compare risk columns (vol /
target near 1, std(ret/forecast vol) near 1, smaller tails and worst days),
not returns. On synthetic data the true drift is zero, so returns are noise.


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

