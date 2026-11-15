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


## Paper trading: held basket, volatility-managed (simulated, delivery)

Target 10% annualized (close to close), capital Rs 1,000,000, at most 1x invested, the rest in cash at 0%. A stock is traded at the open only when its weight is more than 25% away from its target. Zerodha delivery charges + slippage. `buy_hold` = equal amounts bought on day one and never traded.

148 sessions.

| strategy | ann. return % | ann. vol % | vol / target | Sharpe | max DD % | worst day % | std(ret/forecast vol) | turnover x/yr | costs %/yr | orders/month | avg invested |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_hold | -21.06 | 15.01 | 1.50 | -1.40 | -14.93 | -2.94 | - | 0.0 | 0.23 | 0.0 | 0.99 |
| naive | -25.23 | 11.43 | 1.14 | -2.21 | -16.09 | -2.04 | 1.23 | 34.9 | 5.68 | 86.7 | 0.79 |
| hist_mean | -19.89 | 12.07 | 1.21 | -1.65 | -14.57 | -2.82 | 1.18 | 0.6 | 0.36 | 2.6 | 0.74 |
| seasonal_mean_5d | -19.34 | 10.83 | 1.08 | -1.79 | -12.55 | -2.21 | 1.13 | 6.2 | 1.16 | 20.9 | 0.75 |
| ewma_hl1 | -20.80 | 10.90 | 1.09 | -1.91 | -13.72 | -2.02 | 1.16 | 7.9 | 1.52 | 27.8 | 0.77 |
| ewma_hl5 | -20.01 | 10.84 | 1.08 | -1.84 | -13.06 | -2.31 | 1.12 | 1.5 | 0.46 | 6.2 | 0.73 |
| har | -18.74 | 10.25 | 1.02 | -1.83 | -12.47 | -2.15 | 1.07 | 5.4 | 1.10 | 20.4 | 0.70 |
| har_pooled | -18.95 | 10.17 | 1.02 | -1.86 | -12.42 | -2.01 | 1.06 | 5.1 | 1.05 | 20.1 | 0.69 |
| har_mix | -18.95 | 10.17 | 1.02 | -1.86 | -12.42 | -2.01 | 1.06 | 5.1 | 1.05 | 20.1 | 0.69 |
| arma_logrv_mix | -17.71 | 10.63 | 1.06 | -1.67 | -12.28 | -2.22 | 1.12 | 4.8 | 1.04 | 18.3 | 0.73 |
| garch | -23.10 | 11.24 | 1.12 | -2.06 | -14.79 | -2.47 | 1.18 | 11.5 | 2.13 | 37.3 | 0.73 |
| gjr_garch | -22.69 | 11.15 | 1.12 | -2.03 | -14.57 | -2.49 | 1.17 | 11.5 | 2.12 | 37.0 | 0.73 |
| lgbm | -20.47 | 10.26 | 1.03 | -2.00 | -13.44 | -2.08 | 1.06 | 13.1 | 2.36 | 44.8 | 0.72 |
| nbeats | -22.79 | 11.08 | 1.11 | -2.06 | -14.89 | -1.87 | 1.18 | 40.2 | 6.45 | 96.6 | 0.78 |
| nbeats_deep | -7.93 | 9.89 | 0.99 | -0.80 | -6.47 | -1.73 | 1.10 | 13.5 | 2.16 | 41.3 | 0.64 |
| chronos_tiny | -21.54 | 11.81 | 1.18 | -1.82 | -14.30 | -2.67 | 1.25 | 6.1 | 1.23 | 20.9 | 0.79 |
| combo_equal | -19.49 | 10.36 | 1.04 | -1.88 | -12.96 | -2.18 | 1.09 | 5.2 | 1.11 | 20.4 | 0.71 |
| combo_opt | -19.95 | 10.05 | 1.01 | -1.99 | -12.97 | -2.01 | 1.05 | 6.5 | 1.27 | 24.8 | 0.71 |

Started late (no forecasts yet, so in cash; returns not comparable): `nbeats_deep` after 20 sessions.

Volatility forecasts say nothing about direction, and a long-only basket earns
what the market gives: read the risk and cost columns, not the returns.
`turnover x/yr` is value traded / equity after the first purchase; `costs %/yr`
includes the first purchase but no final sale. Dividends are not counted.


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

