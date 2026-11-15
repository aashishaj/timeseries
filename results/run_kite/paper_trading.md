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
