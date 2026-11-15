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
