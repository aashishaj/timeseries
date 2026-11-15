## Paper trading: volatility targeting (simulated, intraday MIS)

Target 15% annualized (intraday), capital Rs 1,000,000, gross exposure capped at 2x, 40 sessions. Long at the open, flat at the close, Zerodha intraday charges + slippage. `constant` = equal notional, no forecast.

| strategy | ann. return % | ann. vol % | vol / target | Sharpe | max DD % | worst day % | std(ret/forecast vol) | p99 abs(ret/forecast vol) | costs %/yr | avg gross lev. |
|---|---|---|---|---|---|---|---|---|---|---|
| constant | -32.73 | 12.83 | 0.86 | -2.55 | -5.53 | -1.96 | - | - | 30.37 | 1.00 |
| naive | -23.62 | 10.10 | 0.67 | -2.34 | -4.31 | -1.51 | 1.18 | 3.38 | 26.37 | 0.82 |
| hist_mean | -26.37 | 10.09 | 0.67 | -2.61 | -4.55 | -1.56 | 1.17 | 3.20 | 26.31 | 0.79 |
| seasonal_mean_5d | -30.96 | 9.66 | 0.64 | -3.21 | -4.76 | -1.42 | 1.11 | 2.84 | 25.68 | 0.78 |
| ewma_hl1 | -30.11 | 9.94 | 0.66 | -3.03 | -4.52 | -1.42 | 1.14 | 2.89 | 26.53 | 0.81 |
| ewma_hl5 | -32.57 | 10.36 | 0.69 | -3.14 | -4.93 | -1.54 | 1.18 | 2.99 | 26.91 | 0.82 |
| har | -26.65 | 9.55 | 0.64 | -2.79 | -4.27 | -1.46 | 1.09 | 2.77 | 25.39 | 0.75 |
| har_pooled | -26.57 | 9.45 | 0.63 | -2.81 | -4.26 | -1.44 | 1.08 | 2.78 | 25.37 | 0.75 |
| har_mix | -26.57 | 9.45 | 0.63 | -2.81 | -4.26 | -1.44 | 1.08 | 2.78 | 25.37 | 0.75 |
| arma_logrv_mix | -27.19 | 9.66 | 0.64 | -2.82 | -4.35 | -1.47 | 1.11 | 2.78 | 25.74 | 0.77 |
| garch | -26.75 | 10.29 | 0.69 | -2.60 | -4.42 | -1.62 | 1.17 | 3.07 | 26.74 | 0.81 |
| gjr_garch | -26.58 | 10.29 | 0.69 | -2.58 | -4.40 | -1.62 | 1.17 | 3.07 | 26.74 | 0.81 |
| lgbm | -26.08 | 9.99 | 0.67 | -2.61 | -4.32 | -1.62 | 1.13 | 2.72 | 26.25 | 0.80 |
| nbeats | -26.08 | 11.14 | 0.74 | -2.34 | -4.20 | -1.64 | 1.30 | 3.66 | 27.10 | 0.86 |
| nbeats_deep | -29.56 | 11.78 | 0.79 | -2.51 | -4.55 | -1.87 | 1.32 | 3.65 | 27.94 | 0.88 |
| chronos_tiny | -27.97 | 11.30 | 0.75 | -2.47 | -4.74 | -1.81 | 1.28 | 3.59 | 27.81 | 0.87 |
| combo_equal | -26.66 | 9.79 | 0.65 | -2.72 | -4.33 | -1.51 | 1.11 | 2.74 | 26.03 | 0.78 |
| combo_opt | -26.25 | 9.44 | 0.63 | -2.78 | -4.14 | -1.56 | 1.07 | 2.69 | 25.25 | 0.75 |

Volatility forecasts say nothing about direction: compare risk columns (vol /
target near 1, std(ret/forecast vol) near 1, smaller tails and worst days),
not returns. On synthetic data the true drift is zero, so returns are noise.
