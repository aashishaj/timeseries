## Paper trading: volatility targeting (simulated, intraday MIS)

Target 15% annualized (intraday), capital Rs 1,000,000, gross exposure capped at 2x, 148 sessions. Long at the open, flat at the close, Zerodha intraday charges + slippage. `constant` = equal notional, no forecast.

| strategy | ann. return % | ann. vol % | vol / target | Sharpe | max DD % | worst day % | std(ret/forecast vol) | p99 abs(ret/forecast vol) | costs %/yr | avg gross lev. |
|---|---|---|---|---|---|---|---|---|---|---|
| constant | -25.09 | 7.62 | 0.51 | -3.29 | -19.39 | -1.46 | - | - | 30.08 | 0.99 |
| naive | -26.07 | 5.34 | 0.36 | -4.88 | -16.67 | -1.00 | 1.11 | 3.21 | 25.50 | 0.78 |
| seasonal_mean_5d | -24.95 | 5.18 | 0.35 | -4.82 | -15.80 | -0.86 | 1.06 | 2.97 | 25.01 | 0.75 |
| arma_logrv_mix | -24.20 | 5.07 | 0.34 | -4.77 | -15.61 | -0.87 | 1.04 | 2.84 | 24.73 | 0.73 |
| garch | -23.12 | 5.22 | 0.35 | -4.43 | -15.24 | -0.92 | 1.05 | 2.97 | 24.73 | 0.73 |
| lgbm | -25.29 | 5.22 | 0.35 | -4.85 | -16.14 | -0.89 | 1.06 | 3.00 | 25.01 | 0.75 |
| combo_opt | -24.23 | 5.04 | 0.34 | -4.81 | -15.47 | -0.88 | 1.03 | 2.81 | 24.57 | 0.73 |

Volatility forecasts say nothing about direction: compare risk columns (vol /
target near 1, std(ret/forecast vol) near 1, smaller tails and worst days),
not returns. On synthetic data the true drift is zero, so returns are noise.
