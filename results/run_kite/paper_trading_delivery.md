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
