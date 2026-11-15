## Paper trading: held basket, volatility-managed (simulated, delivery)

Target 10% annualized (close to close), capital Rs 1,000,000, at most 1x invested, the rest in cash at 0%. A stock is traded at the open only when its weight is more than 25% away from its target. Zerodha delivery charges + slippage. `buy_hold` = equal amounts bought on day one and never traded.

### all sessions (188 sessions)

| strategy | ann. return % | ann. vol % | vol / target | Sharpe | max DD % | worst day % | std(ret/forecast vol) | turnover x/yr | costs %/yr | orders/month | avg invested |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_hold | -2.22 | 15.16 | 1.52 | -0.15 | -16.83 | -2.94 | - | 0.0 | 0.18 | 0.0 | 0.99 |
| hist_mean @ 25% | -4.04 | 12.09 | 1.21 | -0.33 | -15.03 | -2.82 | 1.19 | 0.7 | 0.31 | 2.9 | 0.74 |
| hist_mean @ 50% | -5.80 | 12.71 | 1.27 | -0.46 | -15.69 | -2.99 | 1.19 | 0.4 | 0.24 | 0.8 | 0.77 |
| arma_logrv_mix @ 25% | -4.40 | 10.95 | 1.10 | -0.40 | -14.60 | -2.22 | 1.14 | 5.0 | 1.02 | 18.8 | 0.74 |
| arma_logrv_mix @ 50% | -3.47 | 10.54 | 1.05 | -0.33 | -13.79 | -2.34 | 1.13 | 1.3 | 0.39 | 2.7 | 0.69 |

### run_kite: 2026-02-06 to 2026-09-11 (148 sessions)

| strategy | ann. return % | ann. vol % | vol / target | Sharpe | max DD % | worst day % | std(ret/forecast vol) | turnover x/yr | costs %/yr | orders/month | avg invested |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_hold | -21.06 | 15.01 | 1.50 | -1.40 | -14.93 | -2.94 | - | 0.0 | 0.23 | 0.0 | 0.99 |
| hist_mean @ 25% | -19.89 | 12.07 | 1.21 | -1.65 | -14.57 | -2.82 | 1.18 | 0.6 | 0.36 | 2.6 | 0.74 |
| hist_mean @ 50% | -21.08 | 13.02 | 1.30 | -1.62 | -14.93 | -2.99 | 1.18 | 0.5 | 0.31 | 1.0 | 0.79 |
| arma_logrv_mix @ 25% | -17.71 | 10.63 | 1.06 | -1.67 | -12.28 | -2.22 | 1.12 | 4.8 | 1.04 | 18.3 | 0.73 |
| arma_logrv_mix @ 50% | -17.56 | 10.39 | 1.04 | -1.69 | -12.26 | -2.34 | 1.11 | 1.6 | 0.47 | 3.1 | 0.69 |

### lockbox_kite: 2026-09-15 to 2026-11-12 (40 sessions)

| strategy | ann. return % | ann. vol % | vol / target | Sharpe | max DD % | worst day % | std(ret/forecast vol) | turnover x/yr | costs %/yr | orders/month | avg invested |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_hold | 67.47 | 15.08 | 1.51 | 4.47 | -3.50 | -1.57 | - | 0.0 | 0.00 | 0.0 | 0.99 |
| hist_mean @ 25% | 54.59 | 11.53 | 1.15 | 4.73 | -2.40 | -1.22 | 1.17 | 1.0 | 0.16 | 4.2 | 0.73 |
| hist_mean @ 50% | 50.72 | 10.90 | 1.09 | 4.65 | -2.40 | -1.11 | 1.16 | 0.0 | 0.00 | 0.0 | 0.71 |
| arma_logrv_mix @ 25% | 44.83 | 11.70 | 1.17 | 3.83 | -3.09 | -1.32 | 1.14 | 5.5 | 0.98 | 20.5 | 0.76 |
| arma_logrv_mix @ 50% | 48.67 | 10.56 | 1.06 | 4.61 | -2.35 | -1.12 | 1.12 | 0.4 | 0.07 | 1.1 | 0.69 |

Volatility forecasts say nothing about direction, and a long-only basket earns
what the market gives: read the risk and cost columns, not the returns.
`turnover x/yr` is value traded / equity after the first purchase; `costs %/yr`
includes the first purchase but no final sale. Dividends are not counted.
