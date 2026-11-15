## Paper trading: held basket, volatility-managed (simulated, delivery)

Target 10% annualized (close to close), capital Rs 1,000,000, at most 1x invested, the rest in cash at 0%. A stock is traded at the open only when its weight is more than 0% away from its target. Zerodha delivery charges + slippage. `buy_hold` = equal amounts bought on day one and never traded.

148 sessions.

| strategy | ann. return % | ann. vol % | vol / target | Sharpe | max DD % | worst day % | std(ret/forecast vol) | turnover x/yr | costs %/yr | orders/month | avg invested |
|---|---|---|---|---|---|---|---|---|---|---|---|
| buy_hold | -21.06 | 15.01 | 1.50 | -1.40 | -14.93 | -2.94 | - | 0.0 | 0.23 | 0.0 | 0.99 |
| hist_mean @ 0% | -21.36 | 11.61 | 1.16 | -1.84 | -14.46 | -2.63 | 1.18 | 3.0 | 1.84 | 125.3 | 0.73 |
| hist_mean @ 10% | -19.73 | 11.75 | 1.17 | -1.68 | -14.24 | -2.75 | 1.17 | 1.1 | 0.49 | 10.1 | 0.73 |
| hist_mean @ 25% | -19.89 | 12.07 | 1.21 | -1.65 | -14.57 | -2.82 | 1.18 | 0.6 | 0.36 | 2.6 | 0.74 |
| hist_mean @ 50% | -21.08 | 13.02 | 1.30 | -1.62 | -14.93 | -2.99 | 1.18 | 0.5 | 0.31 | 1.0 | 0.79 |
| arma_logrv_mix @ 0% | -22.97 | 10.88 | 1.09 | -2.11 | -14.92 | -2.12 | 1.12 | 14.7 | 3.92 | 188.4 | 0.75 |
| arma_logrv_mix @ 10% | -21.25 | 10.86 | 1.09 | -1.96 | -14.05 | -2.14 | 1.11 | 10.4 | 2.23 | 74.5 | 0.75 |
| arma_logrv_mix @ 25% | -17.71 | 10.63 | 1.06 | -1.67 | -12.28 | -2.22 | 1.12 | 4.8 | 1.04 | 18.3 | 0.73 |
| arma_logrv_mix @ 50% | -17.56 | 10.39 | 1.04 | -1.69 | -12.26 | -2.34 | 1.11 | 1.6 | 0.47 | 3.1 | 0.69 |
| lgbm @ 0% | -26.67 | 10.49 | 1.05 | -2.54 | -16.52 | -2.02 | 1.06 | 26.7 | 5.60 | 196.4 | 0.74 |
| lgbm @ 10% | -25.32 | 10.49 | 1.05 | -2.41 | -15.86 | -2.07 | 1.06 | 23.3 | 4.40 | 124.0 | 0.74 |
| lgbm @ 25% | -20.47 | 10.26 | 1.03 | -2.00 | -13.44 | -2.08 | 1.06 | 13.1 | 2.36 | 44.8 | 0.72 |
| lgbm @ 50% | -16.66 | 9.28 | 0.93 | -1.80 | -11.30 | -2.04 | 1.06 | 3.7 | 0.76 | 6.8 | 0.63 |

Volatility forecasts say nothing about direction, and a long-only basket earns
what the market gives: read the risk and cost columns, not the returns.
`turnover x/yr` is value traded / equity after the first purchase; `costs %/yr`
includes the first purchase but no final sale. Dividends are not counted.
