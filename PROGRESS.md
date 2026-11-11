# Progress log

Working notes for the self-paced loop. Newest first. The leaderboard has the numbers.

## Goal and stopping rule

Find forecasting models that beat the naive baseline **and** the previous ladder rung,
out of sample, then confirm the final set once on the lockbox. "Positive result" means
a lockbox win on **real** NSE data; synthetic wins only show the pipeline works.

## Blockers / needs from the owner

- **Real data.** The Kite export is not reachable from the cloud session, and network
  access to Yahoo Finance was declined. Until one of those changes, everything runs on
  synthetic data. To get real results, run on a machine with the export:
  `uv run forecasting backtest --config forecasting/configs/rv_ladder.yaml --data-root <export>`.
- Paper trading needs real prices for the results to mean anything; it will be built
  and tested on synthetic data meanwhile.

## Ladder status

- [x] Rung 1: naive, historical mean, seasonal naive (1d), seasonal mean (5d)
- [x] Rung 2a: seasonal EWMA, HAR-RV (per-symbol, pooled), HAR mix (daily x
      time-of-day x intraday correction)
- [ ] Rung 2b: GARCH / GJR on seasonally filtered returns, AR(FI)MA on log RV
- [ ] Compare each rung against the best of the previous rung (DM vs that model, not
      only vs naive)
- [ ] Rung 3: indicators (ATR, range-based vol, Bollinger width, volume, RSI) + LightGBM
- [ ] Ensembles with weights from past forecast errors only
- [ ] Simulated paper trading with NSE costs (vol targeting)
- [ ] Rung 4: deep / foundation models (CPU only; Chronos weights need network access)

## Log

### 2026-11-11, round 1
- Added the lockbox (last 40 sessions) and the trial leaderboard.
- Synthetic data v2: added price jumps, surprise news days and a Tuesday expiry bump,
  so persistence-based models do not win by construction.
- Rung 2a on synthetic dev data:
  - 30min: `har_mix` 0.474 x naive QLIKE, `ewma_hl1` 0.486, best baseline 0.637.
  - next_session: `har_pooled` 0.678, best baseline (`seasonal_mean_5d`) 0.700.
- Next: GARCH family + ARIMA on log RV; add "vs previous rung" comparison.
