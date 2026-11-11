# Progress log

Working notes for the self-paced loop. Newest first. The leaderboard has the numbers.

## Goal and stopping rule

Find forecasting models that beat the naive baseline **and** the previous ladder rung,
out of sample, then confirm the final set once on the lockbox. "Positive result" means
a lockbox win on **real** NSE data; synthetic wins only show the pipeline works.

## Blockers / needs from the owner

- **Real data.** There is no candle export yet. Plan: backfill 1-minute candles with the
  owner's paid Kite Connect historical API on their Mac, then `run-all` (see
  REAL_DATA.md). Credentials stay on the Mac.

## Ladder status

- [x] Rung 1: naive, historical mean, seasonal naive (1d), seasonal mean (5d)
- [x] Rung 2a: seasonal EWMA, HAR-RV (per-symbol, pooled), HAR mix (daily x
      time-of-day x intraday correction)
- [x] Rung 2b: GARCH / GJR on seasonally filtered returns, ARMA(1,1) on log daily RV
- [x] Compare each rung against the best of the previous rung (ratio + DM p-value)
- [x] Rung 3: indicators (ATR, range-based vol, Bollinger width, volume, RSI) + LightGBM
- [x] Ensembles (equal and QLIKE-optimized geometric combinations): no significant gain
- [x] Simulated paper trading with NSE costs (vol targeting)
- [x] Real-data tooling: `kite-login`, `backfill-kite`, `run-all`, REAL_DATA.md
- [ ] Rung 4: deep / foundation models (CPU only; Chronos weights need network access)

## Log

### 2026-11-11, round 4
- Combinations (members: lgbm, garch, arma_logrv_mix, har_mix), synthetic dev:
  - 30min: combo_opt 0.412 = lgbm (1.00x, p=0.99); combo_equal 1.06x lgbm (worse).
  - next_session: both combos 0.638 vs arma 0.642 (0.99x, p=0.65/0.87): not significant.
  - Combinations do not pass the ladder rule here.
- Paper trading (vol targeting, intraday MIS, Rs 10 lakh):
  - Daily open-to-close round trips cost ~25-30% of capital per year (STT + slippage
    dominate): any intraday strategy needs >0.1%/day edge just to break even.
  - Better forecasts = modestly better risk control (std of return / forecast vol
    1.03-1.04 vs 1.11 for naive; worst day -0.87% vs -1.00% naive, -1.46% constant).
  - Sizing assumes perfectly correlated names, so realized vol is ~1/3 of target.
    Next: lower-turnover vol-managed basket (delivery, banded rebalancing) on real data.
- Built real-data tooling (Kite backfill/login, run-all). Loop paused until real data.

### 2026-11-11, round 3
- Indicator library (18 finite-window indicators, all lookahead-tested) and a pooled
  LightGBM model: HAR-like anchor x boosted correction, gamma objective (= QLIKE).
- Synthetic dev results (QLIKE vs naive; vs best of previous rung, DM p):
  - 30min: lgbm 0.412 (0.89 x garch, p~1e-100): rung 3 passes.
  - next_session: lgbm 0.663 (1.03 x arma_logrv_mix, p=0.21): rung 3 does NOT pass.
    Only ~1-2k daily training rows; not tuned further to avoid fitting the dev split.
- Next: forecast combinations (best daily + best intraday models), weights from past
  errors only.

### 2026-11-11, round 2
- GARCH(1,1) / GJR on de-seasonalized returns; ARMA(1,1) on log daily RV (with the
  same time-of-day + intraday-correction wrapper as HAR); previous-rung comparison.
- Synthetic dev results (QLIKE vs naive; vs best of rung 1, DM p):
  - 30min: garch 0.463 (0.727, p~1e-82), arma_logrv_mix 0.469, har_mix 0.474.
  - next_session: arma_logrv_mix 0.645 (0.922, p=0.0002); har_pooled 0.678 (0.969,
    p=0.21, not significant); garch 0.814 (worse than rung 1: intraday GARCH has no
    day-to-day memory).
- Rung 2 passes the ladder rule on both horizons. Next: rung 3 (indicators + LightGBM).

### 2026-11-11, round 1
- Added the lockbox (last 40 sessions) and the trial leaderboard.
- Synthetic data v2: added price jumps, surprise news days and a Tuesday expiry bump,
  so persistence-based models do not win by construction.
- Rung 2a on synthetic dev data:
  - 30min: `har_mix` 0.474 x naive QLIKE, `ewma_hl1` 0.486, best baseline 0.637.
  - next_session: `har_pooled` 0.678, best baseline (`seasonal_mean_5d`) 0.700.
- Next: GARCH family + ARIMA on log RV; add "vs previous rung" comparison.
