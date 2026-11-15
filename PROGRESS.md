# Progress log

Working notes for the self-paced loop. Newest first. The leaderboard has the numbers.

## Goal and stopping rule

Find forecasting models that beat the naive baseline **and** the previous ladder rung,
out of sample, then confirm the final set once on the lockbox. "Positive result" means
a lockbox win on **real** NSE data; synthetic wins only show the pipeline works.

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
- [x] Rung 4: N-BEATS (two sizes) and zero-shot Chronos, on synthetic and real data:
      none beats rung 3 at 30 min or rung 2 at next session.
- [x] Real data: one year of Kite 1-minute candles (10 large caps), development
      results in `results/run_kite/REPORT.md`.
- [ ] Lockbox: final set fixed below (round 6); opened once, after this entry.

## Blockers / needs from the owner

- None. The Kite access token expires every morning; `uv run forecasting kite-login`
  is only needed again before downloading new candles.

## Log

### 2026-11-15, round 6: real data
- **First run on real Kite data** (2025-11-12 -> 2026-11-12, 248 sessions, RELIANCE,
  HDFCBANK, ICICIBANK, INFY, TCS, BHARTIARTL, SBIN, ITC, LT, KOTAKBANK). Three things
  had to be fixed before the numbers meant anything:
  1. *Crash.* Chronos was not installed (it is in the `foundation` extra), returned NaN
     everywhere, and the common-sample rule left zero rows to score. Models that
     forecast < 50% of targets are now listed as "not scored" instead.
  2. *Trading hours changed.* Every session from 2026-08-03 ends at 15:15, not 15:30
     (closing auction session). Those days were counted as "partial", which threw
     away a third of the next-session targets in development and would have thrown
     away all of them in the lockbox. Sessions now use the window they all share,
     09:15-15:15 (72 bars); `close_changes` in `nse_calendar.yaml` records the date.
     Cost: 15:15-15:30 held ~2% of a day's variance before the change. `validate`
     now warns when the newest sessions all stop early.
  3. *Forecast floor.* Naive forecast exactly zero twice in 88,698 (ITC did not move
     for 30 minutes on 2026-08-10). With the old floor (1e-6 of the median target)
     those two rows took naive's 30-min QLIKE from 1.3 to 10.0, so every model
     looked like "0.04 x naive". The floor is now 1e-3 of the median target (about
     the smallest RV a one-tick move allows).
- **Development results** (148 scored sessions, 2026-02-06 -> 2026-09-11; QLIKE vs
  naive; vs best lower rung, DM p):
  - 30min: lgbm 0.329 (0.950 x garch, p=2e-13): rung 3 passes. chronos_tiny 0.378
    (1.149 x lgbm), nbeats_deep 0.379 (1.150 x), nbeats 0.397 (1.206 x): rung 4
    does NOT pass. combo_opt 0.329 (0.999 x lgbm, p=0.68): no gain.
  - next_session: arma_logrv_mix 0.627 (0.866 x seasonal_mean_5d, p=0.0004): rung 2
    passes. lgbm 0.675 (1.076 x arma, p=0.02): rung 3 does NOT pass. chronos_tiny
    0.790, nbeats_deep 0.828, nbeats 1.025: rung 4 does NOT pass. combo_equal 0.619
    (0.987 x arma, p=0.48): no gain.
  - Same ranking as on synthetic data. Common sample: 85,758 (30min) and 1,280
    (next session) forecasts per model; the only exclusions are the first 20 scored
    sessions, where nbeats_deep (60-day lookback) has no forecast yet.
- **Paper trading** (long at the open, flat at the close, vol targeting): every
  version loses 32-46% a year; costs alone are 24-30% of capital a year and the
  open-to-close drift of these stocks was negative. Better forecasts still give
  better risk control (std of return / forecast vol: har 1.01-1.02, lgbm 1.05,
  naive 1.21). Daily intraday round trips are not a way to earn from this.
- **Final set for the lockbox, fixed before opening it** (last 40 sessions,
  2026-09-14 -> 2026-11-12, all after the trading-hours change):
  - 30min: `lgbm`. Claim: QLIKE below naive and below the best rung-2 model.
  - next_session: `arma_logrv_mix`. Claim: QLIKE below naive and below the best
    rung-1 model.
  - A claim is confirmed if the ratio is < 1 with DM p < 0.05. Rungs 4 and 5 are
    scored too, but nothing is claimed for them.

### 2026-11-12, round 5
- Rung 4: N-BEATS (4 blocks, width 64, lookback 30d, QLIKE loss, pooled across symbols).
  Runs torch in a persistent subprocess (spawn) to avoid macOS OpenMP conflict with lgbm.
  Synthetic dev results (QLIKE vs naive; vs best of rung 3, DM p):
  - 30min: nbeats 0.506 (1.23 x lgbm 0.412, p=2e-137): rung 4 does NOT pass.
  - next_session: nbeats 1.036 (1.61 x arma 0.644, p=2e-21): worse than naive.
  - Default params under-fit on synthetic: 30d lookback may be too short, 100 epochs
    too few, or the architecture not expressive enough for intraday patterns.
  - Chronos (zero-shot amazon/chronos-t5-tiny) registered but not benchmarked yet;
    needs network access to download ~32 MB weights on first run.
- Fixed macOS libomp conflict: lgbm and torch cannot share OpenMP in one process.
  Solution: torch subprocess pool (ProcessPoolExecutor, spawn context); lgbm runs
  in the main process, torch runs in a fresh child process for every fold.
  brew install libomp done; KMP_DUPLICATE_LIB_OK=TRUE also set as belt-and-suspenders.
- Real-data run still pending (kite-login requires owner at the terminal).
  When ready: `uv sync --extra kite --extra deep` then the three run-all commands.
- Next: run on real data; tune nbeats (longer lookback, more epochs) if real data
  shows promise, or move to Chronos / PatchTST.

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
