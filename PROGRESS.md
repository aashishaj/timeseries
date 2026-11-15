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
- [x] Lockbox (opened once, 2026-11-15, at commit 2bc461f): both claims confirmed on
      real data. `lgbm` at 30 min, `arma_logrv_mix` at next session.
- [x] Held, volatility-managed basket with banded rebalancing (2026-11-16): charges
      about 1% a year instead of 26%; risk near target; no profit from it.

## Blockers / needs from the owner

- None. The Kite access token expires every morning; `uv run forecasting kite-login`
  is only needed again before downloading new candles.

## Log

### 2026-11-16, round 8: a held basket instead of daily round trips
- New `forecasting papertrade-delivery` (`forecasting/trading/vol_managed.py`): the
  stocks are held overnight (delivery), sized by the next-session forecasts, and a
  stock is traded only when its weight is more than a band away from its target.
  - Forecast intraday RV -> close-to-close volatility with each stock's past ratio of
    close-to-close variance to intraday RV (about 1.1 to 1.8 here).
  - Inverse-volatility weights; basket volatility from those volatilities and the
    correlation of the last 60 sessions (shrunk halfway to its average); scaled to a
    10% a year target; never more than 1x invested, the rest in cash at 0%.
  - Whole shares, trades at the session's first price, Zerodha delivery charges
    (`configs/costs_delivery.yaml`: no brokerage, STT 0.1% each side, stamp duty
    0.015%, Rs 15.34 depository charge per stock sold) + 2 bps slippage per side.
  - Kite prices are split-adjusted: the largest overnight move in the year is 9%.
- **Development sessions** (148, 2026-02-06 -> 2026-09-11; the basket fell about 21%
  a year annualized over them). `results/run_kite/paper_trading_delivery.md`:
  - Costs: `arma_logrv_mix` with the 25% band trades 4.8x capital a year, about 18
    orders a month, for 1.0% a year in charges. The intraday version paid 26.6%.
  - Risk: realized volatility 10.6% against the 10% target; std of return / forecast
    volatility 1.12.
  - What the forecasts add: `hist_mean` (long-run average volatility, so an almost
    constant position) was invested the same on average (0.74 vs 0.73) but ran at
    12.1% volatility, worst day -2.8% (arma -2.2%), max drawdown -14.6% (arma
    -12.3%). The forecasts cost 0.7% a year more in charges.
  - Returns: all negative because the market fell. Sharpe is no better than buy and
    hold (-1.67 vs -1.40). There is no evidence here that sizing by volatility earns
    more; it only steadies the risk.
  - Band (`paper_trading_delivery_bands.md`, arma): 0% -> 3.9% a year in costs, 10% ->
    2.2%, 25% -> 1.0%, 50% -> 0.5%, with realized volatility between 10.4% and 10.9%
    for all four. Default 25%; 50% is the low-effort option (3 orders a month).
  - `nbeats_deep` has no forecast for its first 20 sessions and sat in cash while the
    market fell; its smaller loss is that, not skill. The report now flags it.
- **Check on the 40 lockbox sessions, fixed before running it.** These sessions were
  not used to design the strategy (the intraday version's results on them are known).
  The basket is run straight through from the development sessions, positions carried
  over. Model `arma_logrv_mix`, 10% target, 1x cap, 25% band; the 50% band and
  `hist_mean` are reported next to it. Pass if, over those 40 sessions:
  1. charges are below 2% a year, and
  2. realized volatility is within 0.7 to 1.3 x the target (40 sessions pin a
     volatility down to about +-11%, one standard error).
  Nothing is claimed about returns: 40 sessions cannot show them.
- **Result on those 40 sessions** (2026-09-15 -> 2026-11-12; the basket rose about 11%
  in them). `results/lockbox_kite/paper_trading_delivery.md`:
  - Check 1 passed: charges 0.98% a year (5.5x turnover, about 20 orders a month).
  - Check 2 passed: realized volatility 11.7%, 1.17 x the target.
  - 50% band: charges 0.07% a year, volatility 10.6%, about 1 order a month.
  - Not repeated: the gain over `hist_mean`. On these sessions `hist_mean` ran at 11.5%
    volatility with 0.73 invested, arma at 11.7% with 0.76. So whether the forecasts
    beat a fixed position of the same size is open (yes on 148 sessions, no on 40).
  - Returns, for the record only: arma +44.8% a year annualized, buy and hold +67.5%.
- All 188 sessions: arma -4.4% a year at 11.0% volatility, max drawdown -14.6%; buy
  and hold -2.2% at 15.2%, max drawdown -16.8%. The basket is long only, about three
  quarters invested, and earns what the stocks earn. The forecasts fix the cost
  problem and hold risk near a chosen level; they are not a source of profit.
- Realized volatility runs 6-17% above target and std of return / forecast volatility
  is about 1.1: days with large overnight gaps are fatter-tailed than the forecasts
  imply, and correlations on those days are higher than the trailing estimate.
- Open: earning from a volatility forecast needs an instrument that pays on
  volatility (options against implied volatility). That needs option-chain history,
  which the Kite historical API does not keep for expired contracts.

### 2026-11-15, round 7: lockbox
- Opened once, with the code and final set of commit 2bc461f: 40 sessions
  (2026-09-15 -> 2026-11-12), every model refit every 5 sessions on all earlier data.
  26,800 forecasts per model at 30 min, 400 at next session, nothing excluded.
  `results/lockbox_kite/results.md`.
- **Claim 1, 30min, `lgbm`: confirmed.** QLIKE 0.299 x naive (DM p=1e-06, better on
  10/10 symbols). Against rung 2: 0.946 x garch (the comparator picked on development
  data, p=4e-08) and 0.952 x ewma_hl1 (the best rung-2 model on the lockbox itself,
  p=0.005).
- **Claim 2, next_session, `arma_logrv_mix`: confirmed.** QLIKE 0.694 x naive
  (p=1e-07, 10/10 symbols) and 0.840 x seasonal_mean_5d, the best rung-1 model
  (p=0.0009).
- Not claimed, for the record:
  - Rung 3 at next session: lgbm 0.663 x naive, 0.956 x arma (p=0.35) and 0.995 x
    har_pooled (p=0.91): still no significant gain over rung 2.
  - Rung 4 fails on the lockbox too. 30min: chronos_tiny 1.18 x lgbm, nbeats_deep
    1.18 x, nbeats 1.25 x. Next session: chronos_tiny 0.99 x naive (no better than
    naive), nbeats_deep 1.17 x naive, nbeats 1.42 x naive (worse than naive).
  - Combinations: combo_opt 0.928 x lgbm at next session (p=0.18), 1.000 x at 30min:
    not significant.
- Size of the win in plain terms (lockbox): at 30 min lgbm's median volatility error
  is 31% (naive 40%) and 42% of its forecasts land within +-25% of what happened
  (naive 33%). At next session arma: 18% median error (naive 20%), 63% within +-25%
  (naive 57%).
- Paper trading on the lockbox (40 sessions, too few to read returns): every version
  loses 24-33% a year annualized, costs 25-30% a year. Risk control is again better
  with the better forecasts (std of return / forecast vol: combo_opt 1.07, har_pooled 1.08,
  arma 1.11, naive 1.18, nbeats 1.30).
- **This is the project's positive result:** on real NSE data, out of sample and on
  a lockbox used once, a boosted-tree model with volatility indicators forecasts
  30-minute realized variance better than every simpler model, and ARMA on log daily
  RV forecasts next-session variance better than the baselines. Deep and foundation
  models (at this size, with one year of data) do not help. The forecasts are useful
  for risk sizing, not as a source of intraday profit after costs.
- The lockbox is now spent for this data. New claims need new data (later sessions
  from `backfill-kite`), with a new lockbox.

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
  2026-09-15 -> 2026-11-12, all after the trading-hours change):
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
