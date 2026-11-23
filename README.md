# NSE intraday forecasting

Compare time series forecasting approaches on NSE candle data, using walk-forward
backtests and leakage-tested features. Every model is scored against a naive baseline.

This project **only reads** the Parquet candle export produced by the separate live
Kite pipeline. It never imports that pipeline's code and never writes to its export
directory (there is a test for that).

**First target:** realized volatility of 5-minute returns over the **next 30 minutes**
and the **next session**, per symbol.

## Model ladder

Each rung has to beat the one before it, out of sample, against naive.

| rung | models | status |
|---|---|---|
| 1. Baselines | random walk (naive), historical mean, seasonal naive (1 day, 5-day mean) | done |
| 2. Classical | seasonal EWMA, HAR-RV (+ intraday mix), ARMA on log RV, GARCH / GJR on de-seasonalized returns | done |
| 3. ML | LightGBM on realized measures, range vol, ATR, Bollinger, RSI, volume, time of day | done (wins at 30 min, not at next session) |
| 4. Deep / foundation | N-BEATS (two sizes); zero-shot Chronos | done (does not beat rung 3 at 30 min or rung 2 at next session) |
| 5. Combinations | equal and QLIKE-optimized geometric blends of rungs 2-3 | done (no significant gain) |

## Results on real data

One year of Kite 1-minute candles (12 Nov 2025 – 12 Nov 2026) for RELIANCE, HDFCBANK,
ICICIBANK, INFY, TCS, BHARTIARTL, SBIN, ITC, LT and KOTAKBANK. Models were chosen on
the development period; the last 40 sessions (15 Sep – 12 Nov 2026) were a lockbox,
opened once after the final set was written down.

| horizon | final model | QLIKE vs naive (lockbox) | vs best lower rung (lockbox) | verdict |
|---|---|---|---|---|
| next 30 min | `lgbm` (rung 3) | 0.299 (p = 1e-06) | 0.952 × `ewma_hl1` (p = 0.005) | confirmed |
| next session | `arma_logrv_mix` (rung 2) | 0.694 (p = 1e-07) | 0.840 × `seasonal_mean_5d` (p = 0.0009) | confirmed |

* Rung 3 does not beat rung 2 at the next-session horizon, rung 4 (N-BEATS, Chronos)
  beats neither, and combinations add nothing significant, in development and on
  the lockbox.
* Simulated intraday volatility targeting loses to costs (about 25–30% of capital a
  year) whatever the forecast; better forecasts only give better risk control.
* Details: [development](results/run_kite/REPORT.md),
  [lockbox](results/lockbox_kite/results.md), [log](PROGRESS.md).

### Using the forecasts: a held basket

`forecasting papertrade-delivery` holds the ten stocks overnight, sizes them with the
next-session forecasts toward 10% volatility a year (never more than fully invested),
and trades a stock only when its weight is more than 25% away from its target. With
`arma_logrv_mix`, simulated with Zerodha delivery charges:

| period | charges a year | realized vol (target 10%) | return a year | buy and hold: return, vol |
|---|---|---|---|---|
| development, 148 sessions | 1.0% | 10.6% | -17.7% | -21.1%, 15.0% |
| last 40 sessions | 1.0% | 11.7% | +44.8% | +67.5%, 15.1% |
| all 188 sessions | 1.0% | 11.0% | -4.4% | -2.2%, 15.2% |

* The cost problem is gone: about 1% a year (0.4% with a 50% band and about 3 orders
  a month) against 26% for the intraday version. Both checks written down before the
  last 40 sessions were run passed (charges under 2% a year, volatility within 0.7 to
  1.3 of the target).
* It is still a long-only basket. It earns what these stocks earn, scaled by how much
  is invested (about three quarters on average), and over these 188 sessions that was
  slightly negative. Volatility forecasts do not say which way prices go.
* Whether the forecasts beat a fixed position of the same size is open. In
  development they did (volatility 10.6% against 12.1% for sizing by long-run average
  volatility, worst day -2.2% against -2.8%); on the last 40 sessions they did not
  (11.7% against 11.5%).
* Details: [all models and periods](results/lockbox_kite/paper_trading_delivery.md),
  [band comparison](results/run_kite/paper_trading_delivery_bands.md).

### When to invest and when to stay out

Five textbook rules (200-day trend, a volatility filter, both, trend with a 2%
take-profit, trend with a 2% stop-loss), each on the 10 stocks and NIFTYBEES as
separate Rs 10 lakh accounts, after delivery charges and income tax:

| period | buy and hold (median yearly return) | best rule | rules beating buy and hold |
|---|---|---|---|
| 2007-2023 (development) | 11.9% | volatility filter, 6.3% | 1 of 50 stock/rule pairs; NIFTYBEES none |
| 2023-2026 (lockbox, checked once) | 8.1% | volatility filter, -1.2% | none |

Staying invested beat every rule. The rules made some falls shallower but missed
fast rebounds (2009, 2020) and paid charges and short-term tax on every round trip.
[Development](results/timing/development.md), [lockbox](results/timing/lockbox.md).

## Layout

```
forecasting/
  data/        read-only loader, NSE calendar, resampling to a bar panel, synthetic data
  features/    feature builders + lookahead-leakage harness
  models/      one module per model family, common fit/predict interface
  backtest/    walk-forward engine, targets, metrics, evaluation, reports
  notebooks/   exploration only (no logic results depend on)
  tests/       pytest suite, including leakage tests
  configs/     symbols, NSE calendar, data schema, experiment (horizons, models)
results/       markdown summaries of runs
```

## Real data

See [REAL_DATA.md](REAL_DATA.md): backfill 1-minute candles with your Kite Connect
historical API on your own machine, then one `run-all` command produces the report.

## The app (no commands needed)

Double-click **`NSE Forecasting.command`** in this folder (Finder). It opens a page
at http://127.0.0.1:8766 in your browser:

1. **Log in to Kite** (once a day; Kite logins end at 6 AM). You sign in on
   Zerodha's own page and come straight back.
2. **Download / update daily prices** for the 10 stocks and NIFTYBEES.
3. **When to invest, when to stay out**: results of the study below.

Everything runs on your Mac only. Close the Terminal window the launcher opens to
stop the app. (Same thing from a terminal: `uv run --extra kite forecasting ui`.)

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync                                   # create .venv and install
git config core.hooksPath .githooks       # once per clone: lint before every commit
uv run pytest                             # tests, including leakage tests

# try everything on synthetic data first
uv run forecasting synth --out data/synthetic
uv run forecasting backtest --data-root data/synthetic --out results/synthetic

# then on the real export
uv run forecasting validate --data-root /path/to/candle/export     # data-quality report
uv run forecasting backtest --data-root /path/to/candle/export     # -> results/rv_baselines/
```

`backtest` prints the results table and writes `results.md`, `metrics.csv`,
`metrics_per_symbol.csv`, `forecasts.parquet` and `run.json` (config and git revision).
For a quick run, add `--max-folds 5`, `--horizon 30min` or `--model naive`.

## Data contract

Two directory levels, symbol then date. Hive-style (`symbol=X/date=Y`, as written by
Spark) and plain directory names both work:

```
<root>/symbol=RELIANCE/date=2025-01-02/part-0000.parquet
<root>/RELIANCE/2025-01-02/candles.parquet
```

Each file needs a timestamp plus OHLCV columns. Extra columns, such as latency
instrumentation, are ignored. If your column names, timezone or timestamp convention
differ, edit `forecasting/configs/data.yaml`:

* `columns.timestamp`: Kite's historical API calls it `date`, which is fine.
* `timestamp_timezone`: applied only to tz-naive stamps. tz-aware and integer
  epoch stamps are converted.
* `timestamp_label`: whether a candle's stamp marks its `start` or its `end`.
* `source_interval_minutes`: the export's candle length (1 by default).

Run `forecasting validate` first. It reports duplicates, rows outside market hours,
missing bars and sessions per symbol. It also flags any disagreement between your
data and `configs/nse_calendar.yaml`: data on a "holiday", or a session where no
symbol has data. **Check the calendar against NSE's circulars, especially for 2026.**

## Definitions

* **Bars:** 5-minute bars aligned to the open, 09:15–15:30 IST, giving 75 per session.
  Pre-open, the closing session, holidays and special sessions (such as Muhurat) are
  dropped and counted.
* **Trading-hours change.** From 3 Aug 2026 continuous trading ends at 15:15 (the
  closing auction session took over the last 15 minutes); see `close_changes` in
  `configs/nse_calendar.yaml`. Data that spans the change is cut to the window every
  session shares, 09:15–15:15 (72 bars), so "the next session's RV" means the same
  thing before and after. `validate` says so in its report.
* **Gaps:** a bar with no source candles stays NaN (`is_missing`). Gaps are never
  forward-filled, and a return is never computed across a gap.
* **Returns:** `100 * log(close_t / close_{t-1})` within a session. The first bar uses
  `close / open`, so the overnight gap is excluded and RV measures intraday volatility.
* **Target:** realized variance (RV, in %²), the sum of squared returns over the next
  `h` bars. A target window containing a missing bar is not scored.
* **Horizons** (`configs/rv_baselines.yaml`):
  * `30min`: `h = 6`, with an origin at every bar boundary that leaves 30 minutes
    in the session.
  * `next_session`: every bar of the session (`h = 75`, or 72 on the 09:15–15:15
    window), with one origin per session before the open.

## Evaluation rules

* **Walk-forward only.** The first 60 sessions are used for training. After that,
  models are refit every 5 sessions with an expanding window (rolling is also
  supported). Test blocks never overlap.
* **No lookahead:**
  * `fit` only sees bars before the fold's first test session.
  * `predict` only sees bars that have ended by the origin.
  * Targets are computed by the engine and never passed to a model.
  * Tests check this with:
    * a spy model;
    * future poisoning: scrambling every bar after time T must not change any
      forecast made at or before T;
    * `assert_no_lookahead`, which every feature builder must pass. The harness is
      itself tested against deliberately leaky features (`shift(-1)`, full-sample
      z-score, centred windows).
* **Common sample.** Every model is scored on the same (origin, symbol) pairs. A pair
  counts only if its target is observed and every scored model produced a forecast.
  A model that forecasts less than half of the targets (for example because an
  optional dependency is missing) is listed as "not scored" instead of shrinking the
  sample for everyone.
* **Forecast floor.** A forecast below 0.1% of the median target is raised to that
  level before scoring, and counted (`floored`). No real window is that quiet (it is
  about one tick of movement), and without the floor a single zero forecast, which
  has infinite QLIKE, decides the whole comparison.
* **Metrics vs naive.**
  * QLIKE and MSE are computed on variance (Patton 2011).
  * MASE is MAE divided by the naive model's MAE on the same out-of-sample origins,
    so naive scores exactly 1.
  * Ratios to naive are reported for each metric.
  * Diebold-Mariano on QLIKE uses the per-origin cross-sectional mean loss and
    Newey-West errors, with a lag at least as long as the target-window overlap.
  * Directional accuracy is implemented for the later return targets.

## Adding a model

```python
from forecasting.models.base import ForecastModel


class MyModel(ForecastModel):
    def fit(self, train_df):  # panel slice: bars before the test block
        ...
        return self

    def predict(self, history_df, horizon):  # panel slice: bars ended by the origin
        yhat = ...  # RV forecast per symbol, panel symbol order
        return self.forecast_frame(history_df, yhat)
```

Register it in `forecasting/models/__init__.py` and list it under `models:` in the
experiment YAML. The panel is a balanced grid (see `forecasting/data/panel.py`), so
`tail_matrix(history_df, "ret2", n)` returns the last `n` bars of every symbol cheaply.

## Results

* [`results/leaderboard.md`](results/leaderboard.md): every model configuration ever
  run, losers included, per data set / split / horizon.
* [`results/synthetic_ladder/results.md`](results/synthetic_ladder/results.md): the full
  ladder on **synthetic** data (development split). This checks the pipeline and the
  models on data with known structure. It says nothing about real NSE stocks.
* Real-data results: `uv run forecasting backtest --config forecasting/configs/rv_ladder.yaml
  --data-root <export>`.

### The lockbox

The last `holdout_sessions` (40, about two months) are excluded from every development
run. Iterate on models using the development split only; when a set of models is
final, score the lockbox once with `backtest --final`. Each opening is logged in
`results/lockbox_log.csv`, and repeat openings print a warning, because every extra
look turns the lockbox into another development set.

## Checking from a phone

Every push runs the GitHub Actions workflow in `.github/workflows/ci.yml`: lint, the
full test suite and a synthetic end-to-end backtest. To see it, open the repo in the
GitHub app or on github.com, go to **Actions**, and open the latest run. A green tick
means everything passed, and the run's **Summary** page shows the results table.

That build does not install torch, so the rung 4 tests (N-BEATS, Chronos) are skipped
in it. A second workflow, `.github/workflows/deep.yml` ("deep-models"), installs torch
and Chronos and runs everything on macOS, where lightgbm and torch can clash. It runs
every Monday, on pushes that change the models or dependencies, and by hand (**Actions**
-> **deep-models** -> **Run workflow**). It fails if any test is skipped or any model
produces no forecasts.

The same lint checks run before every commit once the hook is on
(`git config core.hooksPath .githooks`, once per clone). It stops a commit that the
online build would reject for formatting; `uv run ruff format .` fixes most of those.
