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
| 2. Classical | seasonal EWMA, HAR-RV, HAR x time-of-day x intraday mix (done); GARCH family, ARIMA (next) | in progress |
| 3. ML | LightGBM on lagged returns / RV / volume / time of day | |
| 4. Deep / foundation | N-BEATS or PatchTST; zero-shot Chronos / TimesFM | |

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

## Quick start

Needs [uv](https://docs.astral.sh/uv/) and Python 3.11+.

```bash
uv sync                                   # create .venv and install
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
* **Gaps:** a bar with no source candles stays NaN (`is_missing`). Gaps are never
  forward-filled, and a return is never computed across a gap.
* **Returns:** `100 * log(close_t / close_{t-1})` within a session. The first bar uses
  `close / open`, so the overnight gap is excluded and RV measures intraday volatility.
* **Target:** realized variance (RV, in %²), the sum of squared returns over the next
  `h` bars. A target window containing a missing bar is not scored.
* **Horizons** (`configs/rv_baselines.yaml`):
  * `30min`: `h = 6`, with an origin at every bar boundary that leaves 30 minutes
    in the session.
  * `next_session`: `h = 75`, with one origin per session before the open.

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
  counts only if its target is observed and every model produced a forecast.
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
