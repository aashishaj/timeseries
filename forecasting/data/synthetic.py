"""Synthetic 1-minute candles in the same Parquet layout as the real export.

Used for tests, CI and demos when the real export is not at hand. The generator has
known structure, so it doubles as a sanity check of the pipeline:

* persistent daily volatility (market factor + idiosyncratic, log-AR(1));
* an intraday U-shape (busy open, quiet lunch, pick-up into the close);
* slowly varying intraday volatility, Student-t returns, tick-size rounding;
* realistic dirt: missing minutes, multi-minute gaps, whole-day outages,
  duplicates, pre-open/closing-session rows and Muhurat rows on a holiday.

Results on this data say nothing about real NSE stocks.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from forecasting.data.calendar import NSECalendar

MARKER = "_SYNTHETIC"
TICK = 0.05


def _ar1(rng: np.random.Generator, shape: tuple[int, ...], phi: float, sd: float) -> np.ndarray:
    """AR(1) along the last axis, started from its stationary distribution."""
    eps = rng.normal(0.0, sd, shape)
    x = np.empty(shape)
    x[..., 0] = eps[..., 0] / np.sqrt(1 - phi**2)
    for t in range(1, shape[-1]):
        x[..., t] = phi * x[..., t - 1] + eps[..., t]
    return x


def intraday_shape(minutes: int) -> np.ndarray:
    u = np.arange(minutes)
    shape = 0.75 + 2.2 * np.exp(-u / 12.0) + 0.7 * np.exp(-(minutes - 1 - u) / 20.0)
    return shape / shape.mean()


def generate_candles(
    symbols: list[str],
    calendar: NSECalendar,
    start: dt.date | str,
    end: dt.date | str,
    *,
    seed: int = 7,
    missing_minute_prob: float = 0.002,
    gap_blocks_per_symbol: int = 6,
    outage_days_per_symbol: int = 2,
    dirty: bool = True,
) -> pd.DataFrame:
    """Return 1-minute candles: timestamp (naive IST, candle start), OHLCV, extra columns."""
    rng = np.random.default_rng(seed)
    sessions = calendar.sessions(start, end)
    n_days, n_min, n_sub = len(sessions), calendar.session_minutes, 4
    shape = intraday_shape(n_min)
    market = _ar1(rng, (n_days,), 0.97, 0.06)
    open_minute = calendar.open_time.hour * 60 + calendar.open_time.minute
    day_ns = np.array([np.datetime64(d, "m") for d in sessions])
    stamps = day_ns[:, None] + np.arange(open_minute, open_minute + n_min).astype("timedelta64[m]")

    frames = []
    for symbol in symbols:
        daily_vol = rng.uniform(1.1, 2.0)  # percent per day
        log_v = 2 * np.log(daily_vol) + 2 * market + 2 * _ar1(rng, (n_days,), 0.85, 0.07)
        intraday = _ar1(rng, (n_days, n_min), 0.985, 0.4 * np.sqrt(1 - 0.985**2))
        var_min = np.exp(log_v)[:, None] / n_min * shape * np.exp(intraday - 0.08)  # %^2
        z = rng.standard_t(5, (n_days, n_min, n_sub)) / np.sqrt(5 / 3)
        steps = z * np.sqrt(var_min[..., None] / n_sub) / 100.0
        steps[:, 0, 0] += rng.normal(0.0, 0.4 * np.exp(log_v / 2) / 100.0)  # overnight gap
        log_p = np.log(rng.uniform(300, 3000)) + np.cumsum(steps.ravel()).reshape(steps.shape)
        price = np.round(np.exp(log_p) / TICK) * TICK
        volume = (
            rng.lognormal(np.log(rng.uniform(5e3, 5e4)), 0.5, (n_days, n_min))
            * shape**0.8
            * np.exp(intraday / 2)
        ).astype(np.int64) + 1
        frames.append(
            pd.DataFrame(
                {
                    "timestamp": stamps.ravel().astype("datetime64[us]"),
                    "open": price[..., 0].ravel(),
                    "high": price.max(axis=2).ravel(),
                    "low": price.min(axis=2).ravel(),
                    "close": price[..., -1].ravel(),
                    "volume": volume.ravel(),
                    "symbol": symbol,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    if dirty:
        df = _add_dirt(
            df,
            rng,
            calendar,
            sessions,
            missing_minute_prob,
            gap_blocks_per_symbol,
            outage_days_per_symbol,
        )
    df["latency_ms"] = rng.gamma(2.0, 20.0, len(df))  # mimics the live pipeline's extra columns
    df["date"] = df["timestamp"].dt.strftime("%Y-%m-%d")
    return df.sort_values(["symbol", "timestamp"], kind="stable").reset_index(drop=True)


def _add_dirt(
    df: pd.DataFrame,
    rng: np.random.Generator,
    calendar: NSECalendar,
    sessions: list[dt.date],
    missing_minute_prob: float,
    gap_blocks: int,
    outage_days: int,
) -> pd.DataFrame:
    day = df["timestamp"].dt.date
    minute = (df["timestamp"] - df["timestamp"].dt.normalize()).dt.total_seconds() // 60
    drop = rng.random(len(df)) < missing_minute_prob
    for symbol in df["symbol"].unique():
        is_sym = (df["symbol"] == symbol).to_numpy()
        for d in rng.choice(len(sessions), size=min(outage_days, len(sessions)), replace=False):
            drop |= is_sym & (day == sessions[d]).to_numpy()
        for _ in range(gap_blocks):
            d = sessions[rng.integers(len(sessions))]
            first = 555 + rng.integers(0, 330)
            span = rng.integers(10, 45)
            drop |= (
                is_sym
                & (day == d).to_numpy()
                & (minute >= first).to_numpy()
                & (minute < first + span).to_numpy()
            )
    df = df.loc[~drop]

    extras = [df.sample(frac=0.0005, random_state=int(rng.integers(1 << 31)))]  # duplicates
    firsts = df.groupby(["symbol", day.loc[df.index]]).head(1)
    for offset in range(-15, -7):  # pre-open 09:00-09:07
        extras.append(firsts.assign(timestamp=firsts["timestamp"] + pd.Timedelta(minutes=offset)))
    lasts = df.groupby(["symbol", day.loc[df.index]]).tail(1)
    for offset in range(1, 5):  # closing session 15:30-15:33
        extras.append(lasts.assign(timestamp=lasts["timestamp"] + pd.Timedelta(minutes=offset)))
    lo, hi = sessions[0], sessions[-1]
    for special in sorted(calendar.special_sessions):
        if lo <= special <= hi:
            ref = lasts[lasts["timestamp"].dt.date < special].groupby("symbol").tail(1)
            base = pd.Timestamp(special) + pd.Timedelta(hours=18)
            for m in range(60):
                extras.append(ref.assign(timestamp=base + pd.Timedelta(minutes=m)))
    return pd.concat([df, *extras], ignore_index=True)


def write_export(df: pd.DataFrame, root: str | Path) -> Path:
    """Write candles as ``root/symbol=X/date=YYYY-MM-DD/*.parquet``.

    Refuses to touch a non-empty directory that was not created by this function,
    so it can never overwrite a real export.
    """
    root = Path(root)
    if root.exists() and any(root.iterdir()) and not (root / MARKER).exists():
        raise FileExistsError(
            f"{root} is not empty and is not a synthetic export; refusing to write there"
        )
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text("synthetic data generated by forecasting.data.synthetic\n")
    # One symbol at a time keeps open partition files (~250/yr) far below pyarrow's
    # max_open_files; a single call over every partition can stall at exit.
    for _, part in df.groupby("symbol", sort=True):
        pq.write_to_dataset(
            pa.Table.from_pandas(part, preserve_index=False),
            root_path=str(root),
            partition_cols=["symbol", "date"],
            existing_data_behavior="delete_matching",
        )
    return root
