"""The bar *panel*: the single in-memory data contract used by features, models and backtests.

A panel is a long DataFrame on a **balanced grid**: every (session, slot, symbol)
combination exists exactly once, missing bars are NaN rows with
``is_missing=True``, and rows are sorted by ``ts`` then ``symbol`` (symbol order
fixed). Because of that, "the last n bars of every symbol" is simply the last
``n * n_symbols`` rows, which makes walk-forward slicing O(1) and keeps
calendar arithmetic (same slot yesterday = ``bars_per_session`` bars ago) exact.

Columns:
    ts          bar start (tz-aware, Asia/Kolkata)
    bar_end     ts + bar length; the bar is known from this instant
    session     trading date (datetime.date)
    slot        bar index within the session, 0 .. bars_per_session-1
    symbol      instrument
    open high low close volume
    n_source    number of source candles aggregated into the bar (0 if missing)
    is_missing  True when the bar had no source candles

Metadata lives in ``DataFrame.attrs`` (preserved by pandas slicing):
    symbols, bars_per_session, bar_minutes
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PANEL_COLUMNS = (
    "ts",
    "bar_end",
    "session",
    "slot",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "n_source",
    "is_missing",
)


class PanelError(ValueError):
    """Raised when a frame does not satisfy the panel contract."""


def panel_symbols(df: pd.DataFrame) -> tuple[str, ...]:
    try:
        return tuple(df.attrs["symbols"])
    except KeyError as exc:
        raise PanelError("frame has no attrs['symbols']; build it with build_bars()") from exc


def n_symbols(df: pd.DataFrame) -> int:
    return len(panel_symbols(df))


def bars_per_session(df: pd.DataFrame) -> int:
    try:
        return int(df.attrs["bars_per_session"])
    except KeyError as exc:
        raise PanelError("frame has no attrs['bars_per_session']") from exc


def n_bars(df: pd.DataFrame) -> int:
    """Number of time steps (bar positions) in the panel."""
    return len(df) // n_symbols(df)


def to_matrix(df: pd.DataFrame, col: str) -> np.ndarray:
    """Column as an (n_bars, n_symbols) array; row t holds every symbol at bar t."""
    s = n_symbols(df)
    values = df[col].to_numpy()
    if len(values) % s:
        raise PanelError(f"{len(values)} rows is not a multiple of {s} symbols")
    return values.reshape(-1, s)


def tail_matrix(df: pd.DataFrame, col: str, n: int) -> np.ndarray:
    """The last ``n`` bars of ``col`` for every symbol, shape (<=n, n_symbols)."""
    s = n_symbols(df)
    values = df[col].to_numpy()
    take = min(n, len(values) // s) * s
    return values[len(values) - take :].reshape(-1, s)


def validate_panel(df: pd.DataFrame) -> None:
    """Check the balanced-grid contract. O(n); call once per backtest, not per origin."""
    missing_cols = [c for c in PANEL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise PanelError(f"panel missing columns {missing_cols}")
    symbols = panel_symbols(df)
    s, b = len(symbols), bars_per_session(df)
    if len(df) == 0 or len(df) % (s * b):
        raise PanelError(f"{len(df)} rows is not a whole number of sessions x {s} symbols x {b}")
    sym = df["symbol"].to_numpy().reshape(-1, s)
    if not (sym == np.asarray(symbols, dtype=object)).all():
        raise PanelError("symbol column does not repeat attrs['symbols'] in order")
    ts = df["ts"].to_numpy(dtype="datetime64[ns]").view("int64").reshape(-1, s)
    if not (ts == ts[:, :1]).all():
        raise PanelError("rows of one time step have different timestamps")
    if not (np.diff(ts[:, 0]) > 0).all():
        raise PanelError("timestamps are not strictly increasing across bars")
    slot = df["slot"].to_numpy().reshape(-1, s)[:, 0]
    if not (slot == np.tile(np.arange(b), len(slot) // b)).all():
        raise PanelError("slots are not 0..bars_per_session-1 for every session")
