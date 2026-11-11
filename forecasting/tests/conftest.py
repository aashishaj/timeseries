from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from forecasting.data.calendar import NSECalendar
from forecasting.data.panel import PANEL_COLUMNS
from forecasting.data.synthetic import generate_candles, write_export
from forecasting.features.returns import add_returns


@pytest.fixture(scope="session")
def calendar() -> NSECalendar:
    return NSECalendar.from_yaml()


def make_toy_panel(
    n_sessions: int = 12,
    symbols: tuple[str, ...] = ("AAA", "BBB"),
    bars_per_session: int = 6,
    seed: int = 0,
    missing: list[tuple[int, int, str]] | None = None,
) -> pd.DataFrame:
    """A small valid panel with random prices.

    ``missing`` lists (session_index, slot, symbol) bars to blank out.
    """
    rng = np.random.default_rng(seed)
    s, b = len(symbols), bars_per_session
    days = pd.bdate_range("2025-01-06", periods=n_sessions)
    n = n_sessions * b * s
    session = np.repeat([d.date() for d in days], b * s)
    slot = np.tile(np.repeat(np.arange(b), s), n_sessions)
    ts = pd.DatetimeIndex(
        np.repeat(days.to_numpy(), b * s)
        + np.timedelta64(9 * 60 + 15, "m")
        + slot * np.timedelta64(5, "m")
    ).tz_localize("Asia/Kolkata")
    log_p = np.log(1000) + np.cumsum(rng.normal(0, 0.002, (n // s, s)), axis=0)
    close = np.exp(log_p).ravel()
    open_ = close * np.exp(rng.normal(0, 0.001, n))
    panel = pd.DataFrame(
        {
            "ts": ts,
            "bar_end": ts + pd.Timedelta(minutes=5),
            "session": session,
            "slot": slot,
            "symbol": np.tile(np.asarray(symbols, dtype=object), n_sessions * b),
            "open": open_,
            "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999,
            "close": close,
            "volume": rng.integers(100, 1000, n).astype(float),
            "n_source": 5,
            "is_missing": False,
        }
    )[list(PANEL_COLUMNS)]
    for sess_i, sl, sym in missing or []:
        row = (sess_i * b + sl) * s + symbols.index(sym)
        panel.loc[row, ["open", "high", "low", "close", "volume"]] = np.nan
        panel.loc[row, "n_source"] = 0
        panel.loc[row, "is_missing"] = True
    panel.attrs = {"symbols": tuple(symbols), "bars_per_session": b, "bar_minutes": 5}
    return add_returns(panel)


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    return make_toy_panel()


@pytest.fixture(scope="session")
def synthetic_export(tmp_path_factory, calendar) -> dict:
    """A small dirty synthetic export on disk (2 symbols, ~2.5 months)."""
    root = tmp_path_factory.mktemp("export")
    symbols = ["RELIANCE", "INFY"]
    start, end = dt.date(2025, 1, 1), dt.date(2025, 3, 20)
    df = generate_candles(symbols, calendar, start, end, seed=3)
    write_export(df, root)
    return {"root": root, "symbols": symbols, "start": start, "end": end, "candles": df}
