import datetime as dt

import numpy as np
import pandas as pd
import pytest

from forecasting.data.loader import load_candles
from forecasting.data.panel import validate_panel
from forecasting.data.resample import build_bars

IST = "Asia/Kolkata"


def _candles(symbol, day, minutes, price=100.0):
    """1-min candles at the given minutes after 09:15 on ``day`` with increasing prices."""
    start = pd.Timestamp(f"{day} 09:15", tz=IST)
    ts = [start + pd.Timedelta(minutes=int(m)) for m in minutes]
    close = price + np.arange(len(ts), dtype=float)
    return pd.DataFrame(
        {
            "symbol": symbol,
            "ts": pd.DatetimeIndex(ts),
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 10.0,
        }
    )


def test_aggregates_ohlcv_on_session_grid(calendar):
    raw = _candles("A", "2025-01-02", range(375))
    panel, report = build_bars(raw, calendar)
    validate_panel(panel)
    assert len(panel) == 75 and not panel["is_missing"].any()
    first = panel.iloc[0]
    assert first["ts"] == pd.Timestamp("2025-01-02 09:15", tz=IST)
    assert first["bar_end"] == pd.Timestamp("2025-01-02 09:20", tz=IST)
    assert (first["open"], first["high"], first["low"], first["close"]) == (
        99.5,
        105.0,
        99.0,
        104.0,
    )
    assert first["volume"] == 50 and first["n_source"] == 5
    assert panel.iloc[-1]["ts"] == pd.Timestamp("2025-01-02 15:25", tz=IST)
    assert report.rows_used == 375


def test_drops_and_counts_dirty_rows(calendar):
    day = _candles("A", "2025-01-02", range(375))
    pre_open = _candles("A", "2025-01-02", range(-15, -7))
    post_close = _candles("A", "2025-01-02", range(375, 380))
    holiday = _candles("A", "2025-03-14", range(10))  # Holi: calendar says closed
    muhurat = _candles("A", "2025-10-21", range(10))  # known special session
    dup = day.iloc[[4]].assign(close=104.5)  # re-emitted last candle of bar 0; later copy wins
    raw = pd.concat([day, pre_open, post_close, holiday, muhurat, dup], ignore_index=True)
    panel, report = build_bars(raw, calendar)
    assert report.rows_outside_hours == 8 + 5
    assert report.rows_non_session_date == 20
    assert report.unexpected_dates == [dt.date(2025, 3, 14)]
    assert report.special_session_dates == [dt.date(2025, 10, 21)]
    assert report.duplicates_dropped == 1
    assert panel.iloc[0]["close"] == 104.5


def test_missing_bars_are_nan_rows_not_filled(calendar):
    minutes = [m for m in range(375) if not 10 <= m < 15]  # bar 2 (09:25-09:30) absent
    panel, _ = build_bars(_candles("A", "2025-01-02", minutes), calendar)
    assert panel["is_missing"].tolist()[:4] == [False, False, True, False]
    assert np.isnan(panel.iloc[2]["close"]) and panel.iloc[2]["n_source"] == 0


def test_balanced_grid_with_symbol_outage_and_trimmed_range(calendar):
    a = pd.concat(
        [_candles("A", d, range(375)) for d in ["2025-01-02", "2025-01-03", "2025-01-06"]]
    )
    b = pd.concat(
        [_candles("B", d, range(375)) for d in ["2025-01-02", "2025-01-06"]]
    )  # B out on 3rd
    panel, report = build_bars(pd.concat([a, b]), calendar, symbols=["A", "B", "C"])
    validate_panel(panel)
    assert panel.attrs["symbols"] == ("A", "B")
    assert report.symbols_without_data == ["C"]
    assert report.n_sessions == 3  # trimmed to data range, weekend skipped
    assert report.per_symbol.loc["B", "missing_sessions"] == 1
    assert report.empty_sessions == []


def test_reports_sessions_with_no_data_at_all(calendar):
    raw = pd.concat([_candles("A", d, range(375)) for d in ["2025-01-02", "2025-01-06"]])
    _, report = build_bars(raw, calendar)
    assert report.empty_sessions == [dt.date(2025, 1, 3)]


def test_rejects_bar_not_multiple_of_source(calendar):
    with pytest.raises(ValueError, match="multiple"):
        build_bars(
            _candles("A", "2025-01-02", range(5)),
            calendar,
            bar_minutes=5,
            source_interval_minutes=3,
        )


def test_synthetic_export_end_to_end(synthetic_export, calendar):
    raw = load_candles(synthetic_export["root"])
    panel, report = build_bars(raw, calendar)
    validate_panel(panel)
    assert report.duplicates_dropped > 0 and report.rows_outside_hours > 0
    assert 0 < panel["is_missing"].mean() < 0.1


def test_data_spanning_a_close_time_change_uses_the_common_window(calendar):
    # Fri 31 Jul 2026 trades to 15:30; Mon 3 Aug 2026 (closing auction) only to 15:15.
    raw = pd.concat(
        [_candles("A", "2026-07-31", range(375)), _candles("A", "2026-08-03", range(360))],
        ignore_index=True,
    )
    panel, report = build_bars(raw, calendar)
    validate_panel(panel)
    assert panel.attrs["bars_per_session"] == 72 and len(panel) == 2 * 72
    assert not panel["is_missing"].any()  # the short day is complete, not "partial"
    assert panel["ts"].dt.strftime("%H:%M").max() == "15:10"
    assert report.rows_outside_hours == 15 and report.session_window == "09:15-15:15"
    assert "15:15 from 2026-08-03" in report.to_markdown(calendar)

    # Data entirely before the change keeps the full 75-bar session.
    panel, report = build_bars(_candles("A", "2026-07-31", range(375)), calendar)
    assert panel.attrs["bars_per_session"] == 75 and report.hours_note == ""


def test_warns_when_recent_sessions_all_stop_early(calendar):
    # A week in 2025 (no change listed): every day stops at 15:15 -> hours probably changed.
    days = ["2025-01-06", "2025-01-07", "2025-01-08", "2025-01-09", "2025-01-10"]
    short = pd.concat([_candles("A", d, range(360)) for d in days], ignore_index=True)
    _, report = build_bars(short, calendar)
    assert "no data after 15:15" in report.hours_warning
    assert "close_changes" in report.to_markdown(calendar)

    # One short day (an outage) or full days: no warning.
    mixed = pd.concat(
        [_candles("A", d, range(375)) for d in days[:4]] + [_candles("A", days[4], range(360))],
        ignore_index=True,
    )
    assert build_bars(mixed, calendar)[1].hours_warning == ""
