"""Turn raw exported candles into a clean, balanced panel of N-minute bars.

Cleaning rules (all counted in :class:`BuildReport`, nothing is dropped silently):

* duplicate (symbol, ts) rows: keep the last one (pipeline restarts can re-emit candles)
* rows with impossible prices (<= 0, high < low): dropped
* rows on dates that are not sessions: dropped (holidays, weekends, Muhurat, ...)
* rows outside 09:15-15:30 IST (pre-open, closing session): dropped
* bars with no source candles: kept as NaN rows with ``is_missing=True`` so the
  time grid stays calendar-aligned. Gaps are never forward-filled.

Bars are aligned to the session open, so 5-minute bars are 09:15-09:20, ...,
15:25-15:30 (75 per session). A bar's timestamp is its start.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecasting.data.calendar import NSECalendar, to_date
from forecasting.data.panel import PANEL_COLUMNS

log = logging.getLogger(__name__)


@dataclass
class BuildReport:
    """What happened while building bars; rendered by ``forecasting validate``."""

    rows_in: int = 0
    duplicates_dropped: int = 0
    invalid_price_rows: int = 0
    rows_non_session_date: int = 0
    rows_outside_hours: int = 0
    rows_used: int = 0
    first_session: dt.date | None = None
    last_session: dt.date | None = None
    n_sessions: int = 0
    symbols_without_data: list[str] = field(default_factory=list)
    special_session_dates: list[dt.date] = field(default_factory=list)
    unexpected_dates: list[dt.date] = field(default_factory=list)
    empty_sessions: list[dt.date] = field(default_factory=list)
    per_symbol: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_markdown(self, calendar: NSECalendar | None = None) -> str:
        lines = [
            "## Data quality",
            "",
            f"- sessions: {self.n_sessions} ({self.first_session} -> {self.last_session})",
            f"- source rows read: {self.rows_in:,}; used: {self.rows_used:,}",
            f"- duplicate rows dropped: {self.duplicates_dropped:,}",
            f"- invalid price rows dropped: {self.invalid_price_rows:,}",
            f"- rows on non-session dates dropped: {self.rows_non_session_date:,}",
            f"- rows outside 09:15-15:30 dropped: {self.rows_outside_hours:,}",
        ]
        if self.symbols_without_data:
            lines.append(f"- **symbols with no data (excluded):** {self.symbols_without_data}")

        def _dates(ds: list[dt.date]) -> str:
            shown = [
                f"{d}" + (f" ({calendar.describe(d)})" if calendar and calendar.describe(d) else "")
                for d in ds[:10]
            ]
            return ", ".join(shown) + (f", ... (+{len(ds) - 10})" if len(ds) > 10 else "")

        if self.special_session_dates:
            lines.append(
                f"- known special sessions ignored: {_dates(self.special_session_dates)}"
            )
        if self.unexpected_dates:
            lines.append(
                "- **data on dates the calendar says are closed** (check nse_calendar.yaml): "
                + _dates(self.unexpected_dates)
            )
        if self.empty_sessions:
            lines.append(
                "- **sessions with no data for any symbol** (holiday missing from the "
                "calendar, or a pipeline outage): " + _dates(self.empty_sessions)
            )
        if not self.per_symbol.empty:
            lines += ["", _frame_to_markdown(self.per_symbol.reset_index())]
        return "\n".join(lines)


def _frame_to_markdown(df: pd.DataFrame) -> str:
    def fmt(v):
        if isinstance(v, float):
            return f"{v:.2f}"
        return str(v)

    header = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, sep, *rows])


def build_bars(
    candles: pd.DataFrame,
    calendar: NSECalendar,
    *,
    symbols: list[str] | None = None,
    bar_minutes: int = 5,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
    source_interval_minutes: int = 1,
) -> tuple[pd.DataFrame, BuildReport]:
    """Aggregate candles to ``bar_minutes`` bars on the full session grid.

    Args:
        candles: output of :func:`forecasting.data.loader.load_candles`.
        calendar: trading calendar defining sessions and hours.
        symbols: symbols to keep (default: all present). Symbols without any data are dropped.
        bar_minutes: target bar length; must divide the session and be a multiple of
            ``source_interval_minutes``.
        start, end: optional date bounds; the grid is also trimmed to the data's date range.

    Returns:
        (panel, report). See :mod:`forecasting.data.panel` for the panel contract.
    """
    if bar_minutes % source_interval_minutes:
        raise ValueError(
            f"bar_minutes={bar_minutes} is not a multiple of the source interval "
            f"({source_interval_minutes} min)"
        )
    n_slots = calendar.bars_per_session(bar_minutes)
    report = BuildReport(rows_in=len(candles))
    df = candles
    if symbols is not None:
        df = df[df["symbol"].isin(symbols)]

    dup = df.duplicated(["symbol", "ts"], keep="last")
    report.duplicates_dropped = int(dup.sum())
    df = df.loc[~dup]

    bad = (df[["open", "high", "low", "close"]] <= 0).any(axis=1) | (df["high"] < df["low"])
    report.invalid_price_rows = int(bad.sum())
    df = df.loc[~bad]

    day = df["ts"].dt.date
    session_flag = {d: calendar.is_session(d) for d in pd.unique(day)}
    on_session = day.map(session_flag).astype(bool)
    off_days = sorted(d for d, ok in session_flag.items() if not ok)
    report.special_session_dates = [d for d in off_days if d in calendar.special_sessions]
    report.unexpected_dates = [d for d in off_days if d not in calendar.special_sessions]
    report.rows_non_session_date = int((~on_session).sum())
    df = df.loc[on_session]

    open_minutes = calendar.open_time.hour * 60 + calendar.open_time.minute
    minute = (df["ts"].dt.hour * 60 + df["ts"].dt.minute - open_minutes).to_numpy()
    in_hours = (minute >= 0) & (minute < calendar.session_minutes)
    report.rows_outside_hours = int((~in_hours).sum())
    df = df.loc[in_hours]
    df = df.assign(session=df["ts"].dt.date, slot=minute[in_hours] // bar_minutes)
    report.rows_used = len(df)

    present = sorted(df["symbol"].unique().tolist())
    wanted = list(symbols) if symbols is not None else present
    report.symbols_without_data = [s for s in wanted if s not in present]
    if report.symbols_without_data:
        log.warning("no usable data for symbols %s; excluding them", report.symbols_without_data)
    syms = [s for s in wanted if s in present]
    if not syms:
        raise ValueError("no usable candles for any requested symbol")

    first = max(to_date(start), df["session"].min()) if start else df["session"].min()
    last = min(to_date(end), df["session"].max()) if end else df["session"].max()
    sessions = calendar.sessions(first, last)
    report.first_session, report.last_session = sessions[0], sessions[-1]
    report.n_sessions = len(sessions)

    df = df.sort_values(["symbol", "ts"], kind="stable")
    bars = df.groupby(["session", "slot", "symbol"], sort=False).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        n_source=("close", "size"),
    )
    grid = pd.MultiIndex.from_product(
        [sessions, range(n_slots), syms], names=["session", "slot", "symbol"]
    )
    panel = bars.reindex(grid).reset_index()
    panel["n_source"] = panel["n_source"].fillna(0).astype("int64")
    panel["is_missing"] = panel["n_source"] == 0
    panel["symbol"] = panel["symbol"].astype(str)

    opens = pd.DatetimeIndex([calendar.session_open(d) for d in sessions])
    opens_utc = opens.tz_convert("UTC").tz_localize(None).to_numpy()
    per_step = np.repeat(opens_utc, n_slots) + np.tile(
        np.arange(n_slots) * np.timedelta64(bar_minutes, "m"), len(sessions)
    )
    step_ts = pd.DatetimeIndex(per_step).tz_localize("UTC").tz_convert(calendar.tz)
    panel["ts"] = step_ts.repeat(len(syms))
    panel["bar_end"] = panel["ts"] + pd.Timedelta(minutes=bar_minutes)
    panel = panel[list(PANEL_COLUMNS)]
    panel.attrs = {
        "symbols": tuple(syms),
        "bars_per_session": n_slots,
        "bar_minutes": bar_minutes,
    }

    report.per_symbol = _per_symbol_quality(panel, n_slots)
    sess_present = (~panel["is_missing"]).groupby(panel["session"]).any()
    report.empty_sessions = sorted(sess_present.index[~sess_present.to_numpy()].tolist())
    return panel, report


def _per_symbol_quality(panel: pd.DataFrame, n_slots: int) -> pd.DataFrame:
    present = ~panel["is_missing"]
    by_sess = present.groupby([panel["symbol"], panel["session"]]).sum()
    full = by_sess.groupby(level=0).apply(lambda s: int((s == n_slots).sum()))
    empty = by_sess.groupby(level=0).apply(lambda s: int((s == 0).sum()))
    partial = by_sess.groupby(level=0).apply(lambda s: int(((s > 0) & (s < n_slots)).sum()))
    total = panel.groupby("symbol").size()
    have = present.groupby(panel["symbol"]).sum()
    out = pd.DataFrame(
        {
            "full_sessions": full,
            "partial_sessions": partial,
            "missing_sessions": empty,
            "bars_missing": (total - have).astype(int),
            "pct_bars_missing": 100.0 * (total - have) / total,
        }
    )
    out.index.name = "symbol"
    return out
