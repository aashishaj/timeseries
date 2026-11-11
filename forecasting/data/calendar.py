"""NSE trading calendar: which dates are sessions, and the intraday bar grid.

Only the regular continuous session (09:15-15:30 IST) is modelled. Pre-open
(09:00-09:08), the closing session and non-regular special sessions (Muhurat,
DR drills) are excluded on purpose.
"""

from __future__ import annotations

import datetime as dt
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

IST = "Asia/Kolkata"
DEFAULT_CALENDAR_PATH = Path(__file__).resolve().parents[1] / "configs" / "nse_calendar.yaml"


def to_date(value: dt.date | dt.datetime | str | pd.Timestamp) -> dt.date:
    """Coerce a date-like value to ``datetime.date``."""
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def _parse_time(value: str | dt.time) -> dt.time:
    if isinstance(value, dt.time):
        return value
    return dt.datetime.strptime(str(value), "%H:%M").time()


@dataclass(frozen=True)
class NSECalendar:
    """Session calendar for the NSE cash market.

    Attributes:
        holidays: weekday dates on which the exchange is closed.
        extra_sessions: weekend dates with a full regular session.
        special_sessions: dates with non-regular sessions that are deliberately not modelled.
        covered_years: years for which the holiday list was provided.
    """

    holidays: frozenset[dt.date] = frozenset()
    extra_sessions: frozenset[dt.date] = frozenset()
    special_sessions: frozenset[dt.date] = frozenset()
    covered_years: frozenset[int] = frozenset()
    open_time: dt.time = dt.time(9, 15)
    close_time: dt.time = dt.time(15, 30)
    tz: str = IST
    names: dict[dt.date, str] = field(default_factory=dict, compare=False, repr=False)

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> NSECalendar:
        path = Path(path) if path is not None else DEFAULT_CALENDAR_PATH
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        holidays: set[dt.date] = set()
        extra: set[dt.date] = set()
        special: set[dt.date] = set()
        names: dict[dt.date, str] = {}
        years: set[int] = set()
        for year, spec in (raw.get("years") or {}).items():
            years.add(int(year))
            spec = spec or {}
            for bucket, target in (
                ("holidays", holidays),
                ("extra_sessions", extra),
                ("special_sessions", special),
            ):
                for d, name in (spec.get(bucket) or {}).items():
                    d = to_date(d)
                    if d.year != int(year):
                        raise ValueError(f"{d} listed under year {year} in {path}")
                    target.add(d)
                    names.setdefault(d, str(name))
        clash = holidays & extra
        if clash:
            raise ValueError(f"dates are both holidays and extra sessions: {sorted(clash)}")
        session = raw.get("regular_session") or {}
        return cls(
            holidays=frozenset(holidays),
            extra_sessions=frozenset(extra),
            special_sessions=frozenset(special),
            covered_years=frozenset(years),
            open_time=_parse_time(session.get("open", "09:15")),
            close_time=_parse_time(session.get("close", "15:30")),
            tz=raw.get("timezone", IST),
            names=names,
        )

    # ------------------------------------------------------------------ sessions
    @property
    def session_minutes(self) -> int:
        start = dt.datetime.combine(dt.date(2000, 1, 1), self.open_time)
        end = dt.datetime.combine(dt.date(2000, 1, 1), self.close_time)
        return int((end - start).total_seconds() // 60)

    def bars_per_session(self, bar_minutes: int) -> int:
        if bar_minutes <= 0 or self.session_minutes % bar_minutes:
            raise ValueError(
                f"bar_minutes={bar_minutes} does not divide the {self.session_minutes}-minute session"
            )
        return self.session_minutes // bar_minutes

    def is_session(self, day: dt.date | str | pd.Timestamp) -> bool:
        d = to_date(day)
        if d in self.holidays:
            return False
        if d in self.extra_sessions:
            return True
        return d.weekday() < 5

    def sessions(self, start, end) -> list[dt.date]:
        """All regular sessions between ``start`` and ``end`` (inclusive)."""
        start_d, end_d = to_date(start), to_date(end)
        if end_d < start_d:
            return []
        uncovered = sorted(
            {y for y in range(start_d.year, end_d.year + 1)} - set(self.covered_years)
        )
        if uncovered:
            warnings.warn(
                f"No NSE holiday list for {uncovered}; every weekday is treated as a session. "
                "Add the year to configs/nse_calendar.yaml.",
                stacklevel=2,
            )
        days = pd.date_range(start_d, end_d, freq="D")
        return [d.date() for d in days if self.is_session(d)]

    def next_session(self, day) -> dt.date:
        d = to_date(day) + dt.timedelta(days=1)
        for _ in range(30):
            if self.is_session(d):
                return d
            d += dt.timedelta(days=1)
        raise RuntimeError(f"no session within 30 days after {day}")

    def previous_session(self, day) -> dt.date:
        d = to_date(day) - dt.timedelta(days=1)
        for _ in range(30):
            if self.is_session(d):
                return d
            d -= dt.timedelta(days=1)
        raise RuntimeError(f"no session within 30 days before {day}")

    # ------------------------------------------------------------------ intraday grid
    def session_open(self, day) -> pd.Timestamp:
        return pd.Timestamp(dt.datetime.combine(to_date(day), self.open_time), tz=self.tz)

    def session_close(self, day) -> pd.Timestamp:
        return pd.Timestamp(dt.datetime.combine(to_date(day), self.close_time), tz=self.tz)

    def bar_starts(self, day, bar_minutes: int) -> pd.DatetimeIndex:
        """Start timestamps of every bar in a session, e.g. 09:15, 09:20, ..., 15:25."""
        n = self.bars_per_session(bar_minutes)
        return pd.date_range(self.session_open(day), periods=n, freq=f"{bar_minutes}min")

    def describe(self, day) -> str:
        d = to_date(day)
        return self.names.get(d, "")

