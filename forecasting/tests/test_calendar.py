import datetime as dt

import pandas as pd
import pytest

from forecasting.data.calendar import NSECalendar

D = dt.date


def test_holidays_weekends_and_extra_sessions(calendar):
    assert not calendar.is_session(D(2025, 3, 14))  # Holi (Friday)
    assert not calendar.is_session(D(2025, 3, 15))  # Saturday
    assert calendar.is_session(D(2025, 3, 13))  # ordinary Thursday
    assert calendar.is_session(D(2025, 2, 1))  # Budget-day Saturday session
    assert not calendar.is_session(D(2024, 1, 22))  # special holiday


def test_session_count_2025(calendar):
    # 261 weekdays - 14 weekday holidays + 1 Saturday budget session
    assert len(calendar.sessions("2025-01-01", "2025-12-31")) == 248


def test_next_and_previous_session_skip_holidays(calendar):
    assert calendar.next_session(D(2025, 3, 13)) == D(2025, 3, 17)
    assert calendar.previous_session(D(2025, 3, 17)) == D(2025, 3, 13)


def test_bar_grid(calendar):
    bars = calendar.bar_starts(D(2025, 1, 2), 5)
    assert len(bars) == 75 == calendar.bars_per_session(5)
    assert bars[0] == pd.Timestamp("2025-01-02 09:15", tz="Asia/Kolkata")
    assert bars[-1] == pd.Timestamp("2025-01-02 15:25", tz="Asia/Kolkata")
    assert calendar.session_close(D(2025, 1, 2)) == bars[-1] + pd.Timedelta(minutes=5)


def test_bar_minutes_must_divide_session(calendar):
    with pytest.raises(ValueError):
        calendar.bars_per_session(7)


def test_uncovered_year_warns(calendar):
    with pytest.warns(UserWarning, match="No NSE holiday list"):
        calendar.sessions("2031-01-01", "2031-01-10")


def test_yaml_rejects_date_under_wrong_year(tmp_path):
    p = tmp_path / "cal.yaml"
    p.write_text("years:\n  2025:\n    holidays:\n      2024-12-25: Christmas\n")
    with pytest.raises(ValueError, match="listed under year"):
        NSECalendar.from_yaml(p)


def test_yaml_rejects_holiday_that_is_also_extra_session(tmp_path):
    p = tmp_path / "cal.yaml"
    p.write_text(
        "years:\n  2025:\n    holidays:\n      2025-02-03: x\n"
        "    extra_sessions:\n      2025-02-03: y\n"
    )
    with pytest.raises(ValueError, match="both holidays and extra sessions"):
        NSECalendar.from_yaml(p)
