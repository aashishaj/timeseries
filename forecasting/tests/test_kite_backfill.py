import datetime as dt
import stat

import pandas as pd
import pytest

from forecasting.data.calendar import NSECalendar
from forecasting.data.kite_backfill import (
    MAX_DAYS_PER_REQUEST,
    backfill,
    date_chunks,
    read_access_token,
    save_access_token,
)
from forecasting.data.loader import load_candles
from forecasting.data.resample import build_bars


class FakeKite:
    """Mimics KiteConnect.instruments / historical_data (tz-aware IST 'date' column)."""

    def __init__(self, fail_first=0):
        self.calls = []
        self.fail_first = fail_first

    def instruments(self, exchange):
        return [
            {
                "tradingsymbol": "INFY",
                "instrument_token": 408065,
                "segment": "NSE",
                "instrument_type": "EQ",
            },
            {
                "tradingsymbol": "TCS",
                "instrument_token": 2953217,
                "segment": "NSE",
                "instrument_type": "EQ",
            },
            {
                "tradingsymbol": "NIFTY 50",
                "instrument_token": 256265,
                "segment": "INDICES",
                "instrument_type": "EQ",
            },
        ]

    def historical_data(self, token, from_date, to_date, interval, **kwargs):
        if self.fail_first:
            self.fail_first -= 1
            raise ConnectionError("blip")
        assert (to_date - from_date).days < 60  # Kite's minute-candle limit
        self.calls.append((token, from_date, to_date))
        cal = NSECalendar.from_yaml()
        rows = []
        for day in cal.sessions(from_date, to_date):
            for i, ts in enumerate(
                pd.date_range(f"{day} 09:15", periods=375, freq="1min", tz="Asia/Kolkata")
            ):
                p = 1000 + i * 0.05
                rows.append(
                    {
                        "date": ts.to_pydatetime(),
                        "open": p,
                        "high": p + 0.1,
                        "low": p - 0.1,
                        "close": p + 0.05,
                        "volume": 100,
                    }
                )
        return rows


def test_chunks_respect_kite_limit_and_cover_range():
    chunks = list(date_chunks(dt.date(2025, 1, 1), dt.date(2025, 12, 31)))
    assert chunks[0][0] == dt.date(2025, 1, 1) and chunks[-1][1] == dt.date(2025, 12, 31)
    assert all((b - a).days + 1 <= MAX_DAYS_PER_REQUEST for a, b in chunks)
    assert all(
        chunks[i][1] + dt.timedelta(days=1) == chunks[i + 1][0] for i in range(len(chunks) - 1)
    )


def test_backfill_writes_loader_layout_and_resumes(tmp_path):
    kite = FakeKite(fail_first=1)
    start, end = dt.date(2025, 3, 3), dt.date(2025, 3, 14)
    written = backfill(kite, ["INFY", "TCS"], start, end, tmp_path, sleep=lambda s: None)
    assert written == {"INFY": 9, "TCS": 9}  # 10 weekdays minus Holi (2025-03-14)
    df = load_candles(tmp_path)
    panel, report = build_bars(df, NSECalendar.from_yaml())
    assert report.n_sessions == 9 and not panel["is_missing"].any()
    calls = len(kite.calls)
    again = backfill(kite, ["INFY", "TCS"], start, end, tmp_path, sleep=lambda s: None)
    assert again == {"INFY": 0, "TCS": 0}
    assert len(kite.calls) >= calls  # holidays make chunks re-checked, never re-written


def test_backfill_refuses_foreign_folder_and_unknown_symbols(tmp_path):
    (tmp_path / "live_export.parquet").write_text("not ours")
    with pytest.raises(FileExistsError):
        backfill(FakeKite(), ["INFY"], dt.date(2025, 3, 3), dt.date(2025, 3, 4), tmp_path)
    with pytest.raises(KeyError, match="RELIANCEX"):
        backfill(
            FakeKite(),
            ["RELIANCEX"],
            dt.date(2025, 3, 3),
            dt.date(2025, 3, 4),
            tmp_path / "new",
            sleep=lambda s: None,
        )


def test_token_file_is_private_and_env_wins(tmp_path, monkeypatch):
    path = save_access_token("secret-token", tmp_path / "tok")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    monkeypatch.delenv("KITE_ACCESS_TOKEN", raising=False)
    assert read_access_token(path) == "secret-token"
    monkeypatch.setenv("KITE_ACCESS_TOKEN", "from-env")
    assert read_access_token(path) == "from-env"
    monkeypatch.delenv("KITE_ACCESS_TOKEN")
    with pytest.raises(RuntimeError, match="kite-login"):
        read_access_token(tmp_path / "missing")


def _free_port():
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _redirect(port, query):
    import time
    import urllib.request

    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for _ in range(50):  # wait until the listener is up
        try:
            return opener.open(f"http://127.0.0.1:{port}/callback?{query}", timeout=2).read()
        except OSError:
            time.sleep(0.05)
    raise AssertionError("listener never came up")


def test_local_listener_catches_request_token():
    import threading

    from forecasting.data.kite_backfill import wait_for_request_token

    port = _free_port()
    out = {}
    t = threading.Thread(target=lambda: out.update(token=wait_for_request_token(port, 10)))
    t.start()
    body = _redirect(port, "action=login&type=login&status=success&request_token=abc123")
    t.join(5)
    assert out["token"] == "abc123" and b"close this tab" in body


def test_login_saves_token_privately_and_never_prints_secrets(tmp_path, monkeypatch, capsys):
    import sys
    import types

    import forecasting.data.kite_backfill as kb

    seen = {}

    class FakeKiteConnect:
        def __init__(self, api_key):
            seen["api_key"] = api_key

        def login_url(self):
            return "https://kite.example/connect/login?v=3&api_key=KEY"

        def generate_session(self, request_token, api_secret):
            seen.update(request_token=request_token, secret=api_secret)
            return {"access_token": "ACCESS-TOKEN-VALUE"}

    monkeypatch.setitem(
        sys.modules, "kiteconnect", types.SimpleNamespace(KiteConnect=FakeKiteConnect)
    )
    monkeypatch.setattr(kb, "API_KEY_FILE", tmp_path / "kite_api_key")
    monkeypatch.setattr(kb, "wait_for_request_token", lambda port: "REQ")
    monkeypatch.delenv("KITE_API_KEY", raising=False)
    monkeypatch.delenv("KITE_API_SECRET", raising=False)
    path = kb.interactive_login(
        tmp_path / "tok",
        open_browser=False,
        ask=lambda prompt: "KEY",
        ask_secret=lambda prompt: "SUPER-SECRET",
    )
    assert seen == {"api_key": "KEY", "request_token": "REQ", "secret": "SUPER-SECRET"}
    assert path.read_text() == "ACCESS-TOKEN-VALUE"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert (tmp_path / "kite_api_key").read_text() == "KEY"
    printed = capsys.readouterr().out
    assert "SUPER-SECRET" not in printed and "ACCESS-TOKEN-VALUE" not in printed
