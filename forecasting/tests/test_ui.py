"""The local app: its safety checks, the Kite login round trip and a download job,
against a fake Kite (no network, no real credentials)."""

import datetime as dt
import http.client
import json
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from forecasting.ui.server import App, make_handler


class FakeKite:
    sessions: list[tuple[str, str]] = []

    def __init__(self, api_key):
        self.api_key = api_key

    def login_url(self):
        return f"https://kite.zerodha.com/connect/login?api_key={self.api_key}&v=3"

    def generate_session(self, request_token, api_secret):
        FakeKite.sessions.append((request_token, api_secret))
        return {"access_token": "ACCESS-" + request_token}

    def set_access_token(self, token):
        self.token = token

    def instruments(self, exchange):
        return [
            {"tradingsymbol": "INFY", "instrument_token": 1, "segment": "NSE",
             "instrument_type": "EQ"}
        ]  # fmt: skip

    def historical_data(self, token, lo, hi, interval):
        days = pd.bdate_range(max(lo, dt.date(2024, 1, 1)), hi).tz_localize("Asia/Kolkata")
        return [
            {"date": d, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1}
            for d in days
        ]


@pytest.fixture
def running(tmp_path):
    app = App(
        project_root=tmp_path,
        config_dir=tmp_path / "private",
        kite_factory=FakeKite,
        symbols=["INFY"],
        now=lambda: dt.datetime(2024, 3, 1, 10, 0),
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app, 0))
    port = server.server_address[1]
    server.RequestHandlerClass = make_handler(app, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield app, port
    server.shutdown()


def call(port, method, path, body=None, headers=None, host=None):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    h = {"Host": host or f"127.0.0.1:{port}", **(headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    if data:
        h["Content-Type"] = "application/json"
    conn.request(method, path, body=data, headers=h)
    r = conn.getresponse()
    return r.status, r.getheader("Location"), r.read()


def post(app, port, path, body=None, **kw):
    return call(port, "POST", path, body or {}, {"X-UI-Token": app.ui_token}, **kw)


def test_page_carries_the_action_token_and_status_reveals_no_secrets(running):
    app, port = running
    status, _, page = call(port, "GET", "/")
    assert status == 200 and app.ui_token.encode() in page
    post(app, port, "/api/credentials", {"api_key": "KEY123", "api_secret": "SECRET456"})
    _, _, raw = call(port, "GET", "/api/status")
    assert b"KEY123" not in raw and b"SECRET456" not in raw
    kite = json.loads(raw)["kite"]
    assert kite["key_saved"] and kite["secret_saved"] and not kite["logged_in_today"]
    for name in ("kite_api_key", "kite_api_secret"):
        assert (app.config_dir / name).stat().st_mode & 0o077 == 0  # only you can read


def test_actions_need_the_page_token_the_right_host_and_no_foreign_origin(running):
    app, port = running
    assert call(port, "POST", "/api/login", {})[0] == 403  # no token
    bad = {"X-UI-Token": "guess"}
    assert call(port, "POST", "/api/login", {}, bad)[0] == 403
    assert post(app, port, "/api/login", host="evil.example:80")[0] == 403  # DNS rebinding
    assert call(port, "GET", "/api/status", host="evil.example")[0] == 403
    foreign = {"X-UI-Token": app.ui_token, "Origin": "https://evil.example"}
    assert call(port, "POST", "/api/login", {}, foreign)[0] == 403


def test_login_round_trip_saves_the_token_privately(running):
    app, port = running
    post(app, port, "/api/credentials", {"api_key": "K", "api_secret": "S"})
    status, _, raw = post(app, port, "/api/login")
    url = json.loads(raw)["url"]
    assert url.startswith("https://kite.zerodha.com/connect/login?api_key=K")
    state = parse_qs(parse_qs(urlparse(url).query)["redirect_params"][0])["state"][0]

    status, where, _ = call(
        port, "GET", f"/callback?status=success&request_token=RT1&state={state}"
    )
    assert status == 303 and where == "/?login=ok"
    assert FakeKite.sessions[-1] == ("RT1", "S")
    assert (app.config_dir / "kite_access_token").read_text() == "ACCESS-RT1"
    assert (app.config_dir / "kite_access_token").stat().st_mode & 0o077 == 0

    # the same callback again (replay) is refused: the login was single use
    status, where, _ = call(
        port, "GET", f"/callback?status=success&request_token=RT2&state={state}"
    )
    assert where.startswith("/?login=failed")


def test_callback_without_a_login_started_here_or_with_wrong_state_is_refused(running):
    app, port = running
    post(app, port, "/api/credentials", {"api_key": "K", "api_secret": "S"})
    assert call(port, "GET", "/callback?request_token=X")[1].startswith("/?login=failed")
    post(app, port, "/api/login")
    where = call(port, "GET", "/callback?status=success&request_token=X&state=wrong")[1]
    assert where.startswith("/?login=failed")
    assert not (app.config_dir / "kite_access_token").exists()


def test_token_counts_as_fresh_until_six_the_next_morning(tmp_path):
    app = App(project_root=tmp_path, config_dir=tmp_path, now=lambda: dt.datetime(2024, 3, 2, 5, 0))
    (tmp_path / "kite_access_token").write_text("x")
    saved = dt.datetime(2024, 3, 1, 9, 0).timestamp()
    import os

    os.utime(tmp_path / "kite_access_token", (saved, saved))
    assert app.token_fresh()  # 05:00 next day: still valid
    app.now = lambda: dt.datetime(2024, 3, 2, 6, 30)
    assert not app.token_fresh()  # after 06:00: expired


def test_download_needs_a_login_then_runs_as_a_background_job(running):
    app, port = running
    status, _, raw = post(app, port, "/api/download-daily")
    assert status == 409 and b"Log in" in raw
    post(app, port, "/api/credentials", {"api_key": "K", "api_secret": "S"})
    (app.config_dir / "kite_access_token").write_text("T")  # saved "now" (fresh)
    app.now = lambda: dt.datetime.now()
    assert post(app, port, "/api/download-daily")[0] == 200
    assert post(app, port, "/api/download-daily")[0] in (200, 409)  # one job at a time
    for _ in range(100):
        job = json.loads(call(port, "GET", "/api/status")[2])["job"]
        if not job["running"]:
            break
        time.sleep(0.05)
    assert job["ok"], job["lines"]
    daily = json.loads(call(port, "GET", "/api/status")[2])["daily"]["symbols"]
    assert daily[0]["symbol"] == "INFY" and daily[0]["days"] > 200


def test_study_results_are_served_read_only(running):
    app, port = running
    assert json.loads(call(port, "GET", "/api/study")[2]) == {"available": False}
    folder = app.root / "results" / "timing"
    folder.mkdir(parents=True)
    rows = [
        {"instrument": i, "rule": r, "cagr": c, "max_drawdown": -0.3, "months_2pct": 0.4,
         "trades_per_year": 1.0, "passes": p, "start": "2007-01-01", "end": "2023-11-23"}
        for i in ("AAA", "NIFTYBEES")
        for r, c, p in (("B", 0.12, False), ("T", 0.06, False))
    ]  # fmt: skip
    pd.DataFrame(rows).to_csv(folder / "metrics_dev.csv", index=False)
    study = json.loads(call(port, "GET", "/api/study")[2])
    assert study["available"] and "final" not in study
    first, second = study["dev"]["rules"]
    assert first["benchmark"] and first["yearly_return"] == pytest.approx(0.12)
    assert second["stocks_passing"] == 0 and second["market_passes"] is False
    assert call(port, "POST", "/api/study", {})[0] == 403  # nothing to post: read-only
