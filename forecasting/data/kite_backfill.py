"""Backfill 1-minute candles from the Kite Connect historical data API.

Runs on your own machine with your own credentials; nothing here is meant for a
cloud session. Credentials are read from the environment or a private token file
and are never printed, logged or written into the repository.

* ``KITE_API_KEY``      your app's API key (or the file ``kite-login`` writes next to the
  token, so later sessions do not need the environment variable)
* ``KITE_ACCESS_TOKEN`` today's access token, or the file written by ``kite-login``
  (``~/.config/nse-forecasting/kite_access_token`` by default)
* ``KITE_API_SECRET``   only needed by ``kite-login``

Output goes to its own folder in the loader's layout
(``<out>/symbol=X/date=YYYY-MM-DD/part-0.parquet``), never into the live pipeline's
export. Re-running resumes: days already on disk are skipped.

Kite limits (as documented by Zerodha; verify if they change): minute candles at
most 60 days per request, about 3 historical requests per second.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import stat
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Protocol

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

MARKER = "_KITE_BACKFILL"
CONFIG_DIR = Path.home() / ".config" / "nse-forecasting"
DEFAULT_TOKEN_FILE = CONFIG_DIR / "kite_access_token"
API_KEY_FILE = CONFIG_DIR / "kite_api_key"
API_SECRET_FILE = CONFIG_DIR / "kite_api_secret"  # only with --remember-secret
MAX_DAYS_PER_REQUEST = 55  # Kite allows 60 for minute candles; stay under it


class HistoricalClient(Protocol):
    """The two KiteConnect methods we use (lets tests pass a fake client)."""

    def instruments(self, exchange: str) -> list[dict]: ...

    def historical_data(
        self, instrument_token: int, from_date, to_date, interval: str, **kwargs
    ) -> list[dict]: ...


# ---------------------------------------------------------------------- credentials
def read_access_token(token_file: Path | None = None) -> str:
    token = os.environ.get("KITE_ACCESS_TOKEN")
    if token:
        return token.strip()
    path = Path(token_file or DEFAULT_TOKEN_FILE)
    if path.exists():
        return path.read_text().strip()
    raise RuntimeError(
        "no Kite access token: set KITE_ACCESS_TOKEN or run `forecasting kite-login` "
        f"(writes {path})"
    )


def save_access_token(token: str, token_file: Path | None = None) -> Path:
    """Store the token in a file only the current user can read (outside the repo)."""
    path = Path(token_file or DEFAULT_TOKEN_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token)
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return path


def read_api_key(path: Path | None = None) -> str:
    key = os.environ.get("KITE_API_KEY")
    if key:
        return key.strip()
    path = Path(path or API_KEY_FILE)
    if path.exists():
        return path.read_text().strip()
    raise RuntimeError("set KITE_API_KEY or run `forecasting kite-login` once")


def make_client(token_file: Path | None = None):
    """A logged-in KiteConnect client from environment / private config files."""
    from kiteconnect import KiteConnect  # optional dependency: uv sync --extra kite

    api_key = read_api_key()
    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(read_access_token(token_file))
    return kite


def wait_for_request_token(
    port: int = 8766, timeout: float = 300.0, host: str = "127.0.0.1"
) -> str | None:
    """Answer the browser's redirect to ``http://host:port/...`` and return its
    ``request_token`` (None on timeout). The URL is never logged: it holds the token."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from urllib.parse import parse_qs, urlparse

    result: dict[str, str] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 (http.server naming)
            token = parse_qs(urlparse(self.path).query).get("request_token", [None])[0]
            if token:
                result["token"] = token
            body = b"Kite login received. You can close this tab." if token else b"Waiting..."
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:  # keep the token out of the terminal
            return

    server = HTTPServer((host, port), Handler)
    server.timeout = 1.0
    deadline = time.monotonic() + timeout
    try:
        while "token" not in result and time.monotonic() < deadline:
            server.handle_request()
    finally:
        server.server_close()
    return result.get("token")


def interactive_login(
    token_file: Path | None = None,
    *,
    port: int = 8766,
    open_browser: bool = True,
    remember_secret: bool = False,
    ask: Callable[[str], str] = input,
    ask_secret: Callable[[str], str] | None = None,
) -> Path:
    """Browser login -> request token -> access token, saved privately.

    The API key and secret come from the environment, the private config folder, or
    hidden prompts (so they never land in shell history). The secret is stored (0600,
    outside the repo) only with ``remember_secret=True``. The request token
    is caught by a one-shot local server on the app's redirect port (127.0.0.1:<port>),
    falling back to pasting the redirect URL if that port is busy.
    """
    import getpass

    from kiteconnect import KiteConnect

    ask_secret = ask_secret or getpass.getpass
    api_key = os.environ.get("KITE_API_KEY") or (
        API_KEY_FILE.read_text().strip() if API_KEY_FILE.exists() else ""
    )
    api_key = api_key or ask("Kite API key: ").strip()
    secret = os.environ.get("KITE_API_SECRET") or (
        API_SECRET_FILE.read_text().strip() if API_SECRET_FILE.exists() else ""
    )
    secret = secret or ask_secret("Kite API secret (hidden): ")
    if remember_secret:
        save_access_token(secret.strip(), API_SECRET_FILE)  # 0600, outside the repo
    kite = KiteConnect(api_key=api_key)
    url = kite.login_url()
    print("Log in to Zerodha in your browser:\n  " + url)
    if open_browser:
        import webbrowser

        webbrowser.open(url)
    try:
        print(f"Waiting for the redirect to 127.0.0.1:{port} (up to 5 minutes)...")
        request_token = wait_for_request_token(port)
    except OSError:  # port in use (e.g. the live pipeline listens there)
        request_token = None
        print(f"Port {port} is busy; falling back to copy-paste.")
    if not request_token:
        answer = ask("Paste the full URL you were redirected to: ").strip()
        request_token = answer
        if "request_token=" in answer:
            request_token = answer.split("request_token=", 1)[1].split("&", 1)[0]
    session = kite.generate_session(request_token, api_secret=secret.strip())
    save_access_token(api_key, API_KEY_FILE)  # same private permissions; not the secret
    return save_access_token(session["access_token"], token_file)


# ---------------------------------------------------------------------- backfill
def date_chunks(start: dt.date, end: dt.date, days: int = MAX_DAYS_PER_REQUEST):
    """Inclusive [from, to] date ranges of at most ``days`` days covering start..end."""
    cur = start
    while cur <= end:
        stop = min(cur + dt.timedelta(days=days - 1), end)
        yield cur, stop
        cur = stop + dt.timedelta(days=1)


def instrument_tokens(client: HistoricalClient, symbols: Iterable[str]) -> dict[str, int]:
    wanted = set(symbols)
    found = {
        row["tradingsymbol"]: int(row["instrument_token"])
        for row in client.instruments("NSE")
        if row.get("tradingsymbol") in wanted
        and row.get("segment", "NSE") == "NSE"
        and row.get("instrument_type", "EQ") == "EQ"
    }
    missing = sorted(wanted - set(found))
    if missing:
        raise KeyError(f"not found on NSE (check tradingsymbols in symbols.yaml): {missing}")
    return found


def _existing_days(out: Path, symbol: str) -> set[dt.date]:
    folder = out / f"symbol={symbol}"
    if not folder.exists():
        return set()
    days = set()
    for d in folder.iterdir():
        if d.name.startswith("date=") and any(d.glob("*.parquet")):
            days.add(dt.date.fromisoformat(d.name.split("=", 1)[1]))
    return days


def _write_day(out: Path, symbol: str, day: dt.date, candles: pd.DataFrame) -> None:
    folder = out / f"symbol={symbol}" / f"date={day.isoformat()}"
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "part-0.parquet.tmp"
    pq.write_table(pa.Table.from_pandas(candles, preserve_index=False), tmp)
    tmp.replace(folder / "part-0.parquet")  # atomic: no half-written days on interrupt


def backfill(
    client: HistoricalClient,
    symbols: list[str],
    start: dt.date,
    end: dt.date,
    out: str | Path,
    *,
    interval: str = "minute",
    pause_seconds: float = 0.4,
    retries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, int]:
    """Download candles into ``out``; returns the number of days written per symbol."""
    out = Path(out)
    if out.exists() and any(out.iterdir()) and not (out / MARKER).exists():
        raise FileExistsError(f"{out} is not empty and is not a Kite backfill folder")
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text("1-minute candles from the Kite Connect historical API\n")
    tokens = instrument_tokens(client, symbols)
    written: dict[str, int] = {}
    for symbol in symbols:
        have = _existing_days(out, symbol)
        written[symbol] = 0
        for lo, hi in date_chunks(start, end):
            weekdays = {lo + dt.timedelta(days=i) for i in range((hi - lo).days + 1)}
            if {d for d in weekdays if d.weekday() < 5} <= have:
                continue  # every weekday of this chunk is already on disk
            rows = _fetch(client, tokens[symbol], lo, hi, interval, retries, sleep)
            sleep(pause_seconds)
            if not rows:
                continue
            df = pd.DataFrame(rows).rename(columns={"date": "timestamp"})
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("Asia/Kolkata")
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            for day, g in df.groupby(df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.date):
                if day not in have:
                    _write_day(out, symbol, day, g.reset_index(drop=True))
                    written[symbol] += 1
        log.info("%s: %d new days", symbol, written[symbol])
    return written


def _fetch(client, token, lo, hi, interval, retries, sleep) -> list[dict]:
    for attempt in range(retries + 1):
        try:
            return client.historical_data(token, lo, hi, interval)
        except Exception as exc:  # network blips / rate limits: back off and retry
            if attempt == retries:
                raise
            wait = 2.0**attempt
            log.warning("historical_data failed (%s); retrying in %.0fs", type(exc).__name__, wait)
            sleep(wait)
    return []
