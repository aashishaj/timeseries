"""Local control panel for the project, served on http://127.0.0.1:8766.

Port 8766 is the redirect URL registered on the Kite Connect app
(``http://127.0.0.1:8766/callback``), so the Kite login comes straight back here.

Safety:
* listens on 127.0.0.1 only and refuses any other Host header (DNS rebinding);
* every action is a POST that must carry a random token embedded in the page, which
  other web sites cannot read, so they cannot trigger actions from your browser;
* the Kite callback is accepted only for a login started here in the last 10 minutes,
  with a matching random ``state``;
* access tokens, the API key and the API secret are never sent to the page or logged.
"""

from __future__ import annotations

import datetime as dt
import html
import json
import logging
import secrets
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlparse

import pandas as pd

from forecasting.data import kite_backfill as kb

log = logging.getLogger(__name__)

PAGE = Path(__file__).with_name("page.html")
LOGIN_WINDOW_SECONDS = 600
TOKEN_RESET_HOUR = 6  # Kite access tokens stop working at 06:00 IST the next morning


@dataclass
class Job:
    name: str
    started: float = field(default_factory=time.time)
    finished: float | None = None
    ok: bool | None = None
    lines: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "running": self.finished is None,
            "ok": self.ok,
            "started": _clock(self.started),
            "finished": _clock(self.finished) if self.finished else None,
            "seconds": round((self.finished or time.time()) - self.started),
            "lines": self.lines[-300:],
        }


class _JobLogHandler(logging.Handler):
    def __init__(self, job: Job) -> None:
        super().__init__(logging.INFO)
        self.job = job
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        self.job.lines.append(self.format(record))


class App:
    """State and actions behind the web page. Everything is injectable for tests."""

    def __init__(
        self,
        *,
        project_root: Path,
        config_dir: Path = kb.CONFIG_DIR,
        kite_factory: Callable[[str], Any] | None = None,
        symbols: list[str] | None = None,
        now: Callable[[], dt.datetime] = dt.datetime.now,
    ) -> None:
        self.root = Path(project_root)
        self.config_dir = Path(config_dir)
        self.kite_factory = kite_factory or _real_kite
        self.symbols = symbols
        self.now = now
        self.ui_token = secrets.token_urlsafe(24)
        self.daily_dir = self.root / "data" / "kite_daily"
        self.job: Job | None = None
        self._login_state: str | None = None
        self._login_started = 0.0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ files
    @property
    def token_file(self) -> Path:
        return self.config_dir / "kite_access_token"

    @property
    def key_file(self) -> Path:
        return self.config_dir / "kite_api_key"

    @property
    def secret_file(self) -> Path:
        return self.config_dir / "kite_api_secret"

    def _read(self, path: Path) -> str:
        return path.read_text().strip() if path.exists() else ""

    def _save_private(self, path: Path, value: str) -> None:
        kb.save_access_token(value.strip(), path)  # 0600, outside the repo

    # ------------------------------------------------------------------ status
    def token_fresh(self) -> bool:
        if not self.token_file.exists():
            return False
        saved = dt.datetime.fromtimestamp(self.token_file.stat().st_mtime)
        now = self.now()
        reset = now.replace(hour=TOKEN_RESET_HOUR, minute=0, second=0, microsecond=0)
        if now < reset:
            reset -= dt.timedelta(days=1)
        return saved >= reset

    def status(self) -> dict[str, Any]:
        saved = None
        if self.token_file.exists():
            saved = dt.datetime.fromtimestamp(self.token_file.stat().st_mtime)
        return {
            "kite": {
                "key_saved": bool(self._read(self.key_file)),
                "secret_saved": bool(self._read(self.secret_file)),
                "logged_in_today": self.token_fresh(),
                "token_saved_at": saved.strftime("%d %b %Y, %H:%M") if saved else None,
            },
            "daily": self.daily_summary(),
            "job": self.job.as_dict() if self.job else None,
        }

    def study(self) -> dict[str, Any]:
        """The invest / stay-out study's results, if it has been run (read-only)."""
        from forecasting.timing.study import LABELS, MARKET

        folder = self.root / "results" / "timing"
        out: dict[str, Any] = {"available": False}
        for split in ("dev", "final"):
            path = folder / f"metrics_{split}.csv"
            if not path.exists():
                continue
            t = pd.read_csv(path)
            rows = []
            for rule, g in t.groupby("rule", sort=False):
                stocks = g[g["instrument"] != MARKET]
                rows.append(
                    {
                        "rule": LABELS.get(rule, rule),
                        "benchmark": rule == "B",
                        "yearly_return": float(g["cagr"].median()),
                        "largest_fall": float(g["max_drawdown"].median()),
                        "months_2pct": float(g["months_2pct"].mean()),
                        "trades_per_year": float(g["trades_per_year"].mean()),
                        "stocks_passing": None if rule == "B" else int(stocks["passes"].sum()),
                        "n_stocks": len(stocks),
                        "market_passes": None
                        if rule == "B"
                        else bool(g.loc[g["instrument"] == MARKET, "passes"].any()),
                    }
                )
            rows.sort(key=lambda r: (not r["benchmark"], -r["yearly_return"]))
            out[split] = {"start": t["start"].min(), "end": t["end"].max(), "rules": rows}
            out["available"] = True
        return out

    def daily_summary(self) -> dict[str, Any]:
        rows = []
        for path in sorted(self.daily_dir.glob("*.parquet")):
            try:
                dates = pd.read_parquet(path, columns=["date"])["date"]
            except Exception:  # half-written or unreadable: show, do not crash
                rows.append({"symbol": path.stem, "days": 0, "first": None, "last": None})
                continue
            rows.append(
                {
                    "symbol": path.stem,
                    "days": int(len(dates)),
                    "first": str(dates.min()) if len(dates) else None,
                    "last": str(dates.max()) if len(dates) else None,
                }
            )
        return {"folder": str(self.daily_dir.relative_to(self.root)), "symbols": rows}

    # ------------------------------------------------------------------ Kite login
    def save_credentials(self, api_key: str, api_secret: str) -> None:
        if api_key.strip():
            self._save_private(self.key_file, api_key)
        if api_secret.strip():
            self._save_private(self.secret_file, api_secret)

    def start_login(self) -> str:
        api_key = self._read(self.key_file)
        if not api_key:
            raise UserError("Save your Kite API key and secret first.")
        if not self._read(self.secret_file):
            raise UserError("Save your Kite API secret first.")
        with self._lock:
            self._login_state = secrets.token_urlsafe(16)
            self._login_started = time.monotonic()
        url = self.kite_factory(api_key).login_url()
        return f"{url}&redirect_params={quote('state=' + self._login_state)}"

    def finish_login(self, query: dict[str, list[str]]) -> None:
        with self._lock:
            state, started = self._login_state, self._login_started
            self._login_state = None  # one use only
        if state is None or time.monotonic() - started > LOGIN_WINDOW_SECONDS:
            raise UserError("No login was started here recently. Click Log in again.")
        got = query.get("state", [""])[0]
        if got and not secrets.compare_digest(got, state):
            raise UserError("This login did not come from this app. Click Log in again.")
        if query.get("status", ["success"])[0] != "success":
            raise UserError("Zerodha reported the login as not successful.")
        request_token = query.get("request_token", [""])[0]
        if not request_token:
            raise UserError("Zerodha did not send a login code back.")
        kite = self.kite_factory(self._read(self.key_file))
        session = kite.generate_session(request_token, api_secret=self._read(self.secret_file))
        self._save_private(self.token_file, session["access_token"])
        log.info("Kite login complete; access token saved privately")

    # ------------------------------------------------------------------ jobs
    def start_job(self, name: str, work: Callable[[], str | None]) -> None:
        with self._lock:
            if self.job and self.job.finished is None:
                raise UserError(f"Wait for '{self.job.name}' to finish first.")
            job = self.job = Job(name)
        threading.Thread(target=self._run, args=(job, work), daemon=True).start()

    def _run(self, job: Job, work: Callable[[], str | None]) -> None:
        handler = _JobLogHandler(job)
        logger = logging.getLogger("forecasting")
        logger.addHandler(handler)
        try:
            summary = work()
            job.ok = True
            if summary:
                job.lines.append(summary)
        except Exception as exc:  # shown on the page; full trace in the terminal
            job.ok = False
            job.lines.append(f"FAILED: {type(exc).__name__}: {exc}")
            log.error("%s failed:\n%s", job.name, traceback.format_exc())
        finally:
            job.finished = time.time()
            logger.removeHandler(handler)

    def download_daily(self, start: str = "2005-01-01") -> None:
        if not self.token_fresh():
            raise UserError("Log in to Kite first (today's login has not been done).")
        symbols = self.symbols or _default_daily_symbols()

        def work() -> str:
            kite = self.kite_factory(self._read(self.key_file))
            kite.set_access_token(self._read(self.token_file))
            counts = kb.backfill_daily(
                kite, symbols, dt.date.fromisoformat(start), self.now().date(), self.daily_dir
            )
            return f"Done: daily prices for {len(counts)} symbols saved."

        self.start_job("Download daily prices", work)


class UserError(Exception):
    """A problem the page should show as-is (no stack trace)."""


def _real_kite(api_key: str):
    from kiteconnect import KiteConnect  # optional dependency: uv sync --extra kite

    return KiteConnect(api_key=api_key)


def _default_daily_symbols() -> list[str]:
    from forecasting.config import load_config

    return [*load_config().symbols, "NIFTYBEES"]


def _clock(t: float | None) -> str | None:
    return dt.datetime.fromtimestamp(t).strftime("%H:%M:%S") if t else None


# ---------------------------------------------------------------------- HTTP layer
def make_handler(app: App, port: int) -> type[BaseHTTPRequestHandler]:
    allowed_hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}

    class Handler(BaseHTTPRequestHandler):
        server_version = "nse-forecasting-ui"

        def log_message(self, *args) -> None:  # URLs can hold login codes: never log them
            return

        # -------------------------------------------------------------- helpers
        def _send(self, status: int, body: bytes, ctype: str, extra=None) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, payload: dict) -> None:
            self._send(status, json.dumps(payload).encode(), "application/json")

        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts

        def _post_ok(self) -> bool:
            origin = self.headers.get("Origin")
            if origin and urlparse(origin).netloc not in allowed_hosts:
                return False
            sent = self.headers.get("X-UI-Token", "")
            return secrets.compare_digest(sent, app.ui_token)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if not n:
                return {}
            return json.loads(self.rfile.read(min(n, 64_000)) or b"{}")

        # -------------------------------------------------------------- routes
        def do_GET(self) -> None:  # noqa: N802
            if not self._host_ok():
                return self._send(HTTPStatus.FORBIDDEN, b"forbidden", "text/plain")
            url = urlparse(self.path)
            if url.path == "/":
                page = PAGE.read_text().replace("__UI_TOKEN__", app.ui_token)
                return self._send(HTTPStatus.OK, page.encode(), "text/html; charset=utf-8")
            if url.path == "/api/status":
                return self._json(HTTPStatus.OK, app.status())
            if url.path == "/api/study":
                return self._json(HTTPStatus.OK, app.study())
            if url.path == "/callback":
                try:
                    app.finish_login(parse_qs(url.query))
                    target = "/?login=ok"
                except UserError as exc:
                    target = "/?login=failed&why=" + quote(str(exc))
                except Exception as exc:  # Kite refused the code, network, ...
                    log.error("Kite login failed: %s", type(exc).__name__)
                    target = "/?login=failed&why=" + quote(
                        f"Zerodha refused the login ({type(exc).__name__}). Try again."
                    )
                return self._send(HTTPStatus.SEE_OTHER, b"", "text/plain", {"Location": target})
            return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")

        def do_POST(self) -> None:  # noqa: N802
            if not self._host_ok() or not self._post_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "forbidden"})
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == "/api/credentials":
                    app.save_credentials(body.get("api_key", ""), body.get("api_secret", ""))
                    return self._json(HTTPStatus.OK, {"ok": True})
                if path == "/api/login":
                    return self._json(HTTPStatus.OK, {"url": app.start_login()})
                if path == "/api/download-daily":
                    app.download_daily()
                    return self._json(HTTPStatus.OK, {"ok": True})
            except UserError as exc:
                return self._json(HTTPStatus.CONFLICT, {"error": str(exc)})
            except Exception as exc:
                log.error("%s failed: %s", path, traceback.format_exc())
                return self._json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": html.escape(f"{type(exc).__name__}: {exc}")},
                )
            return self._json(HTTPStatus.NOT_FOUND, {"error": "unknown action"})

    return Handler


def serve(app: App, port: int = 8766, open_browser: bool = True) -> None:
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(app, port))
    url = f"http://127.0.0.1:{port}/"
    print(f"NSE forecasting app running at {url}  (close this window or Ctrl-C to stop)")
    if open_browser:
        import webbrowser

        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
