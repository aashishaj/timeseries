"""Read-only loader for OHLCV candles exported by the live Kite pipeline.

This module only *reads* Parquet files. It never writes, moves or deletes
anything under the export root, and it does not import any code from the live
pipeline: the export directory is the only contract between the two projects.

Expected layout: two directory levels, symbol then date. Hive-style names
(``key=value``, as written by Spark/pyarrow) and plain names both work::

    <root>/symbol=RELIANCE/date=2025-01-02/part-0000.parquet
    <root>/RELIANCE/2025-01-02/candles.parquet

Each file holds candles for one symbol and day with (at least) a timestamp and
OHLCV columns. Extra columns (latency instrumentation etc.) are ignored, and a
timestamp column called ``date`` does not clash with the ``date=`` directory.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow.dataset as ds

from forecasting.data.calendar import IST, to_date

log = logging.getLogger(__name__)

CANONICAL_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class ExportSchema:
    """How the exported candles are laid out on disk and how to interpret them."""

    date_format: str = "%Y-%m-%d"  # format of the date directory value
    columns: dict[str, str] = field(
        default_factory=lambda: {
            "timestamp": "timestamp",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    )
    timestamp_timezone: str = IST  # used only when timestamps are tz-naive
    timestamp_unit: Literal["s", "ms", "us", "ns"] = "ms"  # used only for integer epochs (UTC)
    timestamp_label: Literal["start", "end"] = "start"  # does the stamp mark candle start or end?
    source_interval_minutes: int = 1  # candle length in the export

    @classmethod
    def from_dict(cls, raw: dict | None) -> ExportSchema:
        raw = dict(raw or {})
        columns = cls().columns | dict(raw.pop("columns", None) or {})
        allowed = {f for f in cls.__dataclass_fields__ if f != "columns"}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"unknown export schema keys: {sorted(unknown)}")
        return cls(columns=columns, **raw)


def _partition_value(name: str) -> str:
    return name.split("=", 1)[1] if "=" in name else name


def _visible_dirs(path: Path) -> list[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir() and not p.name.startswith((".", "_")))


def discover_files(root: str | Path, schema: ExportSchema | None = None) -> pd.DataFrame:
    """Index the export: one row per Parquet file with its ``symbol`` and ``date``."""
    schema = schema or ExportSchema()
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"candle export root not found: {root}")
    rows: list[tuple[str, dt.date, str]] = []
    unparsed: list[str] = []
    for sym_dir in _visible_dirs(root):
        symbol = _partition_value(sym_dir.name)
        for date_dir in _visible_dirs(sym_dir):
            try:
                day = dt.datetime.strptime(_partition_value(date_dir.name), schema.date_format)
            except ValueError:
                unparsed.append(str(date_dir))
                continue
            for f in sorted(date_dir.glob("*.parquet")):
                if not f.name.startswith((".", "_")):
                    rows.append((symbol, day.date(), str(f)))
    if unparsed:
        log.warning(
            "skipped %d directories whose name does not match date_format=%r, e.g. %s",
            len(unparsed),
            schema.date_format,
            unparsed[0],
        )
    return pd.DataFrame(rows, columns=["symbol", "date", "path"])


def list_symbols(root: str | Path, schema: ExportSchema | None = None) -> list[str]:
    """Symbols present in the export."""
    return sorted(discover_files(root, schema)["symbol"].unique().tolist())


def _to_ist(ts: pd.Series, schema: ExportSchema) -> pd.Series:
    if pd.api.types.is_integer_dtype(ts):
        ts = pd.to_datetime(ts, unit=schema.timestamp_unit, utc=True)
    elif not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize(schema.timestamp_timezone)
    return ts.dt.tz_convert(IST).dt.as_unit("ns")


def _empty_candles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": pd.Series(dtype=str),
            "ts": pd.Series(dtype=f"datetime64[ns, {IST}]"),
            **{c: pd.Series(dtype=float) for c in CANONICAL_COLUMNS},
        }
    )


def load_candles(
    root: str | Path,
    symbols: Sequence[str] | None = None,
    start: dt.date | str | None = None,
    end: dt.date | str | None = None,
    schema: ExportSchema | None = None,
) -> pd.DataFrame:
    """Load candles for ``symbols`` between ``start`` and ``end`` (inclusive dates, IST).

    Returns a long frame with columns ``symbol, ts, open, high, low, close, volume``
    where ``ts`` is the tz-aware (Asia/Kolkata) *start* of each candle, sorted by
    symbol then time. Duplicates and out-of-hours rows are kept here; cleaning
    happens in :func:`forecasting.data.resample.build_bars` so it can be reported.
    """
    schema = schema or ExportSchema()
    files = discover_files(root, schema)
    if symbols is not None:
        files = files[files["symbol"].isin(list(symbols))]
    # One day of padding either side: the exact cut is made on the timestamp below.
    if start is not None:
        files = files[files["date"] >= to_date(start) - dt.timedelta(days=1)]
    if end is not None:
        files = files[files["date"] <= to_date(end) + dt.timedelta(days=1)]
    if files.empty:
        return _empty_candles()

    col = schema.columns
    wanted = [col["timestamp"], *(col[c] for c in CANONICAL_COLUMNS)]
    frames = []
    for symbol, group in files.groupby("symbol", sort=True):
        dataset = ds.dataset(group["path"].tolist(), format="parquet")
        missing = [c for c in wanted if c not in dataset.schema.names]
        if missing:
            raise KeyError(
                f"{symbol}: export is missing columns {missing}; available: "
                f"{dataset.schema.names}. Map them under data.schema.columns in the config."
            )
        df = dataset.to_table(columns=wanted).to_pandas()
        frames.append(
            pd.DataFrame(
                {
                    "symbol": pd.Series(symbol, index=df.index, dtype=str),
                    "ts": _to_ist(df[col["timestamp"]], schema),
                    **{c: df[col[c]].astype(float) for c in CANONICAL_COLUMNS},
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    if schema.timestamp_label == "end":
        out["ts"] = out["ts"] - pd.Timedelta(minutes=schema.source_interval_minutes)
    day = out["ts"].dt.date
    keep = pd.Series(True, index=out.index)
    if start is not None:
        keep &= day >= to_date(start)
    if end is not None:
        keep &= day <= to_date(end)
    out = out.loc[keep]
    return out.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)
