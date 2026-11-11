import hashlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from forecasting.data.loader import ExportSchema, list_symbols, load_candles
from forecasting.data.synthetic import write_export


def _snapshot(root):
    return {
        str(p): (p.stat().st_mtime_ns, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _write(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)


def _one_day(ts, **cols):
    n = len(ts)
    base = {"open": np.full(n, 100.0), "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10}
    return pd.DataFrame({"timestamp": ts, **base, **cols})


def test_loads_synthetic_export(synthetic_export):
    df = load_candles(synthetic_export["root"], synthetic_export["symbols"])
    assert list(df.columns) == ["symbol", "ts", "open", "high", "low", "close", "volume"]
    assert str(df["ts"].dt.tz) == "Asia/Kolkata"
    assert set(df["symbol"]) == set(synthetic_export["symbols"])
    assert df.groupby("symbol")["ts"].apply(lambda s: s.is_monotonic_increasing).all()
    assert list_symbols(synthetic_export["root"]) == sorted(synthetic_export["symbols"])


def test_loader_is_read_only(synthetic_export):
    root = synthetic_export["root"]
    before = _snapshot(root)
    load_candles(root, start="2025-02-01", end="2025-02-28")
    list_symbols(root)
    assert _snapshot(root) == before


def test_filters_symbols_and_dates(synthetic_export):
    df = load_candles(synthetic_export["root"], ["INFY"], start="2025-02-03", end="2025-02-04")
    assert set(df["symbol"]) == {"INFY"}
    assert set(df["ts"].dt.date.astype(str)) == {"2025-02-03", "2025-02-04"}


def test_directory_layout_and_kite_style_date_column(tmp_path):
    ts = pd.date_range("2025-01-02 09:15", periods=3, freq="1min")
    # Kite's historical API calls the timestamp `date`, which collides with date=... dirs
    _write(
        tmp_path / "SBIN" / "2025-01-02" / "c.parquet",
        _one_day(ts).rename(columns={"timestamp": "date"}),
    )
    schema = ExportSchema.from_dict({"columns": {"timestamp": "date"}})
    df = load_candles(tmp_path, schema=schema)
    assert len(df) == 3 and set(df["symbol"]) == {"SBIN"}
    assert df["ts"].iloc[0] == pd.Timestamp("2025-01-02 09:15", tz="Asia/Kolkata")


def test_hive_files_may_also_contain_symbol_column(tmp_path):
    ts = pd.date_range("2025-01-02 09:15", periods=2, freq="1min")
    _write(
        tmp_path / "symbol=TCS" / "date=2025-01-02" / "part-0.parquet",
        _one_day(ts, symbol="TCS", latency_ms=3.0),
    )
    (tmp_path / "_SUCCESS").write_text("")  # Spark marker files are ignored
    df = load_candles(tmp_path)
    assert len(df) == 2 and set(df["symbol"]) == {"TCS"}


def test_utc_and_epoch_timestamps_convert_to_ist(tmp_path):
    utc = pd.date_range("2025-01-02 03:45", periods=2, freq="1min", tz="UTC")
    _write(tmp_path / "A" / "2025-01-02" / "x.parquet", _one_day(utc))
    epoch = (utc.as_unit("ms").asi8).astype("int64")
    _write(tmp_path / "B" / "2025-01-02" / "x.parquet", _one_day(epoch))
    df = load_candles(tmp_path)
    expected = pd.Timestamp("2025-01-02 09:15", tz="Asia/Kolkata")
    assert (df.groupby("symbol")["ts"].first() == expected).all()


def test_timestamp_label_end_is_shifted_to_start(tmp_path):
    ts = pd.date_range("2025-01-02 09:16", periods=2, freq="1min")
    _write(tmp_path / "A" / "2025-01-02" / "x.parquet", _one_day(ts))
    df = load_candles(tmp_path, schema=ExportSchema(timestamp_label="end"))
    assert df["ts"].iloc[0] == pd.Timestamp("2025-01-02 09:15", tz="Asia/Kolkata")


def test_missing_column_has_helpful_error(tmp_path):
    ts = pd.date_range("2025-01-02 09:15", periods=2, freq="1min")
    _write(tmp_path / "A" / "2025-01-02" / "x.parquet", _one_day(ts).drop(columns="volume"))
    with pytest.raises(KeyError, match="missing columns"):
        load_candles(tmp_path)


def test_synthetic_writer_refuses_foreign_directory(tmp_path):
    (tmp_path / "real_export_file.parquet").write_text("not mine")
    with pytest.raises(FileExistsError):
        write_export(pd.DataFrame(), tmp_path)
