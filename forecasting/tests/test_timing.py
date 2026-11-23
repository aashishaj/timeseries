"""Invest / stay-out rules and the after-tax account simulation."""

import json

import numpy as np
import pandas as pd
import pytest

from forecasting.timing.account import run_account
from forecasting.timing.rules import daily_variance, har_forecast, signals, trend_signal
from forecasting.timing.tax import TaxRules
from forecasting.trading.costs import CostModel

FREE = CostModel(
    brokerage_rate=0, brokerage_cap_inr=0, stt_sell_rate=0, exchange_rate=0, sebi_rate=0,
    stamp_buy_rate=0, gst_rate=0, slippage_bps=0,
)  # fmt: skip
NO_TAX = TaxRules(stcg_rate=0, ltcg_rate=0, cess_rate=0)


def prices(close, start="2020-01-01", gap=0.0):
    """Daily bars whose open is the previous close (times 1 + gap)."""
    close = np.asarray(close, dtype=float)
    idx = pd.bdate_range(start, periods=len(close))
    opens = np.r_[close[0], close[:-1] * (1 + gap)]
    return pd.DataFrame(
        {"open": opens, "high": np.maximum(opens, close) * 1.005,
         "low": np.minimum(opens, close) * 0.995, "close": close, "volume": 1.0},
        index=idx,
    )  # fmt: skip


def test_buy_and_hold_with_long_term_tax_matches_a_hand_calculation():
    px = prices(np.linspace(100, 200, 600))  # doubles over ~2.3 years
    start, end = px.index[1], px.index[-1]
    res = run_account(px, pd.Series(1.0, index=px.index), start, end, costs=FREE)
    buy = px["open"].iloc[1]
    qty = int(1_000_000 // buy)
    gain = qty * (px["close"].iloc[-1] - buy)
    tax = (gain - 125_000) * 0.125 * 1.04
    expected = 1_000_000 - qty * buy + qty * px["close"].iloc[-1] - tax
    assert res.value.iloc[-1] == pytest.approx(expected, rel=1e-9)
    assert len(res.trades) == 1 and res.trades[0].exit == "end"


def test_decision_at_a_close_is_executed_at_the_next_open():
    px = prices(np.full(30, 100.0))
    sig = pd.Series(0.0, index=px.index)
    sig.iloc[10:] = 1.0  # decided at the close of day 10
    res = run_account(px, sig, px.index[1], px.index[-1], costs=FREE, rules=NO_TAX)
    assert not res.invested.loc[: px.index[10]].any()
    assert res.invested.loc[px.index[11]]  # bought at day 11's open


def test_charges_reduce_the_result():
    px = prices(np.full(50, 100.0))
    flip = pd.Series(np.tile([1.0, 0.0], 25), index=px.index)  # in and out every day
    paid = run_account(px, flip, px.index[1], px.index[-1], rules=NO_TAX)
    free = run_account(px, flip, px.index[1], px.index[-1], costs=FREE, rules=NO_TAX)
    assert free.value.iloc[-1] == pytest.approx(1_000_000, abs=200)  # whole-share rounding
    assert paid.value.iloc[-1] < 1_000_000 - 20 * 1_000_000 * 0.002  # >0.2% per round trip
    assert paid.charges > 0


def test_tax_is_paid_after_31_march_selling_shares_if_cash_is_short():
    days = pd.bdate_range("2023-06-01", "2024-06-28")
    close = np.linspace(100, 160, len(days))
    px = prices(close, start="2023-06-01")
    sig = pd.Series(1.0, index=px.index)
    sig.loc["2023-12-01"] = 0.0  # one exit in December: a short-term gain, then back in
    res = run_account(px, sig, px.index[1], px.index[-1], costs=FREE)
    (fy, tax), *_ = res.taxes
    assert fy == 2023 and tax > 0
    assert res.value.diff().abs().max() < 0.02 * res.value.max()  # no cliff on paying tax


def test_take_profit_sells_once_two_percent_is_left_after_charges_and_tax():
    px = prices(np.r_[np.full(5, 100.0), np.linspace(100, 110, 40)])
    sig = pd.Series(1.0, index=px.index)
    res = run_account(px, sig, px.index[1], px.index[-1], take_profit=0.02)
    first = res.trades[0]
    assert first.exit == "take-profit"
    after_tax = first.gain - max(0.0, first.gain) * 0.20 * 1.04
    assert 0.02 <= after_tax / first.cost < 0.03
    assert len(res.trades) == 1  # signal never went False: no fresh entry, stays out


def test_stop_loss_exits_after_a_two_percent_fall_and_waits_for_a_fresh_entry():
    close = np.r_[np.full(5, 100.0), [99, 98.5, 97.9, 97, 96], np.full(10, 96.0)]
    px = prices(close)
    sig = pd.Series(1.0, index=px.index)
    sig.iloc[12] = 0.0  # signal goes out and back in: a fresh entry at day 13's close
    res = run_account(px, sig, px.index[1], px.index[-1], costs=FREE, rules=NO_TAX, stop_loss=0.02)
    assert res.trades[0].exit == "stop-loss"
    assert res.trades[0].sold == px.index[8].date()  # close of day 7 <= 98 -> sell day 8
    assert not res.invested.iloc[8:13].any() and res.invested.iloc[13]


def test_trend_signal_compares_the_close_with_its_average():
    close = pd.Series(np.r_[np.full(200, 100.0), 101.0, 99.0])
    t = trend_signal(close)
    assert t.iloc[:199].isna().all() and t.iloc[200] == 1.0 and t.iloc[201] == 0.0


def test_volatility_forecast_tracks_volatility_and_ignores_the_future():
    rng = np.random.default_rng(0)
    vol = np.r_[np.full(500, 0.01), np.full(300, 0.03)]
    close = 100 * np.exp(np.cumsum(vol * rng.standard_normal(len(vol))))
    px = prices(close)
    px["high"] = px[["open", "close"]].max(axis=1) * (1 + vol)  # ranges scale with volatility
    px["low"] = px[["open", "close"]].min(axis=1) * (1 - vol)
    fc = har_forecast(daily_variance(px))
    calm, wild = fc.iloc[400:500].mean(), fc.iloc[700:].mean()
    assert wild > 4 * calm  # 3x the daily volatility is 9x the variance
    poisoned = px.copy()
    poisoned.iloc[650:] *= 1.5
    poisoned.iloc[650:, poisoned.columns.get_loc("high")] *= 1.2
    again = har_forecast(daily_variance(poisoned))
    pd.testing.assert_series_equal(fc.iloc[:649], again.iloc[:649])


def test_all_signals_ignore_the_future():
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(0.015 * rng.standard_normal(1200)))
    px = prices(close)
    cut = 900
    poisoned = px.copy()
    poisoned.iloc[cut:] = poisoned.iloc[cut:] * 3 + 7
    a, b = signals(px), signals(poisoned)
    pd.testing.assert_frame_equal(a.iloc[: cut - 1], b.iloc[: cut - 1])
    assert a["V"].notna().sum() > 300 and a["TV"].dropna().isin([0.0, 1.0]).all()


def _daily_folder(tmp_path, symbols=("AAA", "BBB", "NIFTYBEES"), years=10, seed=3):
    from forecasting.data.kite_backfill import DAILY_MARKER

    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=years * 250)
    for sym in symbols:
        r = 0.0003 + 0.015 * rng.standard_normal(len(idx))
        close = 100 * np.exp(np.cumsum(r))
        px = prices(close, start="2010-01-01")
        px.index = idx
        px.index.name = "date"
        px.reset_index().assign(date=lambda d: d["date"].dt.date).to_parquet(
            tmp_path / f"{sym}.parquet", index=False
        )
    (tmp_path / DAILY_MARKER).write_text("test\n")
    return tmp_path


def test_study_picks_a_finalist_on_development_and_logs_the_lockbox(tmp_path):
    from forecasting.timing.study import RULES, run_study

    data = _daily_folder(tmp_path / "daily")
    out = tmp_path / "results" / "timing"
    with pytest.raises(SystemExit):
        run_study(data, out, "final")  # no finalist yet
    dev = run_study(data, out, "dev")
    table = pd.read_csv(out / "metrics_dev.csv")
    assert set(table["rule"]) == set(RULES) and table["instrument"].nunique() == 3
    end_dev = pd.Timestamp(table["end"].max())
    finalist = json.loads((out / "finalist.json").read_text())["rule"]
    assert finalist in RULES and finalist != "B"
    assert "Rules compared with buy and hold" in dev.read_text()

    run_study(data, out, "final")
    lock = pd.read_csv(out / "metrics_final.csv")
    assert set(lock["rule"]) == {"B", finalist}
    assert pd.Timestamp(lock["start"].min()) > end_dev  # lockbox starts after development
    log = (tmp_path / "results" / "lockbox_log.csv").read_text()
    assert log.count("timing") == 1
