"""Tests for the held, volatility-managed basket (delivery, banded rebalancing)."""

import numpy as np
import pandas as pd
import pytest

from forecasting.features.returns import add_returns
from forecasting.tests.conftest import make_toy_panel
from forecasting.trading.costs import DELIVERY_COSTS, CostModel
from forecasting.trading.vol_managed import (
    BUY_HOLD,
    VolManagedConfig,
    close_to_close,
    overnight_ratio,
    session_data,
    trailing_correlation,
    vol_managed_backtest,
)

B = 6
SYMBOLS = ("AAA", "BBB")
FREE = CostModel(
    brokerage_rate=0, stt_sell_rate=0, exchange_rate=0, sebi_rate=0, stamp_buy_rate=0,
    slippage_bps=0,
)  # fmt: skip


def _held_panel(n_sessions=260, seed=0, vol_sigma=0.6, gap_scale=0.0):
    """Panel whose daily volatility is known and changes from day to day.

    Each session's open is the previous close (times an overnight gap when
    ``gap_scale`` > 0) and the six bar returns share that day's volatility. Returns
    (panel, forecasts): ``perfect`` forecasts the true intraday variance, ``flat``
    forecasts its average and ``noisy`` is ``perfect`` times fresh noise every day.
    """
    rng = np.random.default_rng(seed)
    s = len(SYMBOLS)
    vol = np.exp(rng.normal(np.log(0.012), vol_sigma, (n_sessions, s)))  # daily, intraday
    bar_ret = vol[:, None, :] / np.sqrt(B) * rng.standard_normal((n_sessions, B, s))
    gap = np.zeros((n_sessions, s))  # the gap after a session shares that session's volatility
    gap[1:] = gap_scale * vol[:-1] * rng.standard_normal((n_sessions - 1, s))
    log_open = np.empty((n_sessions, s))
    log_close = np.empty((n_sessions, B, s))
    last = np.full(s, np.log(500.0))
    for d in range(n_sessions):
        log_open[d] = last + gap[d]
        log_close[d] = log_open[d] + np.cumsum(bar_ret[d], axis=0)
        last = log_close[d, -1]
    close = np.exp(log_close)
    open_ = np.concatenate([np.exp(log_open)[:, None, :], close[:, :-1, :]], axis=1)
    attrs = make_toy_panel(n_sessions=1).attrs
    panel = make_toy_panel(n_sessions=n_sessions, seed=seed).drop(columns=["ret", "ret2"])
    panel = panel.assign(
        open=open_.ravel(),
        close=close.ravel(),
        high=np.maximum(open_, close).ravel(),
        low=np.minimum(open_, close).ravel(),
    )
    panel.attrs = attrs
    panel = add_returns(panel)
    sessions = panel["session"].to_numpy()[:: s * B]
    variance = (100 * vol) ** 2
    noise = np.exp(rng.normal(0, 0.5, variance.shape))
    kinds = {
        "perfect": variance,
        "flat": np.full_like(variance, variance.mean()),
        "noisy": variance * noise,
    }
    rows = [
        (name, sessions[d], sym, fc[d, j])
        for name, fc in kinds.items()
        for d in range(60, n_sessions)
        for j, sym in enumerate(SYMBOLS)
    ]
    return panel, pd.DataFrame(rows, columns=["model", "session", "symbol", "yhat"])


def test_delivery_costs_for_one_lakh():
    c = CostModel.from_yaml(DELIVERY_COSTS)
    n = np.array([100_000.0])
    fees = 2.97 + 0.1  # exchange + SEBI; no brokerage on delivery
    buy = fees * 1.18 + 100.0 + 15.0 + 20.0  # + STT, stamp duty, slippage
    sell = fees * 1.18 + 100.0 + 15.34 + 20.0  # + STT, depository charge, slippage
    assert float(c.side(n, buy=True)[0]) == pytest.approx(buy)
    assert float(c.side(n, buy=False)[0]) == pytest.approx(sell)
    assert float(c.side(np.array([0.0]), buy=False)[0]) == 0.0  # no sale, no depository fee


def test_session_prices_skip_missing_bars_and_carry_the_last_close():
    panel = make_toy_panel(
        n_sessions=4,
        missing=[(1, 0, "AAA"), (1, B - 1, "AAA")] + [(2, k, "BBB") for k in range(B)],
    )
    data = session_data(panel)
    rows = panel[(panel["session"] == data.sessions[1]) & (panel["symbol"] == "AAA")]
    assert data.open[1, 0] == pytest.approx(rows["open"].iloc[1])
    assert data.close[1, 0] == pytest.approx(rows["close"].iloc[B - 2])
    assert np.isnan(data.rv[1, 0])  # a session with a missing bar has no RV
    assert np.isnan(data.open[2, 1])  # nothing traded: no price to trade at ...
    assert data.close[2, 1] == data.close[1, 1]  # ... and the last close is carried


def test_overnight_ratio_matches_the_past_and_ignores_the_future():
    panel, _ = _held_panel(n_sessions=120, gap_scale=1.0, vol_sigma=0.1)
    data = session_data(panel)
    ratio = overnight_ratio(data, min_history=20)
    assert (ratio[:21] == 1.0).all()  # too little history: intraday volatility only
    cc = close_to_close(data)
    t = 100
    expected = (cc[1:t] ** 2).sum(axis=0) / data.rv[1:t].sum(axis=0)
    np.testing.assert_allclose(ratio[t], expected)
    assert (np.abs(ratio[t] - 2.0) < 0.6).all()  # gaps as large as the day: twice the RV
    no_gaps = overnight_ratio(session_data(_held_panel(n_sessions=120, vol_sigma=0.1)[0]))
    assert (np.abs(no_gaps[t] - 1.0) < 0.3).all()

    later = panel["session"] >= data.sessions[t]
    poisoned = panel.copy()
    for col in ("open", "close", "ret", "ret2"):
        poisoned.loc[later, col] = poisoned.loc[later, col] * 3 + 1
    again = overnight_ratio(session_data(poisoned), min_history=20)
    np.testing.assert_array_equal(ratio[: t + 1], again[: t + 1])


def test_correlation_is_cautious_without_history_and_shrinks_to_its_average():
    rng = np.random.default_rng(1)
    common = rng.standard_normal((200, 1))
    cc = common + rng.standard_normal((200, 3)) * np.array([0.5, 1.0, 2.0])
    assert (trailing_correlation(cc, 10, min_history=20) == 1.0).all()
    raw = trailing_correlation(cc, 200, window=60, shrink=0.0)
    np.testing.assert_allclose(raw, np.corrcoef(cc[140:200].T))
    flat = trailing_correlation(cc, 200, window=60, shrink=1.0)
    off = flat[~np.eye(3, dtype=bool)]
    np.testing.assert_allclose(off, off[0])
    assert off[0] == pytest.approx((raw.sum() - 3) / 6)
    np.testing.assert_allclose(np.diag(flat), 1.0)


def test_buy_and_hold_trades_only_on_the_first_day():
    panel, forecasts = _held_panel(n_sessions=100)
    d = vol_managed_backtest(panel, forecasts).detail[BUY_HOLD]
    assert d["orders"].iloc[0] == len(SYMBOLS)
    assert (d["orders"].iloc[1:] == 0).all()
    assert d["cost_frac"].iloc[0] > 0 and (d["cost_frac"].iloc[1:] == 0).all()
    assert 0.98 < d["gross"].iloc[0] <= 1.0


def test_a_wider_band_trades_less_and_costs_less():
    panel, forecasts = _held_panel(vol_sigma=0.1)  # volatility that moves a little each day
    noisy = forecasts[forecasts["model"] == "noisy"]
    m = vol_managed_backtest(panel, noisy, bands=[0.0, 0.25, 0.5]).metrics()
    turnover = m.loc[["noisy @ 0%", "noisy @ 25%", "noisy @ 50%"], "turnover_x"]
    costs = m.loc[turnover.index, "costs_pct"]
    assert turnover.is_monotonic_decreasing and turnover.iloc[0] > 3 * turnover.iloc[-1]
    assert costs.is_monotonic_decreasing and costs.iloc[0] > 3 * costs.iloc[-1]


def test_nothing_is_traded_while_weights_stay_inside_the_band():
    panel, forecasts = _held_panel(n_sessions=120, vol_sigma=0.0)  # constant volatility
    flat = forecasts[forecasts["model"] == "flat"]
    d = vol_managed_backtest(panel, flat, VolManagedConfig(band=0.5)).detail["flat"]
    assert d["orders"].iloc[0] == len(SYMBOLS)
    assert d["orders"].iloc[1:].sum() == 0


def test_never_invests_more_than_the_cash_allows():
    panel, forecasts = _held_panel(n_sessions=120)
    cfg = VolManagedConfig(target_annual_vol=5.0, band=0.0)  # wants far more than 1x
    result = vol_managed_backtest(panel, forecasts, cfg)
    for d in result.detail.values():
        assert (d["gross"] <= 1.0 + 1e-12).all()
    assert result.metrics().loc["perfect", "avg_gross"] > 0.95


def test_borrowing_is_allowed_only_up_to_max_gross():
    panel, forecasts = _held_panel(n_sessions=120, vol_sigma=0.1)
    cfg = VolManagedConfig(target_annual_vol=5.0, max_gross=1.5, band=0.0)
    result = vol_managed_backtest(panel, forecasts, cfg)
    d = result.detail["perfect"]
    assert 1.4 < d["gross"].iloc[0] < 1.6  # bought up to the cap at the open (measured at close)
    assert (result.detail[BUY_HOLD]["gross"] <= 1.0 + 1e-12).all()  # the benchmark never borrows


def test_good_forecasts_keep_risk_on_target():
    panel, forecasts = _held_panel(n_sessions=400)
    cfg = VolManagedConfig(max_gross=50.0, band=0.0, whole_shares=False)
    result = vol_managed_backtest(panel, forecasts, cfg, FREE)
    m = result.metrics()
    assert abs(m.loc["perfect", "z_std"] - 1.0) < 0.12  # returns / forecast vol ~ N(0, 1)
    assert abs(m.loc["perfect", "vol_vs_target"] - 1.0) < 0.15
    assert np.isnan(m.loc[BUY_HOLD, "z_std"])  # no forecast behind buy and hold

    def kurtosis(name):
        z = result.detail[name]["z"].dropna()
        return float(((z - z.mean()) ** 4).mean() / z.var(ddof=0) ** 2)

    assert kurtosis("perfect") < 4.0 < kurtosis("flat")  # flat forecasts: fat tails


def test_overnight_gaps_are_priced_into_the_position_size():
    panel, forecasts = _held_panel(n_sessions=400, gap_scale=1.0)  # half the variance overnight
    cfg = VolManagedConfig(max_gross=50.0, band=0.0, whole_shares=False)
    m = vol_managed_backtest(panel, forecasts, cfg, FREE).metrics()
    assert abs(m.loc["perfect", "z_std"] - 1.0) < 0.15
    assert abs(m.loc["perfect", "vol_vs_target"] - 1.0) < 0.2


def test_positions_ignore_the_future():
    panel, forecasts = _held_panel(n_sessions=120)
    sessions = panel["session"].to_numpy()[:: len(SYMBOLS) * B]
    cut = sessions[90]
    poisoned = panel.copy()
    later = poisoned["session"] >= cut
    for col in ("open", "high", "low", "close", "ret", "ret2"):
        poisoned.loc[later, col] = poisoned.loc[later, col] * 2 + 5
    changed = forecasts.copy()
    changed.loc[changed["session"] >= cut, "yhat"] *= 7
    a = vol_managed_backtest(panel, forecasts)
    b = vol_managed_backtest(poisoned, changed)
    for name, d in a.detail.items():
        before = d.index < cut
        cols = ["ret", "equity", "gross", "orders", "cost_frac", "forecast_vol", "z"]
        pd.testing.assert_frame_equal(d.loc[before, cols], b.detail[name].loc[before, cols])


def test_period_metrics_cover_only_their_sessions():
    panel, forecasts = _held_panel(n_sessions=160)
    result = vol_managed_backtest(panel, forecasts)
    days = list(result.daily.index)
    first, second = days[:60], days[60:]
    m1, m2, m = result.metrics(first), result.metrics(second), result.metrics()
    assert m1.loc["perfect", "sessions"] == 60 and m2.loc["perfect", "sessions"] == 40
    assert m.loc["perfect", "sessions"] == 100
    assert m2.loc[BUY_HOLD, "costs_pct"] == 0.0  # the one purchase was in the first period
    md = result.to_markdown({"first": first, "second": second})
    assert "### first (60 sessions)" in md and "### second (40 sessions)" in md


def test_model_without_forecasts_is_left_out():
    panel, forecasts = _held_panel(n_sessions=100)
    dead = forecasts[forecasts["model"] == "flat"].assign(model="dead", yhat=np.nan)
    m = vol_managed_backtest(panel, pd.concat([forecasts, dead])).metrics()
    assert "dead" not in m.index and {"perfect", "flat", BUY_HOLD} <= set(m.index)


def test_a_model_that_starts_late_is_flagged():
    panel, forecasts = _held_panel(n_sessions=100)
    sessions = sorted(forecasts["session"].unique())
    late = forecasts[forecasts["model"] == "flat"].assign(model="late")
    late = late[late["session"] >= sessions[10]]
    result = vol_managed_backtest(panel, pd.concat([forecasts, late]))
    d = result.detail["late"]
    assert (d["gross"].iloc[:10] == 0).all() and d["gross"].iloc[10] > 0
    assert "`late` after 10 sessions" in result.to_markdown()
    assert "Started late" not in vol_managed_backtest(panel, forecasts).to_markdown()


def test_band_must_be_below_one():
    with pytest.raises(ValueError, match="band"):
        VolManagedConfig(band=1.0)
