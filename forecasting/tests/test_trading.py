import numpy as np
import pandas as pd
import pytest

from forecasting.tests.conftest import make_toy_panel
from forecasting.trading.costs import CostModel
from forecasting.trading.vol_target import (
    VolTargetConfig,
    session_open_to_close,
    vol_target_backtest,
)

B = 6


def test_round_trip_costs_for_one_lakh():
    c = CostModel()
    n = 100_000.0
    brokerage = 2 * 20.0  # 0.03% = 30 > cap of 20, per side
    exchange, sebi = 2 * 2.97, 2 * 0.1
    gst = 0.18 * (brokerage + exchange + sebi)
    expected = brokerage + exchange + sebi + gst + 3.0 + 25.0 + 2 * 20.0  # stamp, STT, slip
    assert float(c.round_trip(np.array([n]))[0]) == pytest.approx(expected)
    assert float(c.round_trip(np.array([0.0]))[0]) == 0.0


def test_open_to_close_uses_first_open_and_last_close():
    panel = make_toy_panel(n_sessions=3)
    r = session_open_to_close(panel)
    first = panel.iloc[0]["open"]
    last = panel.iloc[(B - 1) * 2]["close"]
    assert r.iloc[0, 0] == pytest.approx(last / first - 1)


def _heteroskedastic(n_sessions=200, seed=0):
    """Panel whose session open-to-close returns have known, time-varying vol."""
    rng = np.random.default_rng(seed)
    panel = make_toy_panel(n_sessions=n_sessions, seed=seed)
    vol = np.exp(rng.normal(np.log(0.015), 0.6, (n_sessions, 2)))  # daily vol, 1.5% typical
    ret = vol * rng.standard_normal((n_sessions, 2))
    close = np.empty((n_sessions * B, 2))
    open_ = np.full((n_sessions * B, 2), 100.0)
    close[:] = 100.0
    close[B - 1 :: B] = 100.0 * (1 + ret)
    panel = panel.assign(open=open_.ravel(), close=close.ravel())
    panel.attrs = make_toy_panel(n_sessions=1).attrs
    sessions = panel["session"].to_numpy()[:: 2 * B]
    rows = []
    for name, fc in [("perfect", vol), ("flat", np.full_like(vol, 0.015))]:
        for d, day in enumerate(sessions):
            for j, sym in enumerate(("AAA", "BBB")):
                rows.append((name, day, sym, (100 * fc[d, j]) ** 2))
    forecasts = pd.DataFrame(rows, columns=["model", "session", "symbol", "yhat"])
    return panel, forecasts


def test_better_vol_forecasts_give_better_risk_control():
    panel, forecasts = _heteroskedastic()
    cfg = VolTargetConfig(max_gross_leverage=50.0)
    m = vol_target_backtest(panel, forecasts, cfg, CostModel(slippage_bps=0)).metrics
    assert abs(m.loc["perfect", "z_std"] - 1.0) < 0.1  # returns / forecast vol ~ N(0, 1)
    assert m.loc["perfect", "z_abs_p99"] < m.loc["flat", "z_abs_p99"]
    assert abs(m.loc["perfect", "vol_vs_target"] - 1 / np.sqrt(2)) < 0.15  # 2 indep. names
    assert np.isnan(m.loc["constant", "z_std"])


def test_gross_exposure_cap_is_respected():
    panel, forecasts = _heteroskedastic(n_sessions=50)
    cfg = VolTargetConfig(target_annual_vol=1.0, max_gross_leverage=1.5)
    m = vol_target_backtest(panel, forecasts, cfg).metrics
    assert (m["avg_gross"] <= 1.5 + 1e-9).all()
