import numpy as np
import pytest

from forecasting.data.panel import tail_matrix
from forecasting.features.indicators import indicator_matrices
from forecasting.models import LightGBMRV
from forecasting.models.ml import FEATURES
from forecasting.tests.conftest import make_toy_panel

B = 6


@pytest.fixture(scope="module")
def panel():
    return make_toy_panel(n_sessions=40, seed=8, missing=[(31, 2, "AAA")])


@pytest.mark.parametrize("bars", [32 * B, 32 * B + 3, 33 * B + 1])
def test_prediction_features_match_training_features(panel, bars):
    """The feature row built at forecast time from a recent tail must equal the row the
    training path builds for the same origin from the full history."""
    model = LightGBMRV(tail_sessions=24).fit(panel.iloc[: 30 * B * 2])
    x_pred, a_pred = model._predict_design(panel.iloc[: bars * 2], 2)
    hist = panel.iloc[: bars * 2]
    mats = indicator_matrices(hist)
    ret2 = tail_matrix(hist, "ret2", bars)
    sessions = hist["session"].to_numpy()[::2][[bars - 1]]
    weekdays = model._weekday_of_origin(sessions, np.array([bars % B]))
    x_full, a_full = model._design(mats, ret2, np.array([bars]), 2, weekdays)
    np.testing.assert_allclose(x_pred, x_full, rtol=1e-9, equal_nan=True)
    np.testing.assert_allclose(a_pred, a_full, rtol=1e-12)
    assert x_pred.shape == (2, len(FEATURES))


def test_trains_once_per_horizon_and_forecasts_positive(panel):
    model = LightGBMRV(n_estimators=30, min_child_samples=10, min_train_rows=20)
    model = model.fit(panel.iloc[: 30 * B * 2])
    one = model.predict(panel.iloc[: (30 * B + 2) * 2], 2)
    model.predict(panel.iloc[: (30 * B + 3) * 2], 2)
    day = model.predict(panel.iloc[: 31 * B * 2], B)
    assert set(model.boosters_) == {2, B}
    assert (one["yhat"] > 0).all() and (day["yhat"] > 0).all()


def test_weekday_of_origin_uses_next_business_day_at_the_open():
    import datetime as dt

    friday = dt.date(2025, 11, 14)
    days = np.array([friday, friday], dtype=object)
    assert LightGBMRV._weekday_of_origin(days, np.array([3, 0])).tolist() == [4, 0]


@pytest.mark.parametrize("horizon_bars", [2, B])
def test_lightgbm_ignores_the_future(horizon_bars):
    """Future poisoning on a panel big enough for real (non-NaN) forecasts."""
    from forecasting.backtest.engine import Horizon, WalkForward, run_walk_forward

    panel = make_toy_panel(n_sessions=40, seed=9)
    h = Horizon("h", horizon_bars, "intraday" if horizon_bars < B else "session_open")
    wf = WalkForward(initial_train_sessions=26, test_block_sessions=4)
    model = LightGBMRV(n_estimators=20, min_child_samples=5, min_train_rows=20)
    cut = panel["ts"].iloc[33 * B * 2]
    poisoned = panel.copy()
    future = (poisoned["ts"] >= cut).to_numpy()
    for col in ("open", "high", "low", "close", "ret", "ret2", "volume"):
        values = poisoned[col].to_numpy(copy=True)
        values[future] = values[future] * 50 + 7
        poisoned[col] = values
    a = run_walk_forward(panel, model, h, wf)
    b = run_walk_forward(poisoned, model, h, wf)
    past = (a["origin_ts"] <= cut).to_numpy()
    assert np.isfinite(a.loc[past, "yhat"]).all()  # real forecasts, not a trivial pass
    np.testing.assert_array_equal(a.loc[past, "yhat"], b.loc[past, "yhat"])
    assert not np.allclose(a.loc[~past, "yhat"], b.loc[~past, "yhat"], equal_nan=True)
