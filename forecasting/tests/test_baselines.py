import numpy as np
import pytest

from forecasting.data.panel import to_matrix
from forecasting.models import HistoricalMean, RandomWalk, SeasonalNaive, build_model
from forecasting.tests.conftest import make_toy_panel

B = 6  # toy bars per session


def _history(panel, bars):
    return panel.iloc[: bars * 2]


def test_random_walk_is_last_h_bars(toy_panel):
    hist = _history(toy_panel, 3 * B + 2)
    x = to_matrix(hist, "ret2")
    fc = RandomWalk().predict(hist, 4)
    np.testing.assert_allclose(fc["yhat"], x[-4:].sum(axis=0))
    assert fc["symbol"].tolist() == ["AAA", "BBB"]


def test_random_walk_skips_missing_bars():
    panel = make_toy_panel(missing=[(3, 1, "AAA")])  # blanks returns at bars 19 and 20
    hist = _history(panel, 3 * B + 3)
    x = to_matrix(hist, "ret2")[:, 0]
    observed = x[~np.isnan(x)]
    assert RandomWalk().predict(hist, 3)["yhat"].iloc[0] == pytest.approx(observed[-3:].sum())


def test_historical_mean_uses_train_window_only(toy_panel):
    train = _history(toy_panel, 5 * B)
    m = HistoricalMean().fit(train)
    expected = np.nanmean(to_matrix(train, "ret2"), axis=0) * 6
    np.testing.assert_allclose(m.predict(toy_panel, 6)["yhat"], expected)
    rolling = HistoricalMean(window_sessions=2).fit(train)
    expected = np.nanmean(to_matrix(train, "ret2")[-2 * B :], axis=0) * 6
    np.testing.assert_allclose(rolling.predict(toy_panel, 6)["yhat"], expected)


def test_seasonal_naive_uses_same_slots_previous_sessions(toy_panel):
    x = to_matrix(toy_panel, "ret2")
    origin = 4 * B + 2  # session 4, slot 2
    hist = _history(toy_panel, origin)
    one = SeasonalNaive(n_sessions=1).predict(hist, 3)["yhat"]
    np.testing.assert_allclose(one, x[3 * B + 2 : 3 * B + 5].sum(axis=0))
    two = SeasonalNaive(n_sessions=2).predict(hist, 3)["yhat"]
    expected = (x[3 * B + 2 : 3 * B + 5].sum(axis=0) + x[2 * B + 2 : 2 * B + 5].sum(axis=0)) / 2
    np.testing.assert_allclose(two, expected)


def test_seasonal_naive_equals_random_walk_for_full_session_on_clean_data(toy_panel):
    hist = _history(toy_panel, 5 * B)
    np.testing.assert_allclose(
        SeasonalNaive().predict(hist, B)["yhat"], RandomWalk().predict(hist, B)["yhat"]
    )


def test_seasonal_naive_rejects_multi_session_horizon(toy_panel):
    with pytest.raises(ValueError):
        SeasonalNaive().predict(toy_panel, B + 1)


def test_registry():
    m = build_model("seasonal_naive", "sn5", {"n_sessions": 5})
    assert isinstance(m, SeasonalNaive) and m.name == "sn5" and m.n_sessions == 5
    with pytest.raises(KeyError):
        build_model("garch")
