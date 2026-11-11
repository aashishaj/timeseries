import math

import numpy as np
import pytest

from forecasting.backtest.metrics import (
    default_hac_lag,
    diebold_mariano,
    directional_accuracy,
    mase,
    newey_west_variance,
    qlike,
)


def test_qlike_zero_at_truth_and_positive_elsewhere():
    y = np.array([1.0, 2.0, 0.5])
    np.testing.assert_allclose(qlike(y, y), 0.0, atol=1e-15)
    assert (qlike(y, y * 1.3) > 0).all() and (qlike(y, y * 0.7) > 0).all()


def test_qlike_penalises_under_prediction_more():
    assert qlike(1.0, 0.5) > qlike(1.0, 1.5)


def test_best_constant_qlike_forecast_is_the_mean():
    y = np.random.default_rng(0).gamma(2.0, 1.0, 5000)
    grid = np.linspace(0.5 * y.mean(), 1.5 * y.mean(), 201)
    best = grid[np.argmin([qlike(y, c).mean() for c in grid])]
    assert best == pytest.approx(y.mean(), rel=0.01)


def test_mase_is_one_for_naive():
    y, naive, model = np.array([1.0, 2, 3]), np.array([2.0, 2, 2]), np.array([1.0, 2, 4])
    assert mase(y, naive, naive) == 1.0
    assert mase(y, model, naive) == pytest.approx(0.5)


def test_directional_accuracy():
    assert directional_accuracy([1, -1, 2, 0], [0.5, 0.2, 3, -9]) == pytest.approx(2 / 3)


def test_newey_west_lag0_is_variance():
    x = np.random.default_rng(1).normal(size=500)
    assert newey_west_variance(x, 0) == pytest.approx(x.var())


def test_diebold_mariano():
    rng = np.random.default_rng(2)
    base = rng.gamma(2.0, 1.0, 2000)
    stat, p = diebold_mariano(base * 0.8, base, lag=5)
    assert stat < -5 and p < 1e-6
    noise = rng.normal(0, 1, 2000)
    stat, p = diebold_mariano(base + noise, base + rng.permutation(noise), lag=5)
    assert abs(stat) < 4 and p > 1e-4
    assert all(math.isnan(v) for v in diebold_mariano(base, base, lag=5))


def test_hac_lag_respects_overlap():
    assert default_hac_lag(100, overlap=0) == 4
    assert default_hac_lag(100, overlap=9) == 9
