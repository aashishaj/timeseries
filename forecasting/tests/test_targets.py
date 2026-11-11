import numpy as np

from forecasting.backtest.targets import realized_variance


def test_rv_is_forward_window_sum():
    x = np.arange(20, dtype=float).reshape(10, 2)
    y = realized_variance(x, np.array([0, 3, 7]), 3)
    np.testing.assert_allclose(y[0], x[0:3].sum(axis=0))
    np.testing.assert_allclose(y[1], x[3:6].sum(axis=0))
    np.testing.assert_allclose(y[2], x[7:10].sum(axis=0))


def test_rv_nan_when_window_incomplete():
    x = np.ones((10, 2))
    x[4, 1] = np.nan
    y = realized_variance(x, np.array([2, 5, 8]), 3)
    assert y[0, 0] == 3 and np.isnan(y[0, 1])  # missing bar inside window
    assert (y[1] == 3).all()
    assert np.isnan(y[2]).all()  # runs past the end of the data
