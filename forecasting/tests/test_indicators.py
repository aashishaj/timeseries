import numpy as np
import pytest

from forecasting.data.panel import to_matrix
from forecasting.features import assert_no_lookahead
from forecasting.features.indicators import (
    garman_klass,
    indicator_matrices,
    parkinson,
    previous_session_rv,
    rolling_mean,
    rsi,
)
from forecasting.tests.conftest import make_toy_panel


def _panel():
    return make_toy_panel(n_sessions=30, missing=[(4, 2, "AAA"), (9, 0, "BBB")], seed=4)


NAMES = list(indicator_matrices(make_toy_panel(n_sessions=3)))


@pytest.mark.parametrize("name", NAMES)
def test_every_indicator_is_causal(name):
    assert_no_lookahead(lambda p: indicator_matrices(p)[name].ravel(), _panel(), n_cuts=8)


@pytest.mark.parametrize("name", NAMES)
def test_tail_computation_matches_full_history(name):
    """Finite windows: a recent tail gives the same last value as the full history."""
    panel = _panel()
    s, b = 2, 6
    for end_bars in (25 * b + 3, 30 * b):
        full = indicator_matrices(panel.iloc[: end_bars * s])[name][-1]
        start = end_bars - (24 * b + end_bars % b)  # session-aligned 24-session tail
        tail = indicator_matrices(panel.iloc[start * s : end_bars * s])[name][-1]
        np.testing.assert_allclose(tail, full, rtol=1e-9, equal_nan=True)


def test_rolling_mean_matches_manual():
    x = np.arange(10.0)[:, None]
    np.testing.assert_allclose(rolling_mean(x, 3)[5], [4.0])


def test_range_estimators_are_nonnegative_and_scale_with_range():
    panel = _panel()
    assert np.nanmin(parkinson(panel)) >= 0 and np.nanmin(garman_klass(panel)) >= 0


def test_rsi_bounds():
    r = rsi(_panel())
    assert np.nanmin(r) >= 0 and np.nanmax(r) <= 100


def test_previous_session_rv_is_constant_within_a_session_and_lagged():
    panel = make_toy_panel(n_sessions=5)
    prev = previous_session_rv(panel, 1)
    x = to_matrix(panel, "ret2")
    assert np.isnan(prev[:6]).all()  # nothing completed before session 0
    np.testing.assert_allclose(prev[6:12], np.tile(x[:6].sum(axis=0), (6, 1)))
