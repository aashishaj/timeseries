import numpy as np
import pandas as pd
import pytest

from forecasting.data.panel import to_matrix
from forecasting.features import (
    LookaheadError,
    assert_no_lookahead,
    log_returns,
    session_rv_to_date,
    time_features,
    trailing_rv,
)
from forecasting.tests.conftest import make_toy_panel


def test_returns_use_open_for_first_bar_and_never_cross_sessions(toy_panel):
    r = to_matrix(toy_panel, "ret")
    close, open_ = to_matrix(toy_panel, "close"), to_matrix(toy_panel, "open")
    b = toy_panel.attrs["bars_per_session"]
    np.testing.assert_allclose(r[0], 100 * np.log(close[0] / open_[0]))
    np.testing.assert_allclose(r[b], 100 * np.log(close[b] / open_[b]))  # next session open
    np.testing.assert_allclose(r[1], 100 * np.log(close[1] / close[0]))


def test_missing_bar_blanks_its_return_and_the_next():
    panel = make_toy_panel(missing=[(1, 2, "AAA")])
    r = to_matrix(panel, "ret")
    row = 1 * 6 + 2
    assert np.isnan(r[row, 0]) and np.isnan(r[row + 1, 0])
    assert np.isfinite(r[row + 2, 0]) and np.isfinite(r[row, 1])


def test_trailing_rv_matches_manual_sum(toy_panel):
    rv = trailing_rv(toy_panel, 4).to_numpy().reshape(-1, 2)
    x = to_matrix(toy_panel, "ret2")
    np.testing.assert_allclose(rv[10], x[7:11].sum(axis=0))
    assert np.isnan(rv[2]).all()  # window not yet full


BUILDERS = {
    "log_returns": log_returns,
    "trailing_rv_6": lambda p: trailing_rv(p, 6),
    "session_rv_to_date": session_rv_to_date,
    "time_features": time_features,
}


@pytest.mark.parametrize("name", list(BUILDERS))
def test_feature_builders_have_no_lookahead(name):
    panel = make_toy_panel(n_sessions=10, missing=[(3, 1, "BBB"), (6, 0, "AAA")])
    assert_no_lookahead(BUILDERS[name], panel, n_cuts=8)


LEAKY = {
    "next_bar_return": lambda p: (
        pd.Series(log_returns(p)).groupby(p["symbol"].to_numpy()).shift(-1)
    ),
    "full_sample_zscore": lambda p: (p["ret"] - p["ret"].mean()) / p["ret"].std(),
    "centred_window": lambda p: p.groupby("symbol")["ret2"].transform(
        lambda s: s.rolling(5, center=True, min_periods=1).mean()
    ),
}


@pytest.mark.parametrize("name", list(LEAKY))
def test_harness_catches_leaky_features(name):
    with pytest.raises(LookaheadError):
        assert_no_lookahead(LEAKY[name], make_toy_panel(n_sessions=10), n_cuts=8)
