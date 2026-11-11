import warnings

import numpy as np
import pytest

from forecasting.data.panel import to_matrix
from forecasting.models import ARMALogRV, SeasonalGARCH
from forecasting.tests.conftest import make_toy_panel

B = 6


@pytest.fixture(scope="module")
def panel():
    return make_toy_panel(n_sessions=150, seed=11)


def test_variance_recursion_matches_arch(panel):
    from arch import arch_model

    train = panel.iloc[: 120 * B * 2]
    model = SeasonalGARCH(asymmetric=True, burn_in_sessions=10_000).fit(train)
    nxt = model._next_variance(train)
    j = 0
    z = to_matrix(train, "ret")[:, j] / np.sqrt(np.tile(model.profile_[:, j], 120))
    z = z[~np.isnan(z)] * model.scale_[j]
    params = [model.omega_[j], model.alpha_[j], model.gamma_[j], model.beta_[j], 8.0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fixed = arch_model(z, mean="Zero", p=1, o=1, q=1, dist="t", rescale=False).fix(params)
    # one-step forecast after the last return, from arch's own recursion
    expected = fixed.forecast(horizon=1, reindex=False).variance.iloc[-1, 0]
    assert nxt[j] == pytest.approx(expected, rel=1e-3)  # start-up values differ slightly


def test_incremental_state_matches_fresh_pass(panel):
    train = panel.iloc[: 100 * B * 2]
    incremental = SeasonalGARCH(burn_in_sessions=10_000).fit(train)
    for bars in range(100 * B, 110 * B, 7):
        hist = panel.iloc[: bars * 2]
        fresh = SeasonalGARCH(burn_in_sessions=10_000).fit(train)
        np.testing.assert_allclose(
            incremental.predict(hist, 3)["yhat"], fresh.predict(hist, 3)["yhat"], rtol=1e-10
        )


def test_long_horizon_reverts_to_unconditional(panel):
    model = SeasonalGARCH().fit(panel.iloc[: 100 * B * 2])
    hist = panel.iloc[: 100 * B * 2]
    one = model.predict(hist, 1)["yhat"].to_numpy()
    assert np.all(one > 0)
    uncond = model.omega_ / (1 - model.persistence_) / model.scale_**2
    far = model.predict(hist, B)["yhat"].to_numpy() / B  # average bar over a session
    assert np.all(np.abs(far - uncond) <= np.abs(one / model.profile_[0] - uncond) + 1e-12)


def test_too_little_data_gives_no_forecast():
    small = make_toy_panel(n_sessions=10)
    fc = SeasonalGARCH().fit(small).predict(small, 2)
    assert fc["yhat"].isna().all()


def test_arma_log_rv_forecasts_positive(panel):
    model = ARMALogRV(order=(1, 1)).fit(panel.iloc[: 100 * B * 2])
    fc = model.predict(panel.iloc[: 101 * B * 2], B)["yhat"]
    assert (fc > 0).all() and fc.notna().all()
