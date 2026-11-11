import numpy as np
import pytest

from forecasting.data.panel import to_matrix
from forecasting.models import HAR, SeasonalEWMA
from forecasting.models.common import daily_rv, har_features, seasonal_profile
from forecasting.tests.conftest import make_toy_panel

B = 6


def _with_ret2(panel, ret2_matrix):
    out = panel.copy()
    out["ret2"] = ret2_matrix.ravel()
    out.attrs = panel.attrs
    return out


def test_seasonal_profile_recovers_planted_shape():
    panel = make_toy_panel(n_sessions=30)
    shape = np.array([3.0, 1.0, 0.5, 0.5, 1.0, 2.0])
    level = np.random.default_rng(0).lognormal(0, 0.5, (30, 1, 2))  # day-to-day level changes
    ret2 = (level * shape[None, :, None]).reshape(-1, 2)
    prof = seasonal_profile(_with_ret2(panel, ret2), pooled=False)
    np.testing.assert_allclose(prof.mean(axis=0), 1.0)
    np.testing.assert_allclose(prof[:, 0], shape / shape.mean())
    pooled = seasonal_profile(_with_ret2(panel, ret2), pooled=True)
    np.testing.assert_allclose(pooled[:, 0], pooled[:, 1])


def test_daily_rv_sums_and_rescales_missing_bars():
    panel = make_toy_panel(n_sessions=4, missing=[(1, 3, "AAA")])  # blanks bars 3 and 4
    x = to_matrix(panel, "ret2")
    hist = panel.iloc[: (3 * B + 2) * 2]  # 3 full sessions + 2 bars of the 4th
    rv, today = daily_rv(hist, min_coverage=0.5)
    np.testing.assert_allclose(rv[0], x[:B].sum(axis=0))
    observed = x[B : 2 * B, 0][~np.isnan(x[B : 2 * B, 0])]
    assert rv[1, 0] == pytest.approx(observed.sum() / (len(observed) / B))
    assert today.shape == (2, 2)
    assert np.isnan(daily_rv(hist, min_coverage=0.9)[0][1, 0])


def test_har_features_are_trailing_means():
    rv = np.arange(1.0, 11.0)[:, None]
    f = har_features(rv, (1, 5))
    assert f[9, 0, 0] == 10 and f[9, 1, 0] == pytest.approx(np.mean([6, 7, 8, 9, 10]))
    assert np.isnan(f[1, 1, 0])


def test_har_recovers_known_log_ar_coefficients():
    n = 40
    log_rv = np.empty((n, 2))
    log_rv[0] = [3.0, -3.0]  # far from the fixed point so the regressor varies
    for t in range(1, n):
        log_rv[t] = 0.3 + 0.6 * log_rv[t - 1]  # exact, noise-free log-AR(1)
    panel = make_toy_panel(n_sessions=n)
    ret2 = np.repeat(np.exp(log_rv) / B, B, axis=0)
    toy = _with_ret2(panel, ret2)
    model = HAR(lags=(1,), pooled=False).fit(toy.iloc[: 30 * B * 2])
    np.testing.assert_allclose(model.coef_[:, 1], 0.6, rtol=1e-6)
    np.testing.assert_allclose(model.coef_[:, 0], 0.3, rtol=1e-5)
    hist = toy.iloc[: 35 * B * 2]
    fc = model.predict(hist, B)["yhat"].to_numpy()
    expected = np.exp(model.coef_[:, 0] + model.coef_[:, 1] * log_rv[34]) * model.smear_
    np.testing.assert_allclose(fc, expected)


def test_har_intraday_window_and_beta_adjustment():
    panel = make_toy_panel(n_sessions=20, seed=3)
    base = HAR(lags=(1, 2), pooled=True).fit(panel.iloc[: 15 * B * 2])
    mixed = HAR(lags=(1, 2), pooled=True, intraday_beta=0.5).fit(panel.iloc[: 15 * B * 2])
    at_open = panel.iloc[: 16 * B * 2]
    np.testing.assert_allclose(  # at the open the window shares of a full day sum to 1
        base.predict(at_open, B)["yhat"], base._today_forecast(at_open)[0]
    )
    mid = panel.iloc[: (16 * B + 3) * 2]
    fc_day, today = base._today_forecast(mid)
    share = base.profile_ / base.profile_.sum(axis=0)
    ratio = np.clip(today.sum(axis=0) / (fc_day * share[:3].sum(axis=0)), 0.25, 4)
    np.testing.assert_allclose(
        mixed.predict(mid, 2)["yhat"], base.predict(mid, 2)["yhat"] * ratio**0.5
    )


def test_ewma_incremental_matches_fresh_computation():
    panel = make_toy_panel(n_sessions=12)
    train = panel.iloc[: 4 * B * 2]
    incremental = SeasonalEWMA(halflife_sessions=5).fit(train)
    for bars in range(4 * B, 12 * B, 5):
        hist = panel.iloc[: bars * 2]
        fresh = SeasonalEWMA(halflife_sessions=5).fit(train)
        np.testing.assert_allclose(
            incremental.predict(hist, 2)["yhat"], fresh.predict(hist, 2)["yhat"], rtol=1e-12
        )
