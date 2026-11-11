import numpy as np
import pytest

from forecasting.backtest.engine import Horizon, WalkForward, run_walk_forward
from forecasting.models import Combination, RandomWalk, SeasonalNaive
from forecasting.models.base import ForecastModel
from forecasting.models.ensemble import combine
from forecasting.tests.conftest import make_toy_panel

B = 6
CHEAP = [
    {"type": "random_walk"},
    {"type": "seasonal_naive", "params": {"n_sessions": 2}},
    {"type": "ewma", "params": {"halflife_sessions": 1.0}},
]


def test_combine_is_weighted_geometric_mean_and_skips_missing():
    f = np.log(np.array([[1.0, 4.0], [4.0, np.nan]]))  # (K=2, S=2)
    out = combine(f, np.array([0.5, 0.5]))
    np.testing.assert_allclose(out, [2.0, 4.0])  # symbol 2: only member 1 available
    np.testing.assert_allclose(combine(f, np.array([0.5, 0.5]), bias=np.log(2)), [4.0, 8.0])


def test_equal_combination_matches_members():
    panel = make_toy_panel(n_sessions=12)
    train, hist = panel.iloc[: 6 * B * 2], panel.iloc[: (8 * B + 2) * 2]
    combo = Combination(members=CHEAP[:2]).fit(train)
    a = RandomWalk().fit(train).predict(hist, 2)["yhat"].to_numpy()
    b = SeasonalNaive(n_sessions=2).fit(train).predict(hist, 2)["yhat"].to_numpy()
    np.testing.assert_allclose(combo.predict(hist, 2)["yhat"], np.sqrt(a * b))


class Oracle(ForecastModel):
    """Cheats on purpose (only inside this test): knows a constant close to the truth."""

    def __init__(self, name="oracle", level=1.0):
        super().__init__(name)
        self.level = level

    def predict(self, history_df, horizon):
        return self.forecast_frame(history_df, np.full(2, self.level * horizon))


def test_optimized_weights_prefer_the_informative_member(monkeypatch):
    """Variance changes day to day but is flat within a day, so the random walk is
    informative intraday while a constant is not; the bias term cannot fix that."""
    import forecasting.models as registry

    rng = np.random.default_rng(0)
    template = make_toy_panel(n_sessions=30)
    level = rng.lognormal(0.0, 0.7, (30, 1, 2))
    ret2 = np.broadcast_to(level, (30, B, 2)).reshape(-1)
    panel = template.assign(ret2=ret2)
    panel.attrs = template.attrs
    monkeypatch.setitem(registry.MODEL_REGISTRY, "oracle", Oracle)
    combo = Combination(
        members=[{"type": "random_walk"}, {"type": "oracle", "params": {"level": 1.0}}],
        method="qlike_opt",
        validation_sessions=10,
        validation_stride=1,
    ).fit(panel)
    from scipy.optimize import minimize_scalar

    from forecasting.backtest.metrics import qlike

    f, y = combo._validation_forecasts(2)
    log_f, ok = np.log(f), np.isfinite(y) & (y > 0)

    def best_loss(weights):  # each candidate gets its own best bias term
        res = minimize_scalar(
            lambda c: np.mean(qlike(y[ok], combine(log_f, np.asarray(weights), c)[ok])),
            bounds=(-3, 3),
            method="bounded",
        )
        return res.fun

    w, _ = combo._fit_weights(2)
    assert w[0] > 0.5  # the informative member carries most of the weight
    found = best_loss(w)
    for alternative in ([1.0, 0.0], [0.0, 1.0], [0.5, 0.5]):
        assert found <= best_loss(alternative) + 1e-6


@pytest.mark.parametrize("method", ["equal", "qlike_opt"])
def test_combination_ignores_the_future(method):
    panel = make_toy_panel(n_sessions=20, seed=2)
    model = Combination(members=CHEAP, method=method, validation_sessions=4)
    wf = WalkForward(initial_train_sessions=10, test_block_sessions=4)
    h = Horizon("h", 2)
    cut = panel["ts"].iloc[15 * B * 2]
    poisoned = panel.copy()
    future = (poisoned["ts"] >= cut).to_numpy()
    for col in ("open", "high", "low", "close", "ret", "ret2"):
        values = poisoned[col].to_numpy(copy=True)
        values[future] = values[future] * 50 + 7
        poisoned[col] = values
    a = run_walk_forward(panel, model, h, wf)
    b = run_walk_forward(poisoned, model, h, wf)
    past = (a["origin_ts"] <= cut).to_numpy()
    assert np.isfinite(a.loc[past, "yhat"]).all()
    np.testing.assert_array_equal(a.loc[past, "yhat"], b.loc[past, "yhat"])


def test_rejects_bad_config():
    with pytest.raises(ValueError):
        Combination(members=[])
    with pytest.raises(ValueError):
        Combination(members=CHEAP, method="median")
