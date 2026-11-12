"""Tests for rung 4: N-BEATS and Chronos RV models.

N-BEATS tests are skipped when torch is not installed.
Chronos tests are skipped when chronos-forecasting is not installed.
"""

import importlib.util

import numpy as np
import pytest

from forecasting.tests.conftest import make_toy_panel

B = 6

requires_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch not installed"
)
requires_chronos = pytest.mark.skipif(
    importlib.util.find_spec("chronos") is None, reason="chronos-forecasting not installed"
)


@pytest.fixture(scope="module")
def panel():
    return make_toy_panel(n_sessions=50, seed=42)


# ---------------------------------------------------------------------------
# N-BEATS
# ---------------------------------------------------------------------------


@requires_torch
def test_nbeats_fit_predict_shape(panel):
    from forecasting.models.deep import NBEATSRVModel

    model = NBEATSRVModel(
        lookback_days=5,
        n_blocks=2,
        width=16,
        n_epochs=3,
        min_train_windows=5,
    )
    model.fit(panel.iloc[: 30 * B * 2])
    assert model.state_dict_ is not None

    fc = model.predict(panel.iloc[: 35 * B * 2], B)
    assert set(fc.columns) == {"symbol", "yhat"}
    assert len(fc) == 2
    assert (fc["yhat"] > 0).all()


@requires_torch
def test_nbeats_intraday_predict(panel):
    from forecasting.models.deep import NBEATSRVModel

    model = NBEATSRVModel(
        lookback_days=5,
        n_blocks=2,
        width=16,
        n_epochs=3,
        min_train_windows=5,
        intraday_beta=0.5,
    )
    model.fit(panel.iloc[: 30 * B * 2])
    fc = model.predict(panel.iloc[: (30 * B + 3) * 2], 2)
    assert (fc["yhat"] > 0).all()


@requires_torch
def test_nbeats_too_little_data_returns_nan(panel):
    from forecasting.models.deep import NBEATSRVModel

    model = NBEATSRVModel(lookback_days=5, n_epochs=1, min_train_windows=100_000)
    model.fit(panel.iloc[: 10 * B * 2])
    assert model.state_dict_ is None
    fc = model.predict(panel.iloc[: 12 * B * 2], B)
    assert fc["yhat"].isna().all()


@requires_torch
def test_nbeats_ignores_the_future(panel):
    from forecasting.backtest.engine import Horizon, WalkForward, run_walk_forward
    from forecasting.models.deep import NBEATSRVModel

    h = Horizon("h", B, "session_open")
    wf = WalkForward(initial_train_sessions=26, test_block_sessions=4)
    model = NBEATSRVModel(
        lookback_days=5, n_blocks=2, width=16, n_epochs=3, min_train_windows=5
    )
    cut = panel["ts"].iloc[33 * B * 2]
    poisoned = panel.copy()
    future = (poisoned["ts"] >= cut).to_numpy()
    for col in ("open", "high", "low", "close", "ret", "ret2", "volume"):
        vals = poisoned[col].to_numpy(copy=True)
        vals[future] = vals[future] * 50 + 7
        poisoned[col] = vals
    a = run_walk_forward(panel, model, h, wf)
    b_df = run_walk_forward(poisoned, model, h, wf)
    past = (a["origin_ts"] <= cut).to_numpy()
    if np.isfinite(a.loc[past, "yhat"]).any():
        np.testing.assert_array_equal(
            a.loc[past, "yhat"].to_numpy(),
            b_df.loc[past, "yhat"].to_numpy(),
        )


@requires_torch
def test_nbeats_registered():
    from forecasting.models import build_model
    from forecasting.models.deep import NBEATSRVModel

    m = build_model("nbeats", params={"lookback_days": 5, "n_epochs": 1})
    assert isinstance(m, NBEATSRVModel)


# ---------------------------------------------------------------------------
# Chronos (skipped if not installed)
# ---------------------------------------------------------------------------


@requires_chronos
@requires_torch
def test_chronos_fit_predict_shape(panel):
    from forecasting.models.deep import ChronosRVModel

    model = ChronosRVModel(
        model_id="amazon/chronos-t5-tiny",
        num_samples=10,
        context_days=20,
    )
    model.fit(panel.iloc[: 30 * B * 2])
    fc = model.predict(panel.iloc[: 35 * B * 2], B)
    assert set(fc.columns) == {"symbol", "yhat"}
    assert len(fc) == 2
    assert (fc["yhat"] > 0).all()


@requires_chronos
def test_chronos_registered():
    from forecasting.models import build_model
    from forecasting.models.deep import ChronosRVModel

    m = build_model("chronos", params={"num_samples": 5})
    assert isinstance(m, ChronosRVModel)
