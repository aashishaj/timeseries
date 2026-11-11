import numpy as np
import pandas as pd
import pytest

from forecasting.backtest.engine import Horizon, WalkForward, make_folds, run_walk_forward
from forecasting.data.panel import to_matrix
from forecasting.models import HAR, HistoricalMean, RandomWalk, SeasonalEWMA, SeasonalNaive
from forecasting.models.base import ForecastModel
from forecasting.tests.conftest import make_toy_panel

WF = WalkForward(initial_train_sessions=4, test_block_sessions=3)
H_INTRA = Horizon("2bars", bars=2, origin="intraday", stride=1)
H_SESSION = Horizon("session", bars=6, origin="session_open")


def test_expanding_folds_are_chronological_and_disjoint():
    folds = make_folds(20, WalkForward(initial_train_sessions=5, test_block_sessions=4))
    assert [(f.train_start, f.test_start, f.test_end) for f in folds] == [
        (0, 5, 9),
        (0, 9, 13),
        (0, 13, 17),
        (0, 17, 20),
    ]


def test_rolling_folds_and_max_folds():
    wf = WalkForward(5, 4, window="rolling", rolling_train_sessions=3, max_folds=2)
    assert [(f.train_start, f.test_start) for f in make_folds(20, wf)] == [(2, 5), (6, 9)]
    with pytest.raises(ValueError):
        make_folds(20, WalkForward(5, 4, window="rolling"))
    with pytest.raises(ValueError):
        make_folds(5, WalkForward(5, 4))


def test_origin_slots():
    assert Horizon("30min", 6).origin_slots(75).tolist() == list(range(70))
    assert Horizon("30min", 6, stride=6).origin_slots(75)[-1] == 66
    assert Horizon("s", 75, origin="session_open").origin_slots(75).tolist() == [0]
    assert Horizon("30min", 6).overlap == 5 and Horizon("s", 75, "session_open").overlap == 0


class Spy(ForecastModel):
    """Records what the engine shows it; the log is shared across deep copies."""

    def __init__(self, log):
        super().__init__("spy")
        self.log = log

    def __deepcopy__(self, memo):
        return Spy(self.log)

    def fit(self, train_df):
        self.log.append(("fit", train_df["bar_end"].max(), train_df["session"].iloc[0]))
        return self

    def predict(self, history_df, horizon):
        self.log.append(("predict", history_df["bar_end"].max(), len(history_df)))
        return self.forecast_frame(history_df, np.ones(2))


@pytest.mark.parametrize("horizon", [H_INTRA, H_SESSION])
def test_model_never_sees_the_future(horizon):
    panel = make_toy_panel(n_sessions=12)
    log = []
    out = run_walk_forward(panel, Spy(log), horizon, WF)
    origins = out["origin_ts"].iloc[::2].reset_index(drop=True)
    preds = [e for e in log if e[0] == "predict"]
    assert len(preds) == len(origins)
    for (_, last_seen, _), origin in zip(preds, origins, strict=True):
        assert last_seen <= origin  # only bars that have ended by the origin
    fits = [e for e in log if e[0] == "fit"]
    first_test = out.groupby("fold")["origin_ts"].min()
    for (_, last_train, _), fold_start in zip(fits, first_test, strict=True):
        assert last_train <= fold_start


def test_history_is_complete_up_to_origin():
    """History must contain *every* bar before the origin, not just some of them."""
    panel = make_toy_panel(n_sessions=12)
    log = []
    out = run_walk_forward(panel, Spy(log), H_INTRA, WF)
    preds = [e for e in log if e[0] == "predict"]
    rows = out.iloc[::2]
    for (_, last_seen, n_rows), origin, slot in zip(
        preds, rows["origin_ts"], rows["slot"], strict=True
    ):
        assert n_rows == (panel["bar_end"] <= origin).sum()
        if slot > 0:
            assert last_seen == origin  # the bar that just closed is visible
        else:  # at the open: nothing from today, previous session's close at most
            assert last_seen.date() < origin.date()


@pytest.mark.parametrize(
    "model",
    [
        RandomWalk(),
        HistoricalMean(),
        SeasonalNaive(n_sessions=2),
        SeasonalEWMA(halflife_sessions=1.0),
        HAR(lags=(1, 2), pooled=False),
        HAR(lags=(1, 2), pooled=True, intraday_beta=0.5),
    ],
    ids=lambda m: repr(m),
)
@pytest.mark.parametrize("horizon", [H_INTRA, H_SESSION])
def test_poisoning_the_future_does_not_change_past_forecasts(model, horizon):
    panel = make_toy_panel(n_sessions=14, seed=5)
    cut = panel["ts"].iloc[len(panel) // 2]
    poisoned = panel.copy()
    future = (poisoned["ts"] >= cut).to_numpy()
    for col in ("open", "high", "low", "close", "ret", "ret2", "volume"):
        values = poisoned[col].to_numpy(copy=True)
        values[future] = values[future] * 50 + 7
        poisoned[col] = values
    a = run_walk_forward(panel, model, horizon, WF)
    b = run_walk_forward(poisoned, model, horizon, WF)
    past = (a["origin_ts"] <= cut).to_numpy()
    assert past.sum() > 0 and (~past).sum() > 0
    np.testing.assert_array_equal(a.loc[past, "yhat"], b.loc[past, "yhat"])
    assert not np.allclose(a.loc[~past, "yhat"], b.loc[~past, "yhat"])


def test_targets_are_future_window_sums():
    panel = make_toy_panel(n_sessions=10)
    out = run_walk_forward(panel, RandomWalk(), H_INTRA, WF)
    x = to_matrix(panel, "ret2")
    row = out.iloc[7]  # some origin, some symbol
    p = int(np.flatnonzero((panel["ts"].iloc[::2] == row["origin_ts"]).to_numpy())[0])
    j = panel.attrs["symbols"].index(row["symbol"])
    assert row["y"] == pytest.approx(x[p : p + 2, j].sum())


def test_output_shape_and_columns():
    panel = make_toy_panel(n_sessions=10)
    out = run_walk_forward(panel, RandomWalk(), H_INTRA, WF)
    n_test_sessions = 10 - 4
    assert len(out) == n_test_sessions * len(H_INTRA.origin_slots(6)) * 2
    assert list(out.columns) == [
        "model",
        "horizon",
        "fold",
        "origin_ts",
        "session",
        "slot",
        "symbol",
        "yhat",
        "y",
    ]
    assert out["slot"].max() == 4  # 2-bar windows never cross the close


class BadSymbols(ForecastModel):
    def predict(self, history_df, horizon):
        return pd.DataFrame({"symbol": ["AAA", "ZZZ"], "yhat": [1.0, 2.0]})


class Shuffled(ForecastModel):
    def predict(self, history_df, horizon):
        return pd.DataFrame({"symbol": ["BBB", "AAA"], "yhat": [2.0, 1.0]})


def test_forecast_contract_is_enforced():
    panel = make_toy_panel(n_sessions=6)
    with pytest.raises(ValueError, match="unknown symbols"):
        run_walk_forward(panel, BadSymbols(), H_SESSION, WF)
    out = run_walk_forward(panel, Shuffled(), H_SESSION, WF)
    assert out.set_index("symbol")["yhat"].groupby(level=0).first().to_dict() == {
        "AAA": 1.0,
        "BBB": 2.0,
    }
