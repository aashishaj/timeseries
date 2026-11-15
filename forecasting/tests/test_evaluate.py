import numpy as np
import pandas as pd
import pytest

from forecasting.backtest.evaluate import evaluate


def _forecasts(n=200, seed=0):
    rng = np.random.default_rng(seed)
    origins = pd.date_range("2025-01-01 09:15", periods=n, freq="5min", tz="Asia/Kolkata")
    rows = []
    truth = rng.gamma(2.0, 0.5, (n, 2))
    for m, scale in [("naive", 1.4), ("good", 1.0), ("bad", 3.0)]:
        noise = rng.lognormal(0, 0.3, (n, 2))
        for i, o in enumerate(origins):
            for j, s in enumerate(["A", "B"]):
                rows.append((m, "h", o, s, truth[i, j] * scale * noise[i, j], truth[i, j]))
    return pd.DataFrame(rows, columns=["model", "horizon", "origin_ts", "symbol", "yhat", "y"])


def test_naive_is_reference_and_ranking_is_sensible():
    ev = evaluate(_forecasts())
    s = ev.summary.set_index("model")
    assert s.loc["naive", ["MASE", "QLIKE_vs_naive", "MSE_vs_naive"]].tolist() == [1, 1, 1]
    assert s.loc["good", "QLIKE"] < s.loc["naive", "QLIKE"] < s.loc["bad", "QLIKE"]
    assert s.loc["good", "DM_stat"] < 0 < s.loc["bad", "DM_stat"]
    assert s.loc["good", "symbols_beating_naive"] == "2/2"
    assert set(ev.per_symbol["symbol"]) == {"A", "B"}


def test_common_sample_excludes_any_missing_forecast_and_bad_targets():
    f = _forecasts(n=50)
    f.loc[(f["model"] == "bad") & (f.index % 7 == 0), "yhat"] = np.nan
    f.loc[f.index % 11 == 0, "y"] = 0.0  # zero RV target, for every model
    f.loc[f["model"] != "naive", "y"] = f.loc[f["model"] == "naive", "y"].to_numpy().tolist() * 2
    ev = evaluate(f)
    n = ev.summary["n_obs"].unique()
    assert len(n) == 1 and n[0] < 100  # identical sample for every model


def test_models_must_agree_on_targets():
    f = _forecasts(n=20)
    f.loc[f["model"] == "good", "y"] *= 2
    with pytest.raises(ValueError, match="disagree"):
        evaluate(f)


def test_zero_forecasts_are_floored_and_counted():
    f = _forecasts(n=20)
    f.loc[(f["model"] == "bad") & (f.index % 5 == 0), "yhat"] = 0.0
    s = evaluate(f).summary.set_index("model")
    assert s.loc["bad", "floored"] > 0 and np.isfinite(s.loc["bad", "QLIKE"])


def test_each_rung_is_compared_to_best_of_all_lower_rungs():
    ev = evaluate(_forecasts(), rungs={"naive": 1, "bad": 2, "good": 3})
    s = ev.summary.set_index("model")
    assert s.loc["good", "best_lower_rung"] == "naive"  # best of rungs 1-2, not "bad"
    assert s.loc["good", "QLIKE_vs_lower_rungs"] == pytest.approx(
        s.loc["good", "QLIKE"] / s.loc["naive", "QLIKE"]
    )
    assert s.loc["good", "DM_p_vs_lower_rungs"] < 0.05
    assert s.loc["bad", "best_lower_rung"] == "naive"
    assert pd.isna(s.loc["naive", "best_lower_rung"])


def test_model_without_forecasts_is_reported_but_does_not_empty_the_sample():
    from forecasting.backtest.evaluate import plain_accuracy

    f = _forecasts(n=50)
    full = evaluate(f).summary.set_index("model")
    f.loc[f["model"] == "bad", "yhat"] = np.nan  # e.g. an optional dependency is missing
    ev = evaluate(f)
    s = ev.summary.set_index("model")
    assert s.loc["good", "n_obs"] == full.loc["good", "n_obs"] == 100
    assert s.loc["good", "QLIKE"] == pytest.approx(full.loc["good", "QLIKE"])
    assert s.loc["bad", "coverage"] == 0 and np.isnan(s.loc["bad", "QLIKE"])
    acc = plain_accuracy(f).set_index("model")
    assert np.isfinite(acc.loc["good", "explained"]) and np.isnan(acc.loc["bad", "explained"])


def test_a_zero_forecast_is_floored_and_does_not_swamp_the_average():
    f = _forecasts(n=200)
    clean = evaluate(f).summary.set_index("model")
    f.loc[f.index[(f["model"] == "naive")][0], "yhat"] = 0.0  # "no volatility at all"
    s = evaluate(f).summary.set_index("model")
    assert s.loc["naive", "floored"] == 1
    # One absurd forecast in 400 is punished (about 1/floor_rel / 400), but boundedly...
    assert clean.loc["naive", "QLIKE"] < s.loc["naive", "QLIKE"] < 10
    assert s.loc["good", "QLIKE"] == pytest.approx(clean.loc["good", "QLIKE"])
    # ...whereas a near-zero floor lets that single row decide every comparison.
    tiny = evaluate(f, floor_rel=1e-9).summary.set_index("model")
    assert tiny.loc["naive", "QLIKE"] > 1e5
