import pandas as pd

from forecasting.tests.check_scored import main, unscored


def test_flags_models_that_were_not_scored(tmp_path):
    metrics = pd.DataFrame(
        {
            "horizon": ["30min", "30min", "next_session"],
            "model": ["naive", "nbeats", "nbeats"],
            "QLIKE": [1.2, float("nan"), 0.8],
            "coverage": [1.0, 0.0, 1.0],
        }
    )
    assert list(unscored(metrics)["model"]) == ["nbeats"]
    path = tmp_path / "metrics.csv"
    metrics.to_csv(path, index=False)
    assert main(str(path)) == 1
    metrics.loc[1, "QLIKE"] = 1.0
    metrics.to_csv(path, index=False)
    assert main(str(path)) == 0
