import pandas as pd
import yaml

from forecasting.cli import main
from forecasting.config import CONFIG_DIR, load_config


def _small_config(tmp_path, root):
    """The shipped experiment config, shrunk to 2 symbols and a short warm-up."""
    (tmp_path / "symbols.yaml").write_text(yaml.safe_dump({"symbols": ["RELIANCE", "INFY"]}))
    raw = yaml.safe_load((CONFIG_DIR / "rv_baselines.yaml").read_text())
    raw.update(
        symbols="symbols.yaml",
        calendar=str(CONFIG_DIR / "nse_calendar.yaml"),
        data={"root": str(root)},
        walk_forward={"initial_train_sessions": 20, "test_block_sessions": 5},
    )
    path = tmp_path / "exp.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_default_config_loads():
    cfg = load_config()
    assert [h.bars for h in cfg.horizons] == [6, 75]
    assert cfg.baseline == "naive" and len(cfg.symbols) == 10


def test_validate_and_backtest_cli(tmp_path, synthetic_export, capsys):
    cfg = _small_config(tmp_path, synthetic_export["root"])
    assert main(["validate", "--config", str(cfg), "--report", str(tmp_path / "q.md")]) == 0
    assert "Data quality" in (tmp_path / "q.md").read_text()

    out = tmp_path / "run"
    assert main(["backtest", "--config", str(cfg), "--out", str(out), "--max-folds", "3"]) == 0
    md = (out / "results.md").read_text()
    assert "### 30min" in md and "### next_session" in md and "seasonal_mean_5d" in md
    metrics = pd.read_csv(out / "metrics.csv")
    naive = metrics[metrics["model"] == "naive"]
    assert (naive["MASE"] == 1).all() and (naive["QLIKE_vs_naive"] == 1).all()
    forecasts = pd.read_parquet(out / "forecasts.parquet")
    assert set(forecasts["model"]) == {"naive", "hist_mean", "seasonal_naive", "seasonal_mean_5d"}
    assert "QLIKE / naive" in capsys.readouterr().out
