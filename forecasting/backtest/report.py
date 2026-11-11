"""Render and persist backtest results."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pandas as pd

from forecasting.backtest.engine import Horizon, WalkForward
from forecasting.backtest.evaluate import Evaluation

SUMMARY_COLUMNS = [
    ("model", "model", None),
    ("QLIKE", "QLIKE", ".4f"),
    ("QLIKE_vs_naive", "QLIKE / naive", ".3f"),
    ("MSE", "MSE", ".4g"),
    ("MSE_vs_naive", "MSE / naive", ".3f"),
    ("MASE", "MASE", ".3f"),
    ("DM_stat", "DM stat", "+.2f"),
    ("DM_p", "DM p", ".3g"),
    ("symbols_beating_naive", "beats naive (symbols)", None),
    ("QLIKE_vs_lower_rungs", "QLIKE / best lower rung", ".3f"),
    ("DM_p_vs_lower_rungs", "p vs best lower rung", ".3g"),
]


def _fmt(value, spec: str | None) -> str:
    if spec is None:
        return str(value)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return format(value, spec)


def summary_markdown(evaluation: Evaluation, horizons: dict[str, Horizon]) -> str:
    out = []
    for hname, table in evaluation.summary.groupby("horizon", sort=False):
        h = horizons.get(hname)
        best = table["QLIKE"].idxmin()
        first = table.iloc[0]
        desc = ""
        if h is not None:
            where = (
                "at each session open"
                if h.origin == "session_open"
                else (f"every {h.stride} bar(s) intraday")
            )
            desc = f" - RV over next {h.bars} bars, origins {where}"
        out += [
            f"### {hname}{desc}",
            "",
            f"{first['n_obs']:,} scored forecasts per model over {first['n_origins']:,} origins; "
            f"mean target RV {first['mean_target']:.4g} %^2; "
            f"{first['excluded_obs']:,} (origin, symbol) pairs excluded "
            "(missing bars in the target window, zero RV, or a model without a forecast).",
            "",
            "| " + " | ".join(label for _, label, _ in SUMMARY_COLUMNS) + " |",
            "|" + "|".join("---" for _ in SUMMARY_COLUMNS) + "|",
        ]
        for idx, row in table.iterrows():
            cells = [_fmt(row[col], spec) for col, _, spec in SUMMARY_COLUMNS]
            if idx == best:
                cells[1] = f"**{cells[1]}**"
            out.append("| " + " | ".join(cells) + " |")
        cov = table[table["coverage"] < 0.999]
        if not cov.empty:
            out.append("")
            out.append(
                "Forecast coverage below 100%: "
                + ", ".join(f"{r.model} {100 * r.coverage:.1f}%" for r in cov.itertuples())
            )
        floored = table[table["floored"] > 0]
        if not floored.empty:
            out.append(
                "Forecasts floored before QLIKE: "
                + ", ".join(f"{r.model} {r.floored}" for r in floored.itertuples())
            )
        out.append("")
    return "\n".join(out)


LEGEND = """\
How to read this:
* Target: realized variance (sum of squared 5-min percent log returns) over the horizon;
  the first bar of a session uses open->close, so overnight gaps are excluded.
* QLIKE and MSE are on variance; lower is better. `/ naive` < 1 means better than the
  random walk on the same origins. MASE = MAE / naive MAE on the same origins.
* DM stat: Diebold-Mariano on QLIKE vs naive (negative = lower loss than naive), using
  the cross-sectional mean loss per origin and Newey-West errors. Vol forecast errors are
  persistent, so treat borderline p-values with caution.
* beats naive (symbols): symbols where the model's mean QLIKE is below naive's.
* QLIKE / best lower rung: against the lowest-QLIKE model of all lower rungs, on the
  same sample, with its DM p-value. The ladder rule is that this must be < 1.
* Best QLIKE per horizon in bold.
"""


def git_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_results(
    out_dir: str | Path,
    *,
    title: str,
    forecasts: pd.DataFrame,
    evaluation: Evaluation,
    horizons: dict[str, Horizon],
    walk_forward: WalkForward,
    meta: dict,
    quality_markdown: str = "",
) -> Path:
    """Write forecasts, metrics and a markdown summary to ``out_dir``; return the .md path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    forecasts.to_parquet(out / "forecasts.parquet", index=False)
    evaluation.summary.to_csv(out / "metrics.csv", index=False)
    evaluation.per_symbol.to_csv(out / "metrics_per_symbol.csv", index=False)
    meta = {**meta, "git": git_revision(), "walk_forward": walk_forward.__dict__}
    (out / "run.json").write_text(json.dumps(meta, indent=2, default=str))

    wf = walk_forward
    lines = [
        f"# {title}",
        "",
        *(f"- {k}: {v}" for k, v in meta.items() if k != "walk_forward"),
        f"- walk-forward: {wf.window} window, first {wf.initial_train_sessions} sessions for "
        f"training, refit every {wf.test_block_sessions} sessions",
        "",
        "## Results vs naive",
        "",
        summary_markdown(evaluation, horizons),
        LEGEND,
    ]
    if quality_markdown:
        lines += ["", quality_markdown]
    path = out / "results.md"
    path.write_text("\n".join(lines) + "\n")
    return path


LEADERBOARD_COLUMNS = [
    "run_at",
    "experiment",
    "data",
    "split",
    "horizon",
    "model",
    "model_spec",
    "n_obs",
    "QLIKE",
    "QLIKE_vs_naive",
    "MSE_vs_naive",
    "MASE",
    "DM_stat",
    "DM_p",
    "symbols_beating_naive",
    "rung",
    "QLIKE_vs_lower_rungs",
    "DM_p_vs_lower_rungs",
    "git",
]


def append_leaderboard(
    path: str | Path,
    evaluation: Evaluation,
    *,
    experiment: str,
    data: str,
    split: str,
    model_specs: dict[str, str],
    run_at: str,
) -> pd.DataFrame:
    """Append one row per (horizon, model) to the trial log and rewrite its markdown view.

    Every run is kept, including losers: the number of configurations tried is what
    tells you how much a "best" result may owe to luck.
    """
    path = Path(path)
    rows = evaluation.summary.assign(
        run_at=run_at,
        experiment=experiment,
        data=data,
        split=split,
        model_spec=evaluation.summary["model"].map(model_specs),
        git=git_revision(),
    )[LEADERBOARD_COLUMNS]
    if path.exists():
        rows = pd.concat([pd.read_csv(path), rows], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, index=False)
    path.with_suffix(".md").write_text(leaderboard_markdown(rows))
    return rows


def leaderboard_markdown(log: pd.DataFrame) -> str:
    lines = [
        "# Leaderboard",
        "",
        "Latest result per model configuration. `trials` counts every distinct configuration",
        "ever run on that data/split/horizon (losers included). The more trials, the more",
        "a top result may be luck, so the lockbox (`split = final`) is the number that counts.",
        "",
    ]
    for (data, split, horizon), g in log.groupby(["data", "split", "horizon"], sort=True):
        latest = g.drop_duplicates("model_spec", keep="last").sort_values("QLIKE_vs_naive")
        lines += [
            f"## {data} / {split} / {horizon} ({latest['model_spec'].nunique()} trials)",
            "",
            "| model | QLIKE / naive | MSE / naive | MASE | DM p | beats naive | spec | run |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in latest.itertuples():
            lines.append(
                f"| {r.model} | {r.QLIKE_vs_naive:.3f} | {r.MSE_vs_naive:.3f} | {r.MASE:.3f} "
                f"| {_fmt(r.DM_p, '.2g')} | {r.symbols_beating_naive} | `{r.model_spec}` "
                f"| {r.run_at} |"
            )
        lines.append("")
    return "\n".join(lines)
