"""Fail if any model in a backtest's metrics.csv was "not scored".

A model that cannot run (for example torch missing, or a training crash) does not
stop a backtest: it is logged and reported as "not scored". The deep-models CI job
uses this to turn that into a failure. Usage: ``python check_scored.py metrics.csv``.
"""

from __future__ import annotations

import sys

import pandas as pd


def unscored(metrics: pd.DataFrame) -> pd.DataFrame:
    return metrics[metrics["QLIKE"].isna()][["horizon", "model", "coverage"]]


def main(path: str) -> int:
    metrics = pd.read_csv(path)
    bad = unscored(metrics)
    for row in bad.itertuples():
        print(f"not scored: {row.model} at {row.horizon} ({100 * row.coverage:.0f}% coverage)")
    if bad.empty:
        print(f"all {metrics['model'].nunique()} models scored at every horizon")
    return 1 if len(bad) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
