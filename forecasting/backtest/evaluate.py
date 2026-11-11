"""Score backtest forecasts against the naive baseline on a common sample.

Fairness rules:
* every model is scored on exactly the same (origin, symbol) pairs: those where the
  target is observed and positive and *every* model produced a finite forecast;
* every metric is also reported relative to the baseline (naive by default);
* significance uses a Diebold-Mariano test on the per-origin cross-sectional mean
  loss differential, with Newey-West errors (lag >= target-window overlap).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting.backtest.engine import Horizon
from forecasting.backtest.metrics import (
    absolute_error,
    default_hac_lag,
    diebold_mariano,
    qlike,
    squared_error,
)


@dataclass
class Evaluation:
    summary: pd.DataFrame  # one row per (horizon, model)
    per_symbol: pd.DataFrame  # one row per (horizon, model, symbol)


def evaluate(
    forecasts: pd.DataFrame,
    *,
    baseline: str = "naive",
    horizons: Mapping[str, Horizon] | None = None,
    floor_rel: float = 1e-6,
) -> Evaluation:
    """Evaluate the output of :func:`run_walk_forward` for one or more models/horizons.

    Args:
        forecasts: concatenated engine outputs.
        baseline: model name every result is compared to.
        horizons: used for the target-window overlap in the HAC lag.
        floor_rel: forecasts below ``floor_rel * median(y)`` are floored before QLIKE
            (a zero variance forecast has infinite QLIKE); the count is reported.
    """
    horizons = dict(horizons or {})
    summaries, per_symbol = [], []
    for hname, f in forecasts.groupby("horizon", sort=False):
        models = list(dict.fromkeys(f["model"]))
        if baseline not in models:
            raise ValueError(f"baseline {baseline!r} missing for horizon {hname}")
        key = ["origin_ts", "symbol"]
        yhat = f.pivot_table(index=key, columns="model", values="yhat", dropna=False)[models]
        y_all = f.pivot_table(index=key, columns="model", values="y", dropna=False)
        y = y_all[baseline]
        spread = (y_all.max(axis=1) - y_all.min(axis=1)).abs().max()
        if np.isfinite(spread) and spread > 1e-9:
            raise ValueError(f"models disagree on targets for horizon {hname}")
        n_total = len(y)
        target_ok = np.isfinite(y) & (y > 0)
        mask = target_ok & np.isfinite(yhat).all(axis=1)
        yv = y[mask].to_numpy()
        floor = floor_rel * float(np.median(yv)) if len(yv) else 0.0
        origins = y.index.get_level_values("origin_ts")[mask]
        symbols = y.index.get_level_values("symbol")[mask]
        h = horizons.get(hname)
        overlap = h.overlap if h is not None else 0

        losses = {}
        for m in models:
            raw = yhat[m][mask].to_numpy()
            fc = np.maximum(raw, floor)
            losses[m] = {
                "qlike": qlike(yv, fc),
                "se": squared_error(yv, fc),
                "ae": absolute_error(yv, fc),
                "floored": int((raw < floor).sum()),
                "coverage": float(np.isfinite(yhat[m][target_ok]).mean()),
            }
        base = losses[baseline]
        base_ps = _per_symbol_means(base, symbols)
        for m in models:
            lm = losses[m]
            d_t = pd.Series(lm["qlike"] - base["qlike"]).groupby(np.asarray(origins)).mean()
            if m == baseline:
                dm_stat, dm_p = np.nan, np.nan
            else:
                lag = default_hac_lag(len(d_t), overlap)
                dm_stat, dm_p = diebold_mariano(d_t.to_numpy(), np.zeros(len(d_t)), lag)
            ps = _per_symbol_means(lm, symbols)
            wins = int((ps["qlike"] < base_ps["qlike"]).sum())
            summaries.append(
                {
                    "horizon": hname,
                    "model": m,
                    "n_obs": int(mask.sum()),
                    "n_origins": int(len(d_t)),
                    "coverage": lm["coverage"],
                    "QLIKE": float(lm["qlike"].mean()),
                    "MSE": float(lm["se"].mean()),
                    "MAE": float(lm["ae"].mean()),
                    "MASE": float(lm["ae"].mean() / base["ae"].mean()),
                    "QLIKE_vs_naive": float(lm["qlike"].mean() / base["qlike"].mean()),
                    "MSE_vs_naive": float(lm["se"].mean() / base["se"].mean()),
                    "DM_stat": dm_stat,
                    "DM_p": dm_p,
                    "symbols_beating_naive": f"{wins}/{len(ps)}" if m != baseline else "-",
                    "floored": lm["floored"],
                    "excluded_obs": int(n_total - mask.sum()),
                    "mean_target": float(yv.mean()) if len(yv) else np.nan,
                }
            )
            ps = ps.assign(
                horizon=hname,
                model=m,
                QLIKE_vs_naive=ps["qlike"] / base_ps["qlike"],
                MSE_vs_naive=ps["se"] / base_ps["se"],
            )
            per_symbol.append(ps.reset_index())
    summary = pd.DataFrame(summaries)
    per_sym = pd.concat(per_symbol, ignore_index=True).rename(
        columns={"qlike": "QLIKE", "se": "MSE", "ae": "MAE"}
    )
    cols = ["horizon", "model", "symbol", "n_obs", "QLIKE", "MSE", "MAE"]
    per_sym = per_sym[[*cols, "QLIKE_vs_naive", "MSE_vs_naive"]]
    return Evaluation(summary=summary, per_symbol=per_sym)


def _per_symbol_means(losses: dict, symbols: pd.Index) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"symbol": np.asarray(symbols), **{k: losses[k] for k in ("qlike", "se", "ae")}}
    )
    out = frame.groupby("symbol").mean()
    out["n_obs"] = frame.groupby("symbol").size()
    return out
