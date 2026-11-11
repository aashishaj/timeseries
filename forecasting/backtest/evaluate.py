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
    rungs: Mapping[str, int] | None = None,
    floor_rel: float = 1e-6,
) -> Evaluation:
    """Evaluate the output of :func:`run_walk_forward` for one or more models/horizons.

    Args:
        forecasts: concatenated engine outputs.
        baseline: model name every result is compared to.
        horizons: used for the target-window overlap in the HAC lag.
        rungs: ladder rung per model. Each model in rung r > 1 is also compared to the
            best (lowest QLIKE) model of all lower rungs on the same sample.
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
        origin_ids = np.asarray(origins)

        mean_qlike = {m: float(losses[m]["qlike"].mean()) for m in models}
        rung_of = {m: (rungs or {}).get(m) for m in models}
        for m in models:
            lm = losses[m]
            if m == baseline:
                dm_stat, dm_p = np.nan, np.nan
                n_origins = len(np.unique(origin_ids))
            else:
                dm_stat, dm_p, n_origins = _dm(losses[m], losses[baseline], origin_ids, overlap)
            prev = [c for c in models if rung_of[m] and rung_of[c] and rung_of[c] < rung_of[m]]
            prev_best = min(prev, key=mean_qlike.get) if prev else None
            vs_prev = mean_qlike[m] / mean_qlike[prev_best] if prev_best else np.nan
            p_prev = (
                _dm(losses[m], losses[prev_best], origin_ids, overlap)[1] if prev_best else np.nan
            )
            ps = _per_symbol_means(lm, symbols)
            wins = int((ps["qlike"] < base_ps["qlike"]).sum())
            summaries.append(
                {
                    "horizon": hname,
                    "model": m,
                    "n_obs": int(mask.sum()),
                    "n_origins": int(n_origins),
                    "rung": rung_of[m],
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
                    "best_lower_rung": prev_best,
                    "QLIKE_vs_lower_rungs": vs_prev,
                    "DM_p_vs_lower_rungs": p_prev,
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


def _dm(
    loss_a: dict, loss_b: dict, origin_ids: np.ndarray, overlap: int
) -> tuple[float, float, int]:
    """DM test on the per-origin cross-sectional mean QLIKE differential (a - b)."""
    d_t = pd.Series(loss_a["qlike"] - loss_b["qlike"]).groupby(origin_ids).mean()
    lag = default_hac_lag(len(d_t), overlap)
    return (*diebold_mariano(d_t.to_numpy(), np.zeros(len(d_t)), lag), len(d_t))


def _per_symbol_means(losses: dict, symbols: pd.Index) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"symbol": np.asarray(symbols), **{k: losses[k] for k in ("qlike", "se", "ae")}}
    )
    out = frame.groupby("symbol").mean()
    out["n_obs"] = frame.groupby("symbol").size()
    return out


def plain_accuracy(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Easy-to-read accuracy per (horizon, model) on the common sample:

    * ``explained``: R^2 between log forecast and log realized variance;
    * ``median_vol_error``: median |forecast vol / realized vol - 1|;
    * ``within_25pct``: share of forecasts within +-25% of realized volatility.
    """
    rows = []
    for hname, f in forecasts.groupby("horizon", sort=False):
        wide = f.pivot_table(index=["origin_ts", "symbol"], columns="model", values="yhat")
        y = f.groupby(["origin_ts", "symbol"])["y"].first().reindex(wide.index)
        ok = np.isfinite(y) & (y > 0) & (wide > 0).all(axis=1)
        for m in dict.fromkeys(f["model"]):
            fc, yy = wide.loc[ok, m].to_numpy(), y[ok].to_numpy()
            err = np.abs(np.sqrt(fc / yy) - 1)
            rows.append(
                {
                    "horizon": hname,
                    "model": m,
                    "explained": float(np.corrcoef(np.log(yy), np.log(fc))[0, 1] ** 2),
                    "median_vol_error": float(np.median(err)),
                    "within_25pct": float(np.mean(err <= 0.25)),
                }
            )
    return pd.DataFrame(rows)
