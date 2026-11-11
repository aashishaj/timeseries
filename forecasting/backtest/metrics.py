"""Forecast evaluation metrics.

Volatility losses are computed on realized *variance* (Patton 2011): MSE and QLIKE
are the two common losses whose model rankings stay consistent when the true
variance is replaced by a noisy proxy such as RV.
"""

from __future__ import annotations

import math

import numpy as np


def qlike(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    """QLIKE loss ``y/yhat - log(y/yhat) - 1`` per observation (0 at a perfect forecast).

    Requires ``y > 0`` and ``yhat > 0``. Penalises under-prediction of variance more
    than over-prediction and is scale-free, so calm and volatile periods weigh equally.
    """
    ratio = np.asarray(y, dtype=float) / np.asarray(yhat, dtype=float)
    return ratio - np.log(ratio) - 1.0


def squared_error(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return (np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float)) ** 2


def absolute_error(y: np.ndarray, yhat: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(y, dtype=float) - np.asarray(yhat, dtype=float))


def mase(y: np.ndarray, yhat: np.ndarray, yhat_naive: np.ndarray) -> float:
    """MAE scaled by the naive forecast's MAE on the *same* out-of-sample origins.

    < 1 beats naive; the naive model scores exactly 1.
    """
    denom = float(np.mean(absolute_error(y, yhat_naive)))
    return float(np.mean(absolute_error(y, yhat))) / denom if denom > 0 else math.nan


def directional_accuracy(y: np.ndarray, yhat: np.ndarray) -> float:
    """Share of observations where the forecast has the right sign (zeros in ``y`` skipped)."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    keep = (y != 0) & np.isfinite(y) & np.isfinite(yhat)
    return float(np.mean(np.sign(y[keep]) == np.sign(yhat[keep]))) if keep.any() else math.nan


def newey_west_variance(x: np.ndarray, lag: int) -> float:
    """Long-run variance of ``x`` with a Bartlett kernel."""
    u = np.asarray(x, dtype=float) - np.mean(x)
    n = len(u)
    lrv = float(u @ u) / n
    for k in range(1, min(lag, n - 1) + 1):
        lrv += 2.0 * (1.0 - k / (lag + 1)) * float(u[k:] @ u[:-k]) / n
    return lrv


def default_hac_lag(n: int, overlap: int = 0) -> int:
    """Newey-West rule-of-thumb lag, at least the target-window overlap."""
    return max(int(overlap), int(math.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, lag: int) -> tuple[float, float]:
    """Diebold-Mariano test of equal predictive accuracy.

    Statistic on ``d = loss_a - loss_b`` with a Newey-West variance; negative means
    ``a`` has lower loss. Returns (statistic, two-sided normal p-value).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    if len(d) < 10:
        return math.nan, math.nan
    lrv = newey_west_variance(d, lag)
    if lrv <= 0:
        return math.nan, math.nan
    stat = float(np.mean(d) / math.sqrt(lrv / len(d)))
    return stat, math.erfc(abs(stat) / math.sqrt(2.0))
