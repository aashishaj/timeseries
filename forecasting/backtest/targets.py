"""Forecast targets. These look INTO THE FUTURE by definition: they are computed by the
engine for scoring only and must never be passed to a model or used as a feature."""

from __future__ import annotations

import numpy as np


def realized_variance(ret2: np.ndarray, starts: np.ndarray, horizon: int) -> np.ndarray:
    """RV over bars ``[start, start + horizon)`` for every start and symbol.

    Args:
        ret2: (n_bars, n_symbols) squared percent log returns.
        starts: bar positions where each target window begins (the forecast origin).
        horizon: window length in bars.

    Returns:
        (len(starts), n_symbols) array; NaN if any bar in the window is missing or the
        window runs past the end of the data.
    """
    n = len(ret2)
    valid = ~np.isnan(ret2)
    zero = np.zeros((1, ret2.shape[1]))
    cs = np.vstack([zero, np.cumsum(np.where(valid, ret2, 0.0), axis=0)])
    cn = np.vstack([zero, np.cumsum(valid, axis=0)])
    starts = np.asarray(starts, dtype=int)
    ends = np.minimum(starts + horizon, n)
    total = cs[ends] - cs[starts]
    complete = (cn[ends] - cn[starts] == horizon) & ((starts + horizon) <= n)[:, None]
    return np.where(complete, total, np.nan)
