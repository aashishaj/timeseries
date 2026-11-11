"""Realized-variance style features built from ``ret2``. All strictly backward-looking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.data.panel import n_symbols, to_matrix


def _window_sum(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Trailing window sum and count of non-NaN values along axis 0 (inclusive of row t)."""
    valid = ~np.isnan(x)
    cs = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(np.where(valid, x, 0.0), axis=0)])
    cn = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(valid, axis=0)])
    lo = np.maximum(np.arange(1, len(x) + 1) - window, 0)
    hi = np.arange(1, len(x) + 1)
    return cs[hi] - cs[lo], cn[hi] - cn[lo]


def trailing_rv(panel: pd.DataFrame, window_bars: int, min_valid: float = 0.8) -> pd.Series:
    """Realized variance over the last ``window_bars`` bars including the current one.

    Missing returns are skipped and the sum rescaled (mean * window); NaN if fewer than
    ``min_valid`` of the window is observed or the window is not yet full.
    """
    x = to_matrix(panel, "ret2")
    total, count = _window_sum(x, window_bars)
    full = (np.arange(len(x)) >= window_bars - 1)[:, None]
    ok = full & (count >= np.ceil(min_valid * window_bars))
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = np.where(ok, total / count * window_bars, np.nan)
    return pd.Series(rv.ravel(), index=panel.index, name=f"rv_trailing_{window_bars}")


def session_rv_to_date(panel: pd.DataFrame) -> pd.Series:
    """Realized variance of the current session up to and including this bar (NaNs skipped)."""
    s = n_symbols(panel)
    x = np.nan_to_num(to_matrix(panel, "ret2"), nan=0.0)
    session_id = np.cumsum(panel["slot"].to_numpy()[::s] == 0)
    out = np.empty_like(x)
    for sid in np.unique(session_id):
        rows = session_id == sid
        out[rows] = np.cumsum(x[rows], axis=0)
    return pd.Series(out.ravel(), index=panel.index, name="rv_session_to_date")
