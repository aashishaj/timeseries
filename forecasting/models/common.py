"""Building blocks shared by model families: intraday seasonality and daily RV.

All functions take a panel slice and use only the rows they are given, so a model
that calls them inside ``fit``/``predict`` stays leakage-free by construction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.data.panel import bars_per_session, n_symbols, tail_matrix
from forecasting.models.base import nanmean


def seasonal_profile(train_df: pd.DataFrame, pooled: bool = True) -> np.ndarray:
    """Average variance by time-of-day slot relative to the day's average, shape (B, S).

    Each column has mean 1 over the session (so ``profile[k:k+h].sum()`` is the
    expected share of h average bars). ``pooled=True`` averages the normalized shape
    across symbols (far less noisy with a few months of data) and uses it for all.
    """
    b, s = bars_per_session(train_df), n_symbols(train_df)
    n_days = len(train_df) // (b * s)
    x = tail_matrix(train_df, "ret2", n_days * b).reshape(n_days, b, s)
    # Normalize each day first so high-vol days do not dominate the shape.
    day_mean = nanmean(x, axis=1, min_count=b // 2)  # (D, S)
    with np.errstate(invalid="ignore", divide="ignore"):
        rel = x / day_mean[:, None, :]
    prof = nanmean(rel, axis=0)  # (B, S)
    if pooled:
        prof = np.repeat(nanmean(prof, axis=1)[:, None], s, axis=1)
    prof = np.where(np.isfinite(prof) & (prof > 0), prof, np.nan)
    fill = nanmean(prof, axis=0)
    prof = np.where(np.isnan(prof), fill[None, :], prof)
    return prof / prof.mean(axis=0, keepdims=True)


def daily_rv(
    history_df: pd.DataFrame,
    profile: np.ndarray | None = None,
    min_coverage: float = 0.8,
) -> tuple[np.ndarray, np.ndarray]:
    """Realized variance of every *completed* session, plus today's bars so far.

    Missing bars are filled in expectation: observed RV is divided by the expected
    share of the day those bars represent (from ``profile``; uniform if None). Days
    with less than ``min_coverage`` of their expected variance observed are NaN.

    Returns:
        rv: (n_completed_sessions, S)
        today: (k, S) squared returns of the current, partial session (k may be 0)
    """
    b, s = bars_per_session(history_df), n_symbols(history_df)
    n_bars = len(history_df) // s
    k = n_bars % b
    x = tail_matrix(history_df, "ret2", n_bars)
    full, today = x[: n_bars - k], x[n_bars - k :]
    days = full.reshape(-1, b, s)
    w = np.ones((b, s)) / b if profile is None else profile / profile.sum(axis=0, keepdims=True)
    observed = ~np.isnan(days)
    share = (observed * w[None]).sum(axis=1)  # (D, S) expected share actually observed
    total = np.where(observed, days, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = np.where(share >= min_coverage, total / share, np.nan)
    return rv, today


def har_features(rv: np.ndarray, lags: tuple[int, ...]) -> np.ndarray:
    """HAR regressors for every day t: mean RV over the last l days (t-l+1..t), shape
    (D, len(lags), S); NaN until the longest window is half observed."""
    d, s = rv.shape
    out = np.full((d, len(lags), s), np.nan)
    valid = ~np.isnan(rv)
    cs = np.vstack([np.zeros((1, s)), np.cumsum(np.where(valid, rv, 0.0), axis=0)])
    cn = np.vstack([np.zeros((1, s)), np.cumsum(valid, axis=0)])
    for i, lag in enumerate(lags):
        hi = np.arange(1, d + 1)
        lo = np.maximum(hi - lag, 0)
        total, count = cs[hi] - cs[lo], cn[hi] - cn[lo]
        ok = (hi >= lag)[:, None] & (count >= max(1, lag // 2))
        with np.errstate(invalid="ignore", divide="ignore"):
            out[:, i] = np.where(ok, total / count, np.nan)
    return out
