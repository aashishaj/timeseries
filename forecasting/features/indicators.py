"""Technical and volatility indicators as (n_bars, n_symbols) matrices.

Every indicator at bar t uses bars <= t only and finite windows only (no
exponential smoothing with infinite memory), so computing it on a recent tail of
the history gives exactly the same value as computing it on the full history.
That is what lets a model recompute indicators cheaply at every forecast origin.

Conventions: prices are matrices from the panel; ``ret``/``ret2`` are percent log
returns; windows count bars and may span sessions unless the name says "session".
All are leakage-tested in ``tests/test_indicators.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.data.panel import bars_per_session, n_symbols, to_matrix

LN2 = np.log(2.0)


def rolling_mean(x: np.ndarray, window: int, min_frac: float = 0.5) -> np.ndarray:
    """Trailing mean over ``window`` rows (inclusive), NaN-skipping; NaN if too few values."""
    valid = ~np.isnan(x)
    zero = np.zeros((1, x.shape[1]))
    cs = np.vstack([zero, np.cumsum(np.where(valid, x, 0.0), axis=0)])
    cn = np.vstack([zero, np.cumsum(valid, axis=0)])
    hi = np.arange(1, len(x) + 1)
    lo = np.maximum(hi - window, 0)
    total, count = cs[hi] - cs[lo], cn[hi] - cn[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count >= max(1, int(np.ceil(min_frac * window))), total / count, np.nan)


def rolling_std(x: np.ndarray, window: int, min_frac: float = 0.5) -> np.ndarray:
    mean = rolling_mean(x, window, min_frac)
    mean_sq = rolling_mean(x * x, window, min_frac)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def parkinson(panel: pd.DataFrame) -> np.ndarray:
    """Per-bar Parkinson (high-low) variance estimate, in %^2."""
    hl = np.log(to_matrix(panel, "high") / to_matrix(panel, "low"))
    return 1e4 * hl * hl / (4.0 * LN2)


def garman_klass(panel: pd.DataFrame) -> np.ndarray:
    """Per-bar Garman-Klass variance estimate (uses open/high/low/close), in %^2."""
    hl = np.log(to_matrix(panel, "high") / to_matrix(panel, "low"))
    co = np.log(to_matrix(panel, "close") / to_matrix(panel, "open"))
    return 1e4 * np.maximum(0.5 * hl * hl - (2.0 * LN2 - 1.0) * co * co, 0.0)


def true_range_pct(panel: pd.DataFrame) -> np.ndarray:
    """True range as % of the previous close (within a session; first bar uses high-low)."""
    high, low, close = (to_matrix(panel, c) for c in ("high", "low", "close"))
    prev = np.vstack([np.full((1, close.shape[1]), np.nan), close[:-1]])
    first = to_matrix(panel, "slot")[:, :1].ravel() == 0
    prev[first] = np.nan
    tr = np.fmax(high - low, np.fmax(np.abs(high - prev), np.abs(low - prev)))
    base = np.where(np.isnan(prev), close, prev)
    return 100.0 * tr / base


def atr_pct(panel: pd.DataFrame, window: int = 14) -> np.ndarray:
    return rolling_mean(true_range_pct(panel), window)


def bollinger_width(panel: pd.DataFrame, window: int = 20) -> np.ndarray:
    """(upper - lower) / middle band with 2 standard deviations, in %."""
    close = to_matrix(panel, "close")
    return 100.0 * 4.0 * rolling_std(close, window) / rolling_mean(close, window)


def rsi(panel: pd.DataFrame, window: int = 14) -> np.ndarray:
    """Relative strength index (simple-average version), 0..100."""
    r = to_matrix(panel, "ret")
    gain = rolling_mean(np.where(np.isnan(r), np.nan, np.maximum(r, 0.0)), window)
    loss = rolling_mean(np.where(np.isnan(r), np.nan, np.maximum(-r, 0.0)), window)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(gain + loss > 0, 100.0 * gain / (gain + loss), 50.0)


def relative_volume(panel: pd.DataFrame, n_sessions: int = 5) -> np.ndarray:
    """log(volume / mean volume at the same slot over the previous n sessions)."""
    b = bars_per_session(panel)
    vol = to_matrix(panel, "volume").astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        logv = np.log(np.where(vol > 0, vol, np.nan))
    lags = np.full((n_sessions, *logv.shape), np.nan)
    for j in range(1, n_sessions + 1):
        if len(logv) > j * b:
            lags[j - 1, j * b :] = logv[: len(logv) - j * b]
    count = (~np.isnan(lags)).sum(axis=0)
    total = np.where(np.isnan(lags), 0.0, lags).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return logv - np.where(count > 0, total / count, np.nan)


def previous_session_rv(panel: pd.DataFrame, n_sessions: int = 1) -> np.ndarray:
    """Mean RV of the last ``n_sessions`` *completed* sessions, repeated on every bar of the
    current session (known from the open). NaN-bars are skipped and rescaled."""
    b, s = bars_per_session(panel), n_symbols(panel)
    x = to_matrix(panel, "ret2")
    n = len(x)
    n_days = -(-n // b)
    padded = np.full((n_days * b, s), np.nan)
    padded[:n] = x
    days = padded.reshape(n_days, b, s)
    observed = (~np.isnan(days)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        rv = np.where(observed >= 0.8 * b, np.nansum(days, axis=1) * b / observed, np.nan)
    lagged = np.full_like(rv, np.nan)
    lagged[1:] = rolling_mean(rv, n_sessions)[:-1]  # sessions strictly before day d
    return np.repeat(lagged, b, axis=0)[:n]


def indicator_matrices(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """All indicators used by the ML models, each (n_bars, n_symbols)."""
    ret = to_matrix(panel, "ret")
    ret2 = to_matrix(panel, "ret2")
    return {
        "rv_6": rolling_mean(ret2, 6) * 6,
        "rv_12": rolling_mean(ret2, 12) * 12,
        "rv_36": rolling_mean(ret2, 36) * 36,
        "rv_75": rolling_mean(ret2, 75) * 75,
        "rv_prev_session": previous_session_rv(panel, 1),
        "rv_prev_5_sessions": previous_session_rv(panel, 5),
        "rv_prev_22_sessions": previous_session_rv(panel, 22),
        "parkinson_6": rolling_mean(parkinson(panel), 6) * 6,
        "garman_klass_12": rolling_mean(garman_klass(panel), 12) * 12,
        "abs_ret_last": np.abs(ret),
        "ret_last": ret,
        "ret_sum_6": rolling_mean(ret, 6) * 6,
        "max_abs_ret_12": _rolling_max(np.abs(ret), 12),
        "atr_14": atr_pct(panel, 14),
        "bollinger_20": bollinger_width(panel, 20),
        "rsi_14": rsi(panel, 14),
        "rel_volume": relative_volume(panel, 5),
        "rel_volume_6": rolling_mean(relative_volume(panel, 5), 6),
    }


def _rolling_max(x: np.ndarray, window: int) -> np.ndarray:
    """Trailing max over ``window`` rows, NaN-skipping (NaN if the window is all NaN)."""
    padded = np.vstack([np.full((window - 1, x.shape[1]), np.nan), x])
    view = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)  # (n, S, w)
    out = np.where(np.isnan(view), -np.inf, view).max(axis=2)
    return np.where(np.isneginf(out), np.nan, out)
