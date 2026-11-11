"""Baselines: every later model must beat these out of sample.

All forecasts are of realized variance (RV, in %^2) over the next ``horizon`` bars.
Missing bars are skipped and the remaining average rescaled to the horizon, never
treated as zero-volatility bars.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from forecasting.data.panel import bars_per_session, n_symbols, panel_symbols, tail_matrix
from forecasting.models.base import ForecastModel, nanmean


class RandomWalk(ForecastModel):
    """Naive / random walk: next h-bar RV = RV over the last h observed bars.

    For the 30-minute target that is the last 30 minutes (which can reach back into the
    previous session's close at the open); for the next-session target it is the
    previous session's RV.
    """

    def __init__(self, name: str = "naive", max_lookback_sessions: int = 5) -> None:
        super().__init__(name)
        self.max_lookback_sessions = max_lookback_sessions

    def predict(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        b = bars_per_session(history_df)
        window = tail_matrix(history_df, "ret2", self.max_lookback_sessions * b + horizon)
        yhat = np.full(window.shape[1], np.nan)
        for j in range(window.shape[1]):
            col = window[:, j]
            last = col[~np.isnan(col)][-horizon:]
            if len(last) >= max(1, horizon // 2):
                yhat[j] = last.mean() * horizon
        return self.forecast_frame(history_df, yhat)


class HistoricalMean(ForecastModel):
    """Historical mean: average per-bar variance in the training window times h.

    Re-estimated at every walk-forward refit; ``window_sessions=None`` uses the whole
    (expanding) training window. Ignores time of day on purpose.
    """

    def __init__(self, name: str = "hist_mean", window_sessions: int | None = None) -> None:
        super().__init__(name)
        self.window_sessions = window_sessions

    def fit(self, train_df: pd.DataFrame) -> Self:
        b = bars_per_session(train_df)
        n = len(train_df) // n_symbols(train_df)
        if self.window_sessions is not None:
            n = min(n, self.window_sessions * b)
        self.mean_ = nanmean(tail_matrix(train_df, "ret2", n), axis=0)
        self.symbols_ = panel_symbols(train_df)
        return self

    def predict(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if panel_symbols(history_df) != self.symbols_:
            raise ValueError("history symbols differ from the fitted symbols")
        return self.forecast_frame(history_df, self.mean_ * horizon)


class SeasonalNaive(ForecastModel):
    """Seasonal naive: RV of the same time-of-day window in the previous session(s).

    With ``n_sessions=1`` this is the classic seasonal naive (yesterday, same clock time).
    With ``n_sessions=k`` each slot is averaged over the last k sessions, which captures
    the intraday U-shape with less noise. For a full-session horizon with
    ``n_sessions=1`` it equals the random walk on complete data; they differ only in
    how missing bars are handled (the random walk reaches further back for observed
    bars, this model rescales the observed part of the same window).
    """

    def __init__(self, name: str = "seasonal_naive", n_sessions: int = 1) -> None:
        super().__init__(name)
        if n_sessions < 1:
            raise ValueError("n_sessions must be >= 1")
        self.n_sessions = n_sessions

    def predict(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        b = bars_per_session(history_df)
        if horizon > b:
            raise ValueError(f"horizon {horizon} exceeds one session ({b} bars)")
        k = min(self.n_sessions, (len(history_df) // n_symbols(history_df)) // b)
        if k == 0:
            return self.forecast_frame(history_df, np.full(n_symbols(history_df), np.nan))
        # The tail of k*b bars starts exactly k sessions before the origin, at the
        # origin's own slot; session d-j's matching window starts at (k-j)*b.
        tail = tail_matrix(history_df, "ret2", k * b)
        windows = np.stack([tail[(k - j) * b : (k - j) * b + horizon] for j in range(1, k + 1)])
        per_slot = nanmean(windows, axis=0)  # (horizon, symbols)
        yhat = nanmean(per_slot, axis=0, min_count=max(1, horizon // 2)) * horizon
        return self.forecast_frame(history_df, yhat)
