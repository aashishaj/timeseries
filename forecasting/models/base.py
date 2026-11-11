"""Common model interface. Every model family plugs into the backtest engine through it.

Contract
--------
``fit(train_df) -> self``
    ``train_df`` is a panel slice (see :mod:`forecasting.data.panel`) containing only
    bars strictly before the first forecast origin of the fold.

``predict(history_df, horizon) -> forecast_df``
    ``history_df`` is a panel slice holding every bar that has *ended* by the forecast
    origin (and nothing after it). ``horizon`` is the number of bars ahead. The forecast
    is the realized variance (sum of squared percent log returns) over the next
    ``horizon`` bars. ``forecast_df`` has one row per symbol with columns
    ``symbol`` and ``yhat``; NaN means "no forecast" and the origin is then excluded
    from every model's evaluation.

Models never see targets. The engine computes them separately from future bars.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self

import numpy as np
import pandas as pd

from forecasting.data.panel import panel_symbols


class ForecastModel(ABC):
    """Base class: subclasses implement :meth:`predict` and optionally :meth:`fit`."""

    def __init__(self, name: str | None = None) -> None:
        self.name = name or type(self).__name__

    def fit(self, train_df: pd.DataFrame) -> Self:
        return self

    @abstractmethod
    def predict(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame: ...

    def get_params(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.endswith("_") and k != "name"}

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}(name={self.name!r}{', ' if params else ''}{params})"

    @staticmethod
    def forecast_frame(history_df: pd.DataFrame, yhat: np.ndarray) -> pd.DataFrame:
        """Build the standard forecast frame in the panel's symbol order."""
        return pd.DataFrame({"symbol": panel_symbols(history_df), "yhat": yhat})


def nanmean(x: np.ndarray, axis: int = 0, min_count: int = 1) -> np.ndarray:
    """NaN-skipping mean without RuntimeWarnings; NaN where fewer than ``min_count`` values."""
    valid = ~np.isnan(x)
    count = valid.sum(axis=axis)
    total = np.where(valid, x, 0.0).sum(axis=axis)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count >= max(min_count, 1), total / count, np.nan)
