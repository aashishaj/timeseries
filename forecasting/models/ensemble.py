"""Forecast combinations: build our own model by mixing others.

``Combination`` wraps any registered models and combines their variance forecasts
geometrically (a weighted mean of log forecasts), which suits QLIKE and keeps the
result positive:

    log yhat = c + sum_k w_k log f_k,   w_k >= 0, sum_k w_k = 1

* ``method="equal"``: equal weights, c = 0. Simple averages are famously hard to beat
  (the "forecast combination puzzle").
* ``method="qlike_opt"``: weights and the bias term c minimize QLIKE on a validation
  window carved from the *end of the training data*: members are refit on the data
  before that window and forecast it out of sample. Test data is never used.

If a member has no forecast for a symbol, the remaining weights are renormalized.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from forecasting.backtest.metrics import qlike
from forecasting.backtest.targets import realized_variance
from forecasting.data.panel import bars_per_session, n_symbols, panel_symbols, tail_matrix
from forecasting.models.base import ForecastModel


def combine(log_f: np.ndarray, weights: np.ndarray, bias: float = 0.0) -> np.ndarray:
    """Geometric combination over the member axis (axis -2 of ``log_f``: (..., K, S)),
    renormalizing weights over the members that have a forecast."""
    ok = np.isfinite(log_f)
    w = np.where(ok, weights[:, None], 0.0)
    total = w.sum(axis=-2)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = (np.where(ok, log_f, 0.0) * w).sum(axis=-2) / total
    return np.where(total > 0, np.exp(mean + bias), np.nan)


class Combination(ForecastModel):
    def __init__(
        self,
        name: str = "combo",
        members: list[dict] | None = None,
        method: str = "equal",
        validation_sessions: int = 10,
        validation_stride: int = 3,
    ) -> None:
        super().__init__(name)
        if not members:
            raise ValueError("a combination needs at least one member")
        if method not in {"equal", "qlike_opt"}:
            raise ValueError(f"unknown combination method {method!r}")
        self.members = [dict(m) for m in members]
        self.method = method
        self.validation_sessions = validation_sessions
        self.validation_stride = validation_stride

    def _build(self) -> list[ForecastModel]:
        from forecasting.models import build_model

        return [
            build_model(m["type"], m.get("name", m["type"]), m.get("params")) for m in self.members
        ]

    def fit(self, train_df: pd.DataFrame) -> Self:
        self.models_ = [m.fit(train_df) for m in self._build()]
        self.symbols_ = panel_symbols(train_df)
        self.train_ = train_df
        self.weights_: dict[int, tuple[np.ndarray, float]] = {}
        return self

    # ------------------------------------------------------------------ weights
    def _validation_forecasts(self, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        """Out-of-sample member forecasts on the last training sessions: (n, K, S) and y."""
        train = self.train_
        s, b = n_symbols(train), bars_per_session(train)
        n_days = len(train) // (s * b)
        v = min(self.validation_sessions, n_days // 3)
        if v < 1:
            return np.empty((0, len(self.members), s)), np.empty((0, s))
        inner_train = train.iloc[: (n_days - v) * b * s]
        inner = [m.fit(inner_train) for m in self._build()]
        slots = [0] if horizon == b else range(0, b - horizon + 1, self.validation_stride)
        positions = np.array([d * b + k for d in range(n_days - v, n_days) for k in slots])
        positions = positions[positions + horizon <= n_days * b]
        forecasts = np.stack(
            [
                np.stack(
                    [m.predict(train.iloc[: p * s], horizon)["yhat"].to_numpy() for m in inner]
                )
                for p in positions
            ]
        )
        y = realized_variance(tail_matrix(train, "ret2", n_days * b), positions, horizon)
        return forecasts, y

    def _fit_weights(self, horizon: int) -> tuple[np.ndarray, float]:
        k = len(self.members)
        equal = (np.full(k, 1.0 / k), 0.0)
        if self.method == "equal":
            return equal
        from scipy.optimize import minimize

        forecasts, y = self._validation_forecasts(horizon)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_f = np.log(np.where(forecasts > 0, forecasts, np.nan))
        valid = np.isfinite(y) & (y > 0) & np.isfinite(log_f).any(axis=1)
        if valid.sum() < 20:
            return equal

        def loss(theta: np.ndarray) -> float:
            w = np.exp(theta[:k] - theta[:k].max())
            w /= w.sum()
            fc = combine(log_f, w, theta[k])
            return float(np.mean(qlike(y[valid], fc[valid])))

        res = minimize(loss, np.zeros(k + 1), method="L-BFGS-B")
        w = np.exp(res.x[:k] - res.x[:k].max())
        return w / w.sum(), float(res.x[k])

    # ------------------------------------------------------------------ predict
    def predict(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if horizon not in self.weights_:
            self.weights_[horizon] = self._fit_weights(horizon)
        w, bias = self.weights_[horizon]
        f = np.stack([m.predict(history_df, horizon)["yhat"].to_numpy() for m in self.models_])
        with np.errstate(divide="ignore", invalid="ignore"):
            log_f = np.log(np.where(f > 0, f, np.nan))
        return self.forecast_frame(history_df, combine(log_f, w, bias))
