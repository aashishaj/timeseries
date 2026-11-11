"""GARCH-family models on seasonally filtered 5-minute returns.

Intraday returns have a strong time-of-day pattern, which plain GARCH would mistake
for volatility clustering. Following Andersen & Bollerslev (1997) we fit GARCH to
``r_t / sqrt(profile[slot_t])`` and multiply the profile back into the forecast:

    RV forecast over bars t+1..t+h = sum_j profile[slot_{t+j}] * E[sigma^2_{t+j}]

Parameters are estimated with ``arch`` at every refit; inside a test block the
variance recursion is run forward incrementally with those fixed parameters.
"""

from __future__ import annotations

import warnings
from typing import Self

import numpy as np
import pandas as pd

from forecasting.data.panel import bars_per_session, n_symbols, panel_symbols, tail_matrix
from forecasting.models.base import ForecastModel
from forecasting.models.common import seasonal_profile


class SeasonalGARCH(ForecastModel):
    """GARCH(1,1) (``asymmetric=False``) or GJR-GARCH(1,1) per symbol.

    Args:
        asymmetric: add the GJR leverage term (bad news raises volatility more).
        dist: innovation distribution for estimation ("normal", "t", "skewt").
        fit_window_sessions: estimate on at most this many recent training sessions.
        burn_in_sessions: sessions of history used to warm up the variance recursion.
        min_obs: symbols with fewer training returns get no forecast (NaN).
    """

    def __init__(
        self,
        name: str = "garch",
        asymmetric: bool = False,
        dist: str = "t",
        fit_window_sessions: int | None = 120,
        burn_in_sessions: int = 20,
        pooled_seasonality: bool = True,
        min_obs: int = 500,
    ) -> None:
        super().__init__(name)
        self.min_obs = min_obs
        self.asymmetric = asymmetric
        self.dist = dist
        self.fit_window_sessions = fit_window_sessions
        self.burn_in_sessions = burn_in_sessions
        self.pooled_seasonality = pooled_seasonality

    def fit(self, train_df: pd.DataFrame) -> Self:
        from arch import arch_model

        s, b = n_symbols(train_df), bars_per_session(train_df)
        self.profile_ = seasonal_profile(train_df, self.pooled_seasonality)
        self.symbols_ = panel_symbols(train_df)
        n_days = len(train_df) // (s * b)
        if self.fit_window_sessions:
            n_days = min(n_days, self.fit_window_sessions)
        r = tail_matrix(train_df, "ret", n_days * b)
        z = r / np.sqrt(np.tile(self.profile_, (n_days, 1)))
        self.omega_ = np.full(s, np.nan)
        self.alpha_ = np.zeros(s)
        self.gamma_ = np.zeros(s)
        self.beta_ = np.zeros(s)
        self.scale_ = np.ones(s)
        for j in range(s):
            zj = z[:, j][~np.isnan(z[:, j])]
            if len(zj) < self.min_obs:
                continue
            scale = 1.0 / np.std(zj)  # fit on unit-variance data; arch is scale-sensitive
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = arch_model(
                    zj * scale,
                    mean="Zero",
                    vol="GARCH",
                    p=1,
                    o=1 if self.asymmetric else 0,
                    q=1,
                    dist=self.dist,
                    rescale=False,
                ).fit(disp="off", show_warning=False)
            params = res.params
            self.scale_[j] = scale
            self.omega_[j] = params["omega"]
            self.alpha_[j] = params["alpha[1]"]
            self.beta_[j] = params["beta[1]"]
            self.gamma_[j] = params.get("gamma[1]", 0.0)
        self.state_: tuple[int, pd.Timestamp, np.ndarray] | None = None
        return self

    @property
    def persistence_(self) -> np.ndarray:
        return np.minimum(self.alpha_ + self.beta_ + 0.5 * self.gamma_, 0.9999)

    def _next_variance(self, history_df: pd.DataFrame) -> np.ndarray:
        """sigma^2 for the next bar (in scaled, de-seasonalized units), per symbol."""
        s, b = n_symbols(history_df), bars_per_session(history_df)
        n_bars = len(history_df) // s
        start, var = 0, None
        if self.state_ is not None:
            seen, last_ts, cached = self.state_
            if seen <= n_bars and history_df["ts"].iat[(seen - 1) * s] == last_ts:
                start, var = seen, cached.copy()
        if var is None:
            start = max(0, n_bars - self.burn_in_sessions * b)
            start -= start % b
            var = self.omega_ / (1.0 - self.persistence_)  # unconditional variance
        r = tail_matrix(history_df, "ret", n_bars - start)
        slots = (start + np.arange(len(r))) % b
        for row, slot in zip(r, slots, strict=True):
            eps = row / np.sqrt(self.profile_[slot]) * self.scale_
            ok = ~np.isnan(eps)
            e2 = np.where(ok, eps * eps, 0.0)
            new = self.omega_ + (self.alpha_ + self.gamma_ * (eps < 0)) * e2 + self.beta_ * var
            var = np.where(ok, new, var)
        if n_bars:
            self.state_ = (n_bars, history_df["ts"].iat[(n_bars - 1) * s], var.copy())
        return var

    def predict(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        b = bars_per_session(history_df)
        k = (len(history_df) // n_symbols(history_df)) % b
        nxt = self._next_variance(history_df)
        rho = self.persistence_
        uncond = self.omega_ / (1.0 - rho)
        steps = np.arange(horizon)[:, None]
        path = uncond + rho[None, :] ** steps * (nxt - uncond)[None, :]  # E[sigma^2_{t+j}]
        prof = self.profile_[(k + np.arange(horizon)) % b]
        yhat = (prof * path).sum(axis=0) / self.scale_**2
        return self.forecast_frame(history_df, yhat)
