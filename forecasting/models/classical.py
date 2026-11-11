"""Classical realized-volatility models: seasonal EWMA and HAR-RV (with an intraday mix).

Both forecast the variance of the next ``horizon`` bars as

    (daily or per-bar variance level) x (time-of-day profile over those bars)

which is the multiplicative component idea of Andersen & Bollerslev (1997) and
Engle & Sokalska (2012): a slow daily level, a fixed intraday shape, and (optionally)
a fast intraday correction for how today is going so far.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from forecasting.data.panel import bars_per_session, n_symbols, panel_symbols, tail_matrix
from forecasting.models.base import ForecastModel, nanmean
from forecasting.models.common import daily_rv, har_features, seasonal_profile


def _window_profile(profile: np.ndarray, k: int, horizon: int) -> np.ndarray:
    """Sum of the profile over slots k..k+horizon-1 (wrapping into the next session)."""
    b = profile.shape[0]
    return profile[(k + np.arange(horizon)) % b].sum(axis=0)


class SeasonalEWMA(ForecastModel):
    """RiskMetrics-style EWMA on de-seasonalized squared 5-min returns.

    ``level`` tracks the variance of an "average" bar; the forecast multiplies it by
    the time-of-day profile over the target window. State is updated incrementally as
    the history grows, so each origin costs only the bars added since the last one.
    """

    def __init__(
        self, name: str = "ewma", halflife_sessions: float = 1.0, pooled_seasonality: bool = True
    ) -> None:
        super().__init__(name)
        self.halflife_sessions = halflife_sessions
        self.pooled_seasonality = pooled_seasonality

    def fit(self, train_df: pd.DataFrame) -> Self:
        self.profile_ = seasonal_profile(train_df, self.pooled_seasonality)
        self.symbols_ = panel_symbols(train_df)
        self.state_: tuple[int, pd.Timestamp, np.ndarray] | None = None
        return self

    def _level(self, history_df: pd.DataFrame) -> np.ndarray:
        s, b = n_symbols(history_df), bars_per_session(history_df)
        n_bars = len(history_df) // s
        alpha = 1.0 - 0.5 ** (1.0 / (self.halflife_sessions * b))
        start, level = 0, None
        if self.state_ is not None:
            seen, last_ts, cached = self.state_
            if seen <= n_bars and history_df["ts"].iat[(seen - 1) * s] == last_ts:
                start, level = seen, cached.copy()
        if level is None:
            # Older bars carry < 0.1% weight; start ~10 half-lives back at a session start.
            start = max(0, n_bars - int(10 * self.halflife_sessions + 1) * b)
            start -= start % b
            first = tail_matrix(history_df, "ret2", n_bars)[start : start + b]
            level = nanmean(first / self.profile_, axis=0)
        x = tail_matrix(history_df, "ret2", n_bars - start)
        slots = (start + np.arange(len(x))) % b
        for row, slot in zip(x, slots, strict=True):
            u = row / self.profile_[slot]
            ok = ~np.isnan(u)
            level[ok] += alpha * (u[ok] - level[ok])
        if n_bars:
            self.state_ = (n_bars, history_df["ts"].iat[(n_bars - 1) * s], level.copy())
        return level

    def predict(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        b = bars_per_session(history_df)
        k = (len(history_df) // n_symbols(history_df)) % b
        yhat = self._level(history_df) * _window_profile(self.profile_, k, horizon)
        return self.forecast_frame(history_df, yhat)


class DailyComponentModel(ForecastModel):
    """Daily RV forecast x time-of-day share x optional intraday correction.

    Subclasses provide a daily model through :meth:`_fit_daily` and
    :meth:`_forecast_daily`; this class turns it into forecasts for any horizon up to a
    session:

    * next-session horizon: the daily forecast of the next session's RV;
    * intraday horizons: today's daily forecast x the time-of-day share of the target
      window x ``(realized so far / expected so far) ** intraday_beta`` (clipped to
      [1/4, 4]; ``intraday_beta=0`` disables it).
    """

    history_days: int = 60  # completed sessions passed to _forecast_daily

    def __init__(
        self,
        name: str,
        intraday_beta: float = 0.0,
        pooled_seasonality: bool = True,
        min_coverage: float = 0.8,
    ) -> None:
        super().__init__(name)
        self.intraday_beta = intraday_beta
        self.pooled_seasonality = pooled_seasonality
        self.min_coverage = min_coverage

    def _fit_daily(self, rv: np.ndarray) -> None:
        raise NotImplementedError

    def _forecast_daily(self, rv: np.ndarray) -> np.ndarray:
        """Forecast of the next session's RV given completed-session RVs (D, S)."""
        raise NotImplementedError

    def fit(self, train_df: pd.DataFrame) -> Self:
        self.profile_ = seasonal_profile(train_df, self.pooled_seasonality)
        self.symbols_ = panel_symbols(train_df)
        rv, _ = daily_rv(train_df, self.profile_, self.min_coverage)
        self._fit_daily(rv)
        self.cache_: tuple[int, pd.Timestamp, np.ndarray] | None = None
        return self

    def _today_forecast(self, history_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        s, b = n_symbols(history_df), bars_per_session(history_df)
        n_bars = len(history_df) // s
        k = n_bars % b
        need = (self.history_days + 1) * b + k
        tail = history_df.iloc[-min(need, n_bars) * s :] if n_bars else history_df
        rv, today = daily_rv(tail, self.profile_, self.min_coverage)
        n_days = n_bars // b
        key_ts = history_df["ts"].iat[(n_days * b - 1) * s] if n_days else None
        if self.cache_ is not None and self.cache_[:2] == (n_days, key_ts):
            return self.cache_[2], today
        fc = self._forecast_daily(rv) if len(rv) else np.full(s, np.nan)
        self.cache_ = (n_days, key_ts, fc)
        return fc, today

    def predict(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if horizon > bars_per_session(history_df):
            raise ValueError(f"{type(self).__name__} forecasts at most one session ahead")
        fc_day, today = self._today_forecast(history_df)
        k = len(today)
        share = self.profile_ / self.profile_.sum(axis=0, keepdims=True)
        yhat = fc_day * _window_profile(share, k, horizon)
        if k and self.intraday_beta:
            observed = ~np.isnan(today)
            expected = fc_day * (observed * share[:k]).sum(axis=0)
            realized = np.where(observed, today, 0.0).sum(axis=0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.clip(realized / expected, 0.25, 4.0)
            yhat = yhat * np.where(np.isfinite(ratio), ratio**self.intraday_beta, 1.0)
        return self.forecast_frame(history_df, yhat)


class HAR(DailyComponentModel):
    """HAR-RV (Corsi 2009) on daily realized variance, in logs, with a smearing correction.

    ``pooled=True`` shares the slopes across symbols (symbol-specific intercepts),
    which is much more stable with a few months of data.
    """

    def __init__(
        self,
        name: str = "har",
        lags: tuple[int, ...] = (1, 5, 22),
        pooled: bool = True,
        intraday_beta: float = 0.0,
        pooled_seasonality: bool = True,
        min_coverage: float = 0.8,
    ) -> None:
        super().__init__(name, intraday_beta, pooled_seasonality, min_coverage)
        self.lags = tuple(lags)
        self.pooled = pooled
        self.history_days = max(self.lags)

    def _fit_daily(self, rv: np.ndarray) -> None:
        feats = har_features(rv, self.lags)  # (D, L, S)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = np.log(feats[:-1])  # features known at the close of day t
            y = np.log(rv[1:])  # RV of day t+1
        s, n_lags = rv.shape[1], len(self.lags)
        self.coef_ = np.zeros((s, 1 + n_lags))
        self.smear_ = np.ones(s)
        rows = [np.isfinite(x[:, :, j]).all(axis=1) & np.isfinite(y[:, j]) for j in range(s)]
        if self.pooled:
            design, target, owner = [], [], []
            for j in range(s):
                design.append(x[rows[j], :, j])
                target.append(y[rows[j], j])
                owner.append(np.full(rows[j].sum(), j))
            xs, ys, own = np.vstack(design), np.concatenate(target), np.concatenate(owner)
            dummies = (own[:, None] == np.arange(s)[None, :]).astype(float)
            beta, *_ = np.linalg.lstsq(np.hstack([dummies, xs]), ys, rcond=None)
            self.coef_[:, 0] = beta[:s]
            self.coef_[:, 1:] = beta[s:]
            resid = ys - np.hstack([dummies, xs]) @ beta
            for j in range(s):
                self.smear_[j] = np.mean(np.exp(resid[own == j])) if (own == j).any() else 1.0
        else:
            for j in range(s):
                xj, yj = x[rows[j], :, j], y[rows[j], j]
                if len(yj) < 3 * (n_lags + 1):  # too short: fall back to the mean
                    self.coef_[j, 0] = np.mean(yj) if len(yj) else np.nan
                    continue
                design = np.hstack([np.ones((len(yj), 1)), xj])
                beta, *_ = np.linalg.lstsq(design, yj, rcond=None)
                self.coef_[j] = beta
                self.smear_[j] = np.mean(np.exp(yj - design @ beta))

    def _forecast_daily(self, rv: np.ndarray) -> np.ndarray:
        feats = har_features(rv, self.lags)[-1]  # (L, S)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = self.coef_[:, 0] + (self.coef_[:, 1:] * np.log(feats).T).sum(axis=1)
        return np.exp(z) * self.smear_


class ARMALogRV(DailyComponentModel):
    """ARMA(p, q) on log daily RV per symbol (statsmodels), with a lognormal mean correction."""

    def __init__(
        self,
        name: str = "arma_logrv",
        order: tuple[int, int] = (1, 1),
        intraday_beta: float = 0.0,
        pooled_seasonality: bool = True,
        min_coverage: float = 0.8,
    ) -> None:
        super().__init__(name, intraday_beta, pooled_seasonality, min_coverage)
        self.order = tuple(order)
        self.history_days = 60

    def _fit_daily(self, rv: np.ndarray) -> None:
        import warnings

        from statsmodels.tsa.arima.model import ARIMA

        self.results_ = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j in range(rv.shape[1]):
                y = np.log(rv[:, j])
                y = np.where(np.isfinite(y), y, np.nan)  # statsmodels handles NaN as missing
                p, q = self.order
                res = ARIMA(y, order=(p, 0, q), trend="c").fit()
                self.results_.append(res)

    def _forecast_daily(self, rv: np.ndarray) -> np.ndarray:
        import warnings

        out = np.full(rv.shape[1], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for j, res in enumerate(self.results_):
                y = np.log(rv[:, j])
                y = np.where(np.isfinite(y), y, np.nan)
                fc = res.apply(y).get_forecast(1)
                out[j] = np.exp(fc.predicted_mean[0] + 0.5 * fc.var_pred_mean[0])
        return out
