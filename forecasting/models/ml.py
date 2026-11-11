"""Gradient-boosted RV model (LightGBM), pooled across symbols.

Design:
* **Anchor + correction.** Each forecast starts from a simple anchor,
  ``mean RV of the last 5 sessions x time-of-day share of the target window``, and
  the trees learn a multiplicative correction (the anchor enters as LightGBM's
  ``init_score`` on the log scale). Features are expressed relative to the anchor's
  per-bar variance, so one model can serve symbols with very different volatility.
* **Loss = QLIKE.** The gamma objective's deviance is ``2 * (y/mu - log(y/mu) - 1)``,
  i.e. twice QLIKE, the main evaluation loss.
* **Horizon-specific.** ``fit`` only stores the training panel; the booster for a
  horizon is trained on the first ``predict`` call for it, using training data only.
* **No leakage.** Indicators use finite trailing windows (see
  :mod:`forecasting.features.indicators`); targets come from bars inside the training
  window only.
"""

from __future__ import annotations

from typing import Self

import numpy as np
import pandas as pd

from forecasting.backtest.targets import realized_variance
from forecasting.data.panel import bars_per_session, n_symbols, panel_symbols, tail_matrix
from forecasting.features.indicators import indicator_matrices
from forecasting.models.base import ForecastModel
from forecasting.models.common import seasonal_profile

FEATURES = [
    "slot",
    "log_window_share",
    "day_of_week",
    "rv_6",
    "rv_12",
    "rv_36",
    "rv_75",
    "rv_prev_session",
    "rv_prev_22_sessions",
    "parkinson_6",
    "garman_klass_12",
    "abs_ret_last",
    "max_abs_ret_12",
    "atr_14",
    "bollinger_20",
    "ret_last",
    "ret_sum_6",
    "rsi_14",
    "rel_volume",
    "rel_volume_6",
    "today_so_far",
]


class LightGBMRV(ForecastModel):
    def __init__(
        self,
        name: str = "lgbm",
        n_estimators: int = 300,
        learning_rate: float = 0.03,
        num_leaves: int = 15,
        min_child_samples: int = 100,
        subsample: float = 0.8,
        colsample: float = 0.8,
        reg_lambda: float = 1.0,
        train_stride: int = 2,
        tail_sessions: int = 24,
        pooled_seasonality: bool = True,
        min_train_rows: int = 200,
        seed: int = 0,
    ) -> None:
        super().__init__(name)
        self.min_train_rows = min_train_rows
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample = colsample
        self.reg_lambda = reg_lambda
        self.train_stride = train_stride
        self.tail_sessions = tail_sessions
        self.pooled_seasonality = pooled_seasonality
        self.seed = seed

    # ------------------------------------------------------------------ features
    def _design(
        self,
        mats: dict[str, np.ndarray],
        ret2: np.ndarray,
        positions: np.ndarray,
        horizon: int,
        weekdays: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Feature matrix and log-anchor for origins at bar ``positions`` (bars < p known).

        Returns X of shape (len(positions) * S, n_features) and log_anchor (len * S,),
        rows ordered origin-major, symbol-minor.
        """
        b, s = self.profile_.shape
        last = positions - 1
        k = positions % b
        share = self.profile_ / b  # fraction of a day's variance per slot, (B, S)
        table = np.stack([share[(kk + np.arange(horizon)) % b].sum(axis=0) for kk in range(b)])
        window = table[k]  # (n, S): share of a day covered by each target window
        # Anchor per-bar variance: last 5 sessions, else last session, else last 75 bars.
        bar_var = np.full((len(positions), s), np.nan)
        for name, n in (("rv_75", 75.0), ("rv_prev_session", b), ("rv_prev_5_sessions", b)):
            candidate = mats[name][last] / n
            bar_var = np.where(np.isfinite(candidate) & (candidate > 0), candidate, bar_var)
        anchor = bar_var * b * window
        with np.errstate(divide="ignore", invalid="ignore"):

            def rel(name: str, n_bars: float) -> np.ndarray:
                return np.log(mats[name][last] / (n_bars * bar_var))

            # Today's realized variance so far vs what the anchor expected for the
            # bars actually observed (ret2 starts at a session boundary).
            observed = ~np.isnan(ret2)
            zero = np.zeros((1, s))
            cx = np.vstack([zero, np.cumsum(np.where(observed, ret2, 0.0), axis=0)])
            slot_rows = np.arange(len(ret2)) % b
            cw = np.vstack([zero, np.cumsum(share[slot_rows] * observed, axis=0)])
            start = positions - k
            num = cx[positions] - cx[start]
            den = bar_var * b * (cw[positions] - cw[start])
            so_far = np.where((k > 0)[:, None], np.log(num / den), np.nan)
            cols = {
                "slot": np.repeat(k[:, None], s, axis=1).astype(float),
                "log_window_share": np.log(window),
                "day_of_week": np.repeat(weekdays[:, None], s, axis=1).astype(float),
                "rv_6": rel("rv_6", 6),
                "rv_12": rel("rv_12", 12),
                "rv_36": rel("rv_36", 36),
                "rv_75": rel("rv_75", 75),
                "rv_prev_session": rel("rv_prev_session", b),
                "rv_prev_22_sessions": rel("rv_prev_22_sessions", b),
                "parkinson_6": rel("parkinson_6", 6),
                "garman_klass_12": rel("garman_klass_12", 12),
                "abs_ret_last": np.log(mats["abs_ret_last"][last] ** 2 / bar_var),
                "max_abs_ret_12": np.log(mats["max_abs_ret_12"][last] ** 2 / bar_var),
                "atr_14": np.log(mats["atr_14"][last] ** 2 / bar_var),
                "bollinger_20": np.log(mats["bollinger_20"][last] ** 2 / bar_var),
                "ret_last": mats["ret_last"][last] / np.sqrt(bar_var),
                "ret_sum_6": mats["ret_sum_6"][last] / np.sqrt(6 * bar_var),
                "rsi_14": mats["rsi_14"][last],
                "rel_volume": mats["rel_volume"][last],
                "rel_volume_6": mats["rel_volume_6"][last],
                "today_so_far": so_far,
            }
            log_anchor = np.log(anchor)
        x = np.stack([cols[f] for f in FEATURES], axis=-1)  # (n, S, F)
        x = np.where(np.isfinite(x), x, np.nan)
        return x.reshape(-1, len(FEATURES)), log_anchor.reshape(-1)

    @staticmethod
    def _weekday_of_origin(sessions: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Weekday of the session being forecast. At the open (k=0) that is the next
        business day after the last known session (holidays ignored: public anyway)."""
        days = np.array([np.datetime64(d, "D") for d in sessions])
        target = np.where(k > 0, days, np.busday_offset(days, 1, roll="forward"))
        return ((target.astype("datetime64[D]").astype(int) + 3) % 7).astype(int)  # Mon=0

    # ------------------------------------------------------------------ fit / train
    def fit(self, train_df: pd.DataFrame) -> Self:
        self.profile_ = seasonal_profile(train_df, self.pooled_seasonality)
        self.symbols_ = panel_symbols(train_df)
        self.train_ = train_df
        self.boosters_: dict[int, object] = {}
        return self

    def _train(self, horizon: int):
        import lightgbm as lgb

        train = self.train_
        s, b = n_symbols(train), bars_per_session(train)
        n_bars = len(train) // s
        mats = indicator_matrices(train)
        ret2 = tail_matrix(train, "ret2", n_bars)
        slots = np.array([0]) if horizon == b else np.arange(0, b - horizon + 1, self.train_stride)
        days = np.arange(1, n_bars // b)  # day 0 has no history at its open
        positions = (days[:, None] * b + slots[None, :]).ravel()
        positions = positions[positions + horizon <= n_bars]
        y = realized_variance(ret2, positions, horizon).reshape(-1)
        sessions = train["session"].to_numpy()[::s][positions - 1]
        weekdays = self._weekday_of_origin(sessions, positions % b)
        x, log_anchor = self._design(mats, ret2, positions, horizon, weekdays)
        keep = np.isfinite(y) & (y > 0) & np.isfinite(log_anchor)
        if keep.sum() < max(self.min_train_rows, 2 * self.min_child_samples):
            return None  # too little training data: no forecast rather than a bad one
        data = lgb.Dataset(x[keep], y[keep], init_score=log_anchor[keep], free_raw_data=True)
        params = {
            "objective": "gamma",
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_data_in_leaf": self.min_child_samples,
            "bagging_fraction": self.subsample,
            "bagging_freq": 1,
            "feature_fraction": self.colsample,
            "lambda_l2": self.reg_lambda,
            "seed": self.seed,
            "deterministic": True,
            "num_threads": 2,
            "verbose": -1,
        }
        return lgb.train(params, data, num_boost_round=self.n_estimators)

    # ------------------------------------------------------------------ predict
    def predict(self, history_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        if horizon not in self.boosters_:
            self.boosters_[horizon] = self._train(horizon)
        booster = self.boosters_[horizon]
        if booster is None:
            return self.forecast_frame(history_df, np.full(n_symbols(history_df), np.nan))
        x, log_anchor = self._predict_design(history_df, horizon)
        raw = booster.predict(x, raw_score=True)
        return self.forecast_frame(history_df, np.exp(raw + log_anchor))

    def _predict_design(self, history_df: pd.DataFrame, horizon: int):
        """Features for the origin right after the last bar of ``history_df``."""
        s, b = n_symbols(history_df), bars_per_session(history_df)
        n_bars = len(history_df) // s
        k = n_bars % b
        take = min(n_bars, self.tail_sessions * b + k)
        tail = history_df.iloc[(n_bars - take) * s :]
        mats = indicator_matrices(tail)
        ret2 = tail_matrix(tail, "ret2", take)
        session = tail["session"].iat[-1]
        weekdays = self._weekday_of_origin(np.array([session], dtype=object), np.array([k]))
        return self._design(mats, ret2, np.array([take]), horizon, weekdays)
