"""Walk-forward (expanding or rolling window) backtest engine.

Timeline, in sessions::

    |---- train (fit) ----|-- test block 1 --|
    |-------- train (fit) ----------|-- test block 2 --|
    ...

* Folds are chronological; test blocks never overlap; no random splits.
* ``fit`` sees only bars strictly before the fold's first test session.
* At every forecast origin inside a test block, ``predict`` sees only bars that have
  ended by the origin (the model is not refit inside a block, but may condition on
  the newest bars, exactly as it could in live use).
* Targets are computed here from future bars and never handed to the model.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from forecasting.backtest.targets import realized_variance
from forecasting.data.panel import (
    bars_per_session,
    n_symbols,
    panel_symbols,
    to_matrix,
    validate_panel,
)
from forecasting.models.base import ForecastModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Horizon:
    """A forecasting task: RV over the next ``bars`` bars from each origin.

    ``origin="intraday"``: origins at every ``stride``-th bar boundary in the session
    such that the window ends by the close (windows never cross sessions).
    ``origin="session_open"``: one origin per session, before the first bar.
    """

    name: str
    bars: int
    origin: Literal["intraday", "session_open"] = "intraday"
    stride: int = 1

    def origin_slots(self, bars_per_session: int) -> np.ndarray:
        if not 1 <= self.bars <= bars_per_session:
            raise ValueError(f"{self.name}: bars must be in [1, {bars_per_session}]")
        if self.origin == "session_open":
            return np.array([0])
        if self.origin == "intraday":
            return np.arange(0, bars_per_session - self.bars + 1, self.stride)
        raise ValueError(f"unknown origin type {self.origin!r}")

    @property
    def overlap(self) -> int:
        """How many later origins share target bars with a given origin (for HAC lags)."""
        if self.origin == "session_open":
            return 0
        return max(0, int(np.ceil(self.bars / self.stride)) - 1)


@dataclass(frozen=True)
class WalkForward:
    initial_train_sessions: int = 60
    test_block_sessions: int = 5
    window: Literal["expanding", "rolling"] = "expanding"
    rolling_train_sessions: int | None = None
    max_folds: int | None = None
    holdout_sessions: int = 0  # lockbox: final sessions kept out of development runs


@dataclass(frozen=True)
class Fold:
    index: int
    train_start: int  # session index, inclusive
    test_start: int  # == train end (exclusive)
    test_end: int  # exclusive


def make_folds(n_sessions: int, wf: WalkForward) -> list[Fold]:
    if wf.initial_train_sessions < 1 or wf.test_block_sessions < 1:
        raise ValueError("initial_train_sessions and test_block_sessions must be >= 1")
    if wf.window == "rolling" and not wf.rolling_train_sessions:
        raise ValueError("rolling window needs rolling_train_sessions")
    folds = []
    test_start = wf.initial_train_sessions
    while test_start < n_sessions:
        if wf.max_folds is not None and len(folds) >= wf.max_folds:
            break
        test_end = min(test_start + wf.test_block_sessions, n_sessions)
        train_start = (
            max(0, test_start - wf.rolling_train_sessions) if wf.window == "rolling" else 0
        )
        folds.append(Fold(len(folds), train_start, test_start, test_end))
        test_start = test_end
    if not folds:
        raise ValueError(
            f"only {n_sessions} sessions; need more than initial_train_sessions="
            f"{wf.initial_train_sessions}"
        )
    return folds


def development_panel(panel: pd.DataFrame, wf: WalkForward) -> pd.DataFrame:
    """Development view of the data: the panel without the last ``holdout_sessions``.

    The lockbox is scored only by :func:`lockbox_walk_forward`, ideally once, at the end.
    """
    s, b = n_symbols(panel), bars_per_session(panel)
    n_sessions = len(panel) // (s * b)
    if wf.holdout_sessions <= 0:
        return panel
    keep = n_sessions - wf.holdout_sessions
    if keep <= wf.initial_train_sessions:
        raise ValueError(
            f"{n_sessions} sessions leave no development test period after holding out "
            f"{wf.holdout_sessions} and training on {wf.initial_train_sessions}"
        )
    return panel.iloc[: keep * s * b]


def lockbox_walk_forward(panel: pd.DataFrame, wf: WalkForward) -> WalkForward:
    """Walk-forward spec whose test blocks cover exactly the lockbox sessions."""
    if wf.holdout_sessions <= 0:
        raise ValueError("no lockbox configured (walk_forward.holdout_sessions is 0)")
    n_sessions = len(panel) // (n_symbols(panel) * bars_per_session(panel))
    return WalkForward(
        initial_train_sessions=n_sessions - wf.holdout_sessions,
        test_block_sessions=wf.test_block_sessions,
        window=wf.window,
        rolling_train_sessions=wf.rolling_train_sessions,
        holdout_sessions=wf.holdout_sessions,
    )


def run_walk_forward(
    panel: pd.DataFrame,
    model: ForecastModel,
    horizon: Horizon,
    wf: WalkForward,
) -> pd.DataFrame:
    """Backtest one model on one horizon.

    Returns one row per (origin, symbol) with columns
    ``model, horizon, fold, origin_ts, session, slot, symbol, yhat, y``.
    """
    validate_panel(panel)
    if "ret2" not in panel.columns:
        raise ValueError("panel needs ret2; run forecasting.features.add_returns first")
    s, b = n_symbols(panel), bars_per_session(panel)
    symbols = panel_symbols(panel)
    n_sessions = len(panel) // (s * b)
    ret2 = to_matrix(panel, "ret2")
    step_ts = panel["ts"].iloc[::s].reset_index(drop=True)
    step_session = panel["session"].to_numpy()[::s]
    slots = horizon.origin_slots(b)
    folds = make_folds(n_sessions, wf)

    positions, fold_ids, preds = [], [], []
    t0 = time.perf_counter()
    for fold in folds:
        train = panel.iloc[fold.train_start * b * s : fold.test_start * b * s]
        fitted = copy.deepcopy(model).fit(train)
        for sess in range(fold.test_start, fold.test_end):
            for slot in slots:
                p = sess * b + int(slot)
                fc = fitted.predict(panel.iloc[: p * s], horizon.bars)
                preds.append(_aligned_yhat(fc, symbols, model.name))
                positions.append(p)
                fold_ids.append(fold.index)
    log.info(
        "%s / %s: %d folds, %d origins in %.1fs",
        model.name,
        horizon.name,
        len(folds),
        len(positions),
        time.perf_counter() - t0,
    )
    pos = np.asarray(positions)
    y = realized_variance(ret2, pos, horizon.bars)
    yhat = np.vstack(preds)
    n = len(pos)
    return pd.DataFrame(
        {
            "model": model.name,
            "horizon": horizon.name,
            "fold": np.repeat(fold_ids, s),
            "origin_ts": step_ts.iloc[np.repeat(pos, s)].reset_index(drop=True),
            "session": np.repeat(step_session[pos], s),
            "slot": np.repeat(pos % b, s),
            "symbol": np.tile(np.asarray(symbols, dtype=object), n),
            "yhat": yhat.ravel(),
            "y": y.ravel(),
        }
    )


def _aligned_yhat(fc: pd.DataFrame, symbols: tuple[str, ...], name: str) -> np.ndarray:
    if "symbol" not in fc.columns or "yhat" not in fc.columns:
        raise ValueError(f"{name}.predict must return columns ['symbol', 'yhat']")
    sym = fc["symbol"].tolist()
    if tuple(sym) == symbols:  # fast path: exactly the panel's symbols, in order
        return fc["yhat"].to_numpy(dtype=float)
    if len(set(sym)) != len(sym):
        raise ValueError(f"{name}.predict returned duplicate symbols")
    extra = set(sym) - set(symbols)
    if extra:
        raise ValueError(f"{name}.predict returned unknown symbols {sorted(extra)}")
    return fc.set_index("symbol")["yhat"].reindex(list(symbols)).to_numpy(dtype=float)
