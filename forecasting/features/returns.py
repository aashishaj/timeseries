"""Bar returns. Everything here is backward-looking: the value at bar t uses bars <= t only."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.data.panel import n_symbols, to_matrix


def log_returns(panel: pd.DataFrame) -> np.ndarray:
    """Intraday log returns in percent, aligned with the panel rows.

    * bar t (t > 0 in the session): ``100 * log(close_t / close_{t-1})``
    * first bar of a session: ``100 * log(close / open)`` -- the overnight gap is
      excluded, so realized variance measures intraday volatility only
    * NaN when the bar, or the bar before it, is missing (gaps are never bridged)
    """
    close = to_matrix(panel, "close").astype(float)
    open_ = to_matrix(panel, "open").astype(float)
    prev = np.empty_like(close)
    prev[0] = np.nan
    prev[1:] = close[:-1]
    first = panel["slot"].to_numpy()[:: n_symbols(panel)] == 0
    prev[first] = open_[first]
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 100.0 * np.log(close / prev)
    return r.ravel()


def add_returns(panel: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``panel`` with ``ret`` (percent log return) and ``ret2`` (its square)."""
    r = log_returns(panel)
    return panel.assign(ret=r, ret2=r * r)
