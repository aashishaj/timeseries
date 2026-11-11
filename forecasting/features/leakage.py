"""Lookahead-leakage checks for feature builders.

A feature builder is any ``f(panel) -> Series | DataFrame | ndarray`` aligned with the
panel rows. It is *causal* if the value at bar t depends only on bars <= t. We test
that two ways at random cut points c (bar boundaries):

1. **truncation**: ``f(panel[:c]) == f(panel)[:c]`` -- dropping the future changes nothing;
2. **perturbation**: scrambling every numeric value at bars >= c leaves ``f(...)[:c]``
   unchanged -- the future cannot leak in even indirectly (e.g. via full-sample
   normalisation, ``shift(-1)``, centred windows, or a rolling window over the wrong axis).

Every feature builder must have a test that calls :func:`assert_no_lookahead`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from forecasting.data.panel import n_symbols

STRUCTURAL = {"ts", "bar_end", "session", "slot", "symbol", "is_missing", "n_source"}


class LookaheadError(AssertionError):
    """A feature at bar t changed when only bars after t were changed or removed."""


def _as_2d(x) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        x = x.to_numpy(dtype=float)
    elif isinstance(x, pd.Series):
        x = x.to_numpy(dtype=float)
    x = np.asarray(x, dtype=float)
    return x.reshape(len(x), -1)


def _perturb_after(panel: pd.DataFrame, cut_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    out = panel.copy()
    n_after = len(out) - cut_rows
    for col in out.columns:
        if col in STRUCTURAL or not pd.api.types.is_float_dtype(out[col]):
            continue
        values = out[col].to_numpy(copy=True)
        values[cut_rows:] = values[cut_rows:] * rng.lognormal(0.0, 1.0, n_after) + rng.normal(
            0.0, 1.0, n_after
        )
        out[col] = values
    return out


def assert_no_lookahead(
    builder: Callable[[pd.DataFrame], object],
    panel: pd.DataFrame,
    *,
    n_cuts: int = 6,
    min_bars: int = 1,
    seed: int = 0,
    rtol: float = 1e-9,
    atol: float = 1e-12,
) -> None:
    """Raise :class:`LookaheadError` if ``builder`` uses information from the future."""
    s = n_symbols(panel)
    total_bars = len(panel) // s
    if total_bars <= min_bars + 1:
        raise ValueError("panel too short for a lookahead test")
    rng = np.random.default_rng(seed)
    cuts = np.unique(rng.integers(min_bars, total_bars, size=n_cuts))
    full = _as_2d(builder(panel))
    if len(full) != len(panel):
        raise ValueError("builder output must be aligned with the panel rows")
    for cut in cuts:
        rows = int(cut) * s
        truncated = _as_2d(builder(panel.iloc[:rows]))
        if not np.allclose(truncated, full[:rows], rtol=rtol, atol=atol, equal_nan=True):
            bad = np.argwhere(~np.isclose(truncated, full[:rows], rtol, atol, equal_nan=True))[0]
            raise LookaheadError(
                f"truncation at bar {cut}: row {bad[0]} differs "
                f"({truncated[tuple(bad)]} vs {full[:rows][tuple(bad)]})"
            )
        perturbed = _as_2d(builder(_perturb_after(panel, rows, rng)))[:rows]
        if not np.allclose(perturbed, full[:rows], rtol=rtol, atol=atol, equal_nan=True):
            bad = np.argwhere(~np.isclose(perturbed, full[:rows], rtol, atol, equal_nan=True))[0]
            raise LookaheadError(
                f"perturbing bars >= {cut} changed row {bad[0]} "
                f"({perturbed[tuple(bad)]} vs {full[:rows][tuple(bad)]})"
            )
