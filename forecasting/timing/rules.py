"""Invest / stay-out signals on daily prices.

Every signal is decided at the close of day t from prices up to and including day t,
and acted on at the open of day t+1 (the simulation does the shift). ``True`` means
"be invested". Before a signal has enough history it is NaN (no decision).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TREND_WINDOW = 200
VOL_LOOKBACK = 252
VOL_PERCENTILE = 0.80
HAR_MIN_TRAIN = 250
HAR_REFIT_EVERY = 21


def daily_variance(px: pd.DataFrame) -> pd.Series:
    """One day's price variance from open/high/low/close: the overnight gap squared plus
    the Garman-Klass estimate of the day's own variance."""
    o, h, low, c = (np.log(px[k]) for k in ("open", "high", "low", "close"))
    overnight = (o - c.shift()) ** 2
    gk = 0.5 * (h - low) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    v = overnight.fillna(0.0) + gk.clip(lower=0.0)
    return v.clip(lower=1e-8)  # flat days would break the log


def har_forecast(
    var: pd.Series, min_train: int = HAR_MIN_TRAIN, refit_every: int = HAR_REFIT_EVERY
) -> pd.Series:
    """Forecast of tomorrow's variance made at each close (HAR on logs: today, the past
    week and the past month). Refit on all earlier days every ``refit_every`` days."""
    lv = np.log(var)
    x = pd.DataFrame(
        {
            "one": 1.0,
            "day": lv,
            "week": np.log(var.rolling(5).mean()),
            "month": np.log(var.rolling(22).mean()),
        }
    )
    y = lv.shift(-1)  # tomorrow: known only after tomorrow's close
    out = pd.Series(np.nan, index=var.index)
    xs, ys = x.to_numpy(), y.to_numpy()
    usable = np.isfinite(xs).all(axis=1)
    n = len(var)
    for k in range(min_train + 22, n, refit_every):
        train = usable[:k] & np.isfinite(ys[:k])
        train[k - 1] = False  # conservative: only targets known before day k
        beta, *_ = np.linalg.lstsq(xs[:k][train], ys[:k][train], rcond=None)
        stop = min(k + refit_every, n)
        block = xs[k:stop]
        out.iloc[k:stop] = np.where(usable[k:stop], np.exp(block @ beta), np.nan)
    return out


def trend_signal(close: pd.Series, window: int = TREND_WINDOW) -> pd.Series:
    """Invested while the close is above its ``window``-day average."""
    avg = close.rolling(window).mean()
    return (close > avg).where(avg.notna())


def vol_signal(
    forecast: pd.Series, lookback: int = VOL_LOOKBACK, percentile: float = VOL_PERCENTILE
) -> pd.Series:
    """Invested unless tomorrow's forecast volatility is above its own past-year
    ``percentile`` (the past year includes today's forecast)."""
    cut = forecast.rolling(lookback).quantile(percentile)
    return (forecast <= cut).where(cut.notna() & forecast.notna())


def signals(px: pd.DataFrame) -> pd.DataFrame:
    """B (always), T (trend), V (volatility) and TV (both) for one instrument."""
    t = trend_signal(px["close"])
    v = vol_signal(har_forecast(daily_variance(px)))
    tv = (t.astype("boolean") & v.astype("boolean")).astype(float).where(t.notna() & v.notna())
    return pd.DataFrame(
        {"B": 1.0, "T": t.astype(float), "V": v.astype(float), "TV": tv}, index=px.index
    )


def explain(px: pd.DataFrame, sig: pd.DataFrame) -> dict:
    """Plain reasons behind the latest signals (for the app)."""
    close = px["close"]
    avg = close.rolling(TREND_WINDOW).mean()
    fc = har_forecast(daily_variance(px))
    cut = fc.rolling(VOL_LOOKBACK).quantile(VOL_PERCENTILE)
    last = px.index[-1]
    return {
        "date": str(last.date()),
        "close": float(close.iloc[-1]),
        "avg200": float(avg.iloc[-1]),
        "above_avg_pct": float(100 * (close.iloc[-1] / avg.iloc[-1] - 1)),
        "vol_forecast_annual_pct": float(100 * np.sqrt(fc.iloc[-1] * 248)),
        "vol_cutoff_annual_pct": float(100 * np.sqrt(cut.iloc[-1] * 248)),
        "signals": {k: bool(sig[k].iloc[-1]) for k in sig.columns if pd.notna(sig[k].iloc[-1])},
    }
