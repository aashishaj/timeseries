"""Volatility-targeting paper trading driven by next-session RV forecasts.

Every session, before the open, each symbol gets a long intraday position sized so
its *forecast* volatility equals an equal share of a portfolio volatility target:

    weight_i = (daily_target / n_symbols) / forecast_daily_vol_i

It is bought at the session's first price and sold at its last (MIS), paying Zerodha
intraday charges plus assumed slippage. Gross exposure is capped. Capital is fixed
(no compounding) so days are comparable.

Volatility forecasts carry no view on direction, so this is a risk-control test,
not an alpha test. Better forecasts should show up as:
* realized volatility closer to the target,
* standardized position returns (``return / forecast vol``) with standard deviation
  closer to 1 and thinner tails,
* smaller worst days and drawdowns, for the same costs.
Positions use only forecasts made before the open, from the walk-forward backtest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from forecasting.data.panel import bars_per_session, n_symbols, panel_symbols, to_matrix
from forecasting.trading.costs import CostModel


@dataclass(frozen=True)
class VolTargetConfig:
    target_annual_vol: float = 0.15
    capital_inr: float = 1_000_000.0
    max_gross_leverage: float = 2.0
    sessions_per_year: int = 248

    @property
    def target_daily_vol(self) -> float:
        return self.target_annual_vol / np.sqrt(self.sessions_per_year)


def session_open_to_close(panel: pd.DataFrame) -> pd.DataFrame:
    """Simple open-to-close return per (session, symbol); NaN if the first or last bar
    is missing (no fill would be possible)."""
    s, b = n_symbols(panel), bars_per_session(panel)
    first_open = to_matrix(panel, "open")[0::b]
    last_close = to_matrix(panel, "close")[b - 1 :: b]
    sessions = panel["session"].to_numpy()[:: s * b]
    ret = last_close / first_open - 1.0
    return pd.DataFrame(ret, index=pd.Index(sessions, name="session"), columns=panel_symbols(panel))


@dataclass
class PaperResult:
    daily: pd.DataFrame  # net P&L in INR per session (rows) and strategy (columns)
    metrics: pd.DataFrame  # one row per strategy

    def to_markdown(self, cfg: VolTargetConfig) -> str:
        cols = [
            ("ann_return_pct", "ann. return %", ".2f"),
            ("ann_vol_pct", "ann. vol %", ".2f"),
            ("vol_vs_target", "vol / target", ".2f"),
            ("sharpe", "Sharpe", ".2f"),
            ("max_drawdown_pct", "max DD %", ".2f"),
            ("worst_day_pct", "worst day %", ".2f"),
            ("z_std", "std(ret/forecast vol)", ".2f"),
            ("z_abs_p99", "p99 abs(ret/forecast vol)", ".2f"),
            ("costs_pct", "costs %/yr", ".2f"),
            ("avg_gross", "avg gross lev.", ".2f"),
        ]
        lines = [
            "## Paper trading: volatility targeting (simulated, intraday MIS)",
            "",
            f"Target {100 * cfg.target_annual_vol:.0f}% annualized (intraday), capital "
            f"Rs {cfg.capital_inr:,.0f}, gross exposure capped at {cfg.max_gross_leverage:g}x, "
            f"{len(self.daily)} sessions. Long at the open, flat at the close, Zerodha "
            "intraday charges + slippage. `constant` = equal notional, no forecast.",
            "",
            "| strategy | " + " | ".join(c[1] for c in cols) + " |",
            "|---|" + "|".join("---" for _ in cols) + "|",
        ]
        for name, row in self.metrics.iterrows():
            cells = [format(row[c], f) if np.isfinite(row[c]) else "-" for c, _, f in cols]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines += [
            "",
            "Volatility forecasts say nothing about direction: compare risk columns (vol /",
            "target near 1, std(ret/forecast vol) near 1, smaller tails and worst days),",
            "not returns. On synthetic data the true drift is zero, so returns are noise.",
        ]
        return "\n".join(lines)


def vol_target_backtest(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    cfg: VolTargetConfig | None = None,
    costs: CostModel | None = None,
) -> PaperResult:
    """Simulate volatility targeting for every model in ``forecasts`` (next-session
    horizon: one forecast per session and symbol) plus a constant-notional benchmark."""
    cfg = cfg or VolTargetConfig()
    costs = costs or CostModel()
    rets = session_open_to_close(panel)
    sessions = sorted(forecasts["session"].unique())
    rets = rets.loc[rets.index.isin(sessions)].sort_index()
    symbols = list(rets.columns)
    s = len(symbols)

    weights: dict[str, pd.DataFrame] = {
        "constant": pd.DataFrame(1.0 / s, index=rets.index, columns=symbols)
    }
    for model, f in forecasts.groupby("model", sort=False):
        fc = f.pivot_table(index="session", columns="symbol", values="yhat").reindex(
            index=rets.index, columns=symbols
        )
        daily_vol = np.sqrt(fc.where(fc > 0)) / 100.0
        weights[model] = (cfg.target_daily_vol / s) / daily_vol

    daily, rows = {}, {}
    for name, w in weights.items():
        w = w.where(rets.notna())  # no fill possible -> no position
        gross = w.abs().sum(axis=1)
        scale = np.minimum(1.0, cfg.max_gross_leverage / gross.replace(0, np.nan)).fillna(0)
        w = w.mul(scale, axis=0).fillna(0.0)
        notional = w * cfg.capital_inr
        gross_pnl = (notional * rets.fillna(0.0)).sum(axis=1)
        cost = pd.DataFrame(
            costs.round_trip(notional.to_numpy()), index=rets.index, columns=symbols
        ).sum(axis=1)
        pnl = gross_pnl - cost
        daily[name] = pnl
        rows[name] = _metrics(
            pnl, cost, w, rets, None if name == "constant" else weights[name], cfg
        )
    return PaperResult(daily=pd.DataFrame(daily), metrics=pd.DataFrame(rows).T)


def _metrics(pnl, cost, w, rets, raw_weights, cfg: VolTargetConfig) -> dict:
    r = pnl / cfg.capital_inr
    n = cfg.sessions_per_year
    equity = r.cumsum()
    drawdown = equity - equity.cummax()
    ann_vol = r.std() * np.sqrt(n)
    # Standardized position returns: return / the vol the sizing assumed (forecast
    # strategies only; the constant benchmark assumes no volatility at all).
    z = np.array([])
    if raw_weights is not None:
        implied_vol = (cfg.target_daily_vol / w.shape[1]) / raw_weights
        z = (rets / implied_vol).to_numpy().ravel()
        z = z[np.isfinite(z)]
    return {
        "sessions": len(r),
        "ann_return_pct": 100 * r.mean() * n,
        "ann_vol_pct": 100 * ann_vol,
        "vol_vs_target": ann_vol / cfg.target_annual_vol,
        "sharpe": r.mean() / r.std() * np.sqrt(n) if r.std() > 0 else np.nan,
        "max_drawdown_pct": 100 * drawdown.min(),
        "worst_day_pct": 100 * r.min(),
        "z_std": float(np.std(z)) if len(z) else np.nan,
        "z_abs_p99": float(np.quantile(np.abs(z), 0.99)) if len(z) else np.nan,
        "costs_pct": 100 * cost.sum() / cfg.capital_inr * n / max(len(r), 1),
        "avg_gross": float(w.abs().sum(axis=1).mean()),
    }
