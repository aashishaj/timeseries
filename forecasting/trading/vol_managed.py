"""A held, volatility-managed basket with banded rebalancing (delivery / CNC).

The intraday version (``vol_target.py``) buys and sells the whole book every session,
so charges of about 0.12% of the book per day swamp everything else. This version
holds the stocks overnight and trades a stock only when its weight has drifted too
far from where the forecasts say it should be.

Every session, before the open:

1. Each stock's next-session RV forecast (intraday only) is turned into a
   close-to-close volatility with its *overnight ratio*: past close-to-close variance
   divided by past intraday RV, from all earlier sessions.
2. Base weights are inverse-volatility (every stock carries the same risk). The
   basket's volatility is ``sqrt(b' D C D b)``: ``D`` holds the forecast volatilities
   and ``C`` is the correlation of the last ``corr_window`` sessions of close-to-close
   returns, shrunk toward its average.
3. The basket is scaled to the volatility target. Gross exposure is capped at
   ``max_gross`` (1.0 by default: delivery trades cannot borrow), the rest is cash.
4. A stock is traded, at the session's first price, only when its current weight is
   more than ``band`` x its target weight away from the target; it is then traded all
   the way to the target. Otherwise the position is left alone.

Volatility forecasts carry no view on direction. A long-only basket earns whatever
the market gives; better forecasts can only make the risk steadier (realized
volatility near the target, ``return / forecast volatility`` with standard deviation
near 1, smaller worst days). Returns over a few months are market direction and noise.

Positions for a session use only prices and forecasts from before its open.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from forecasting.data.panel import bars_per_session, n_symbols, panel_symbols, to_matrix
from forecasting.trading.costs import DELIVERY_COSTS, CostModel

log = logging.getLogger(__name__)

BUY_HOLD = "buy_hold"


@dataclass(frozen=True)
class VolManagedConfig:
    target_annual_vol: float = 0.10
    capital_inr: float = 1_000_000.0
    max_gross: float = 1.0  # delivery: no borrowing
    band: float = 0.25  # trade a stock when |weight - target| > band * target
    corr_window: int = 60  # sessions of close-to-close returns behind the correlations
    corr_shrink: float = 0.5  # 0 = sample correlation, 1 = one average correlation
    min_history: int = 20  # sessions needed before a ratio or correlation is trusted
    whole_shares: bool = True
    sessions_per_year: int = 248

    def __post_init__(self) -> None:
        if not 0.0 <= self.band < 1.0:
            raise ValueError(f"band must be in [0, 1), got {self.band}")

    @property
    def target_daily_vol(self) -> float:
        return self.target_annual_vol / np.sqrt(self.sessions_per_year)


@dataclass(frozen=True)
class SessionData:
    """One row per session: prices to trade and value at, and the intraday RV."""

    sessions: np.ndarray  # trading dates
    symbols: tuple[str, ...]
    open: np.ndarray  # (n_sessions, n_symbols) first traded price; NaN if none
    close: np.ndarray  # last traded price, carried forward over sessions without one
    rv: np.ndarray  # intraday realized variance in %^2; NaN if a bar is missing


def session_data(panel: pd.DataFrame) -> SessionData:
    s, b = n_symbols(panel), bars_per_session(panel)
    n = len(panel) // (s * b)
    open_ = to_matrix(panel, "open").astype(float).reshape(n, b, s)
    close = to_matrix(panel, "close").astype(float).reshape(n, b, s)
    last_close = pd.DataFrame(_first_valid(close[:, ::-1, :])).ffill().to_numpy()
    return SessionData(
        sessions=panel["session"].to_numpy()[:: s * b],
        symbols=panel_symbols(panel),
        open=_first_valid(open_),
        close=last_close,
        rv=to_matrix(panel, "ret2").astype(float).reshape(n, b, s).sum(axis=1),
    )


def _first_valid(a: np.ndarray) -> np.ndarray:
    """First finite value along axis 1 of (n, b, s); NaN where there is none."""
    idx = np.argmax(np.isfinite(a), axis=1)
    return np.take_along_axis(a, idx[:, None, :], axis=1)[:, 0, :]


def close_to_close(data: SessionData) -> np.ndarray:
    """Percent log return from the previous session's close; row 0 is NaN."""
    cc = np.full_like(data.close, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        cc[1:] = 100.0 * np.log(data.close[1:] / data.close[:-1])
    return cc


def overnight_ratio(data: SessionData, min_history: int = 20) -> np.ndarray:
    """Close-to-close variance / intraday RV per symbol, from sessions strictly before
    each row. 1.0 (intraday volatility only) until ``min_history`` sessions exist."""
    cc, rv = close_to_close(data), data.rv
    ok = np.isfinite(cc) & np.isfinite(rv)
    num = np.cumsum(np.where(ok, cc * cc, 0.0), axis=0)
    den = np.cumsum(np.where(ok, rv, 0.0), axis=0)
    cnt = np.cumsum(ok, axis=0)
    ratio = np.ones_like(rv)
    with np.errstate(divide="ignore", invalid="ignore"):
        past = num[:-1] / den[:-1]
    usable = (cnt[:-1] >= min_history) & (den[:-1] > 0)
    ratio[1:] = np.where(usable, past, 1.0)
    return ratio


def trailing_correlation(
    cc: np.ndarray, t: int, window: int = 60, shrink: float = 0.5, min_history: int = 20
) -> np.ndarray:
    """Correlation of close-to-close returns over the ``window`` sessions before ``t``,
    shrunk toward its average off-diagonal value. With too little history every pair
    gets correlation 1, the cautious choice: it overstates the basket's volatility."""
    s = cc.shape[1]
    rows = cc[max(0, t - window) : t]
    rows = rows[np.isfinite(rows).all(axis=1)]
    if s == 1:
        return np.ones((1, 1))
    if len(rows) < min_history or (rows.std(axis=0) == 0).any():
        return np.ones((s, s))
    c = np.corrcoef(rows.T)
    mean_off = (c.sum() - s) / (s * s - s)
    flat = np.full((s, s), mean_off)
    np.fill_diagonal(flat, 1.0)
    return (1.0 - shrink) * c + shrink * flat


def _target_weights(vol: np.ndarray, corr: np.ndarray, cfg: VolManagedConfig) -> np.ndarray:
    """Inverse-volatility weights scaled to the volatility target; NaN = no view."""
    w = np.full_like(vol, np.nan)
    ok = np.isfinite(vol) & (vol > 0)
    if not ok.any():
        return w
    base = 1.0 / vol[ok]
    base /= base.sum()
    risk = base * vol[ok]
    basket_vol = float(np.sqrt(risk @ corr[np.ix_(ok, ok)] @ risk))
    w[ok] = min(cfg.max_gross, cfg.target_daily_vol / basket_vol) * base
    return w


def _execute(shares, desired, price, cash, costs: CostModel, whole: bool, credit: float = 0.0):
    """Trade ``shares`` toward ``desired`` at ``price``. Buys are cut back when they need
    more than the cash on hand plus ``credit`` (borrowing room; 0 for delivery)."""
    px = np.where(np.isfinite(price), price, 0.0)
    delta = np.where(np.isfinite(price), desired - shares, 0.0)
    buy_q, sell_q = np.clip(delta, 0.0, None), np.clip(-delta, 0.0, None)
    sell_n = sell_q * px
    c_sell = costs.side(sell_n, buy=False)
    available = cash + credit + float((sell_n - c_sell).sum())
    buy_n = buy_q * px
    c_buy = costs.side(buy_n, buy=True)
    needed = float((buy_n + c_buy).sum())
    if needed > available and needed > 0:
        buy_q = buy_q * max(available, 0.0) / needed
        if whole:
            buy_q = np.floor(buy_q)
        buy_n = buy_q * px
        c_buy = costs.side(buy_n, buy=True)
    cost = float(c_buy.sum() + c_sell.sum())
    cash = cash + float(sell_n.sum() - buy_n.sum()) - cost
    traded = float(buy_n.sum() + sell_n.sum())
    orders = int((buy_q > 0).sum() + (sell_q > 0).sum())
    return shares + buy_q - sell_q, cash, cost, traded, orders


def _simulate(
    data: SessionData,
    span: range,
    targets: np.ndarray,
    vols: np.ndarray | None,
    corrs: dict[int, np.ndarray],
    cfg: VolManagedConfig,
    costs: CostModel,
) -> pd.DataFrame:
    """Run one strategy over the sessions in ``span``. ``targets[j]`` are the weights
    wanted at the open of ``span[j]`` (NaN = leave that stock alone)."""
    s = len(data.symbols)
    shares, cash = np.zeros(s), float(cfg.capital_inr)
    prev_equity = cash
    rows = []
    for j, t in enumerate(span):
        prev_close = data.close[t - 1] if t > 0 else np.full(s, np.nan)
        value_px = np.where(np.isfinite(data.open[t]), data.open[t], prev_close)
        value_px = np.where(np.isfinite(value_px), value_px, 0.0)
        equity_open = cash + float(shares @ value_px)
        current = shares * value_px / equity_open
        want = targets[j]
        trade = np.isfinite(want) & np.isfinite(data.open[t])
        with np.errstate(invalid="ignore"):
            trade &= np.abs(current - want) > cfg.band * want
        desired = shares.copy()
        if trade.any():
            qty = want[trade] * equity_open / data.open[t][trade]
            desired[trade] = np.floor(qty + 1e-9) if cfg.whole_shares else qty
        credit = max(cfg.max_gross - 1.0, 0.0) * equity_open
        shares, cash, cost, traded, orders = _execute(
            shares, desired, data.open[t], cash, costs, cfg.whole_shares, credit
        )
        held = shares * value_px / (cash + float(shares @ value_px))
        forecast_vol = np.nan
        if vols is not None and np.isfinite(vols[j][held > 0]).all() and (held > 0).any():
            risk = np.where(held > 0, held * vols[j], 0.0)
            forecast_vol = float(np.sqrt(risk @ corrs[t] @ risk))
        close_px = np.where(np.isfinite(data.close[t]), data.close[t], value_px)
        equity_close = cash + float(shares @ close_px)
        rows.append(
            {
                "session": data.sessions[t],
                "ret": equity_close / prev_equity - 1.0,
                "equity": equity_close,
                "equity_open": equity_open,
                "gross": float(shares @ close_px) / equity_close,
                "cost_frac": cost / equity_open,
                "traded_frac": traded / equity_open,
                "orders": orders,
                "forecast_vol": forecast_vol,
            }
        )
        prev_equity = equity_close
    d = pd.DataFrame(rows).set_index("session")
    # Standardized returns: the day's close-to-close return over the volatility the
    # forecasts implied for the book held from that day's open. The first day is left
    # out: it has no overnight leg and carries the cost of the first purchase.
    d["z"] = d["ret"] / d["forecast_vol"]
    d.iloc[0, d.columns.get_loc("z")] = np.nan
    return d


@dataclass
class ManagedResult:
    detail: dict[str, pd.DataFrame]  # strategy -> one row per session
    cfg: VolManagedConfig

    @property
    def daily(self) -> pd.DataFrame:
        """Net daily return (fraction of equity) per session and strategy."""
        return pd.DataFrame({name: d["ret"] for name, d in self.detail.items()})

    def metrics(self, sessions: Sequence | None = None) -> pd.DataFrame:
        """One row per strategy, over all sessions or only those listed."""
        rows = {}
        for name, d in self.detail.items():
            part = d if sessions is None else d.loc[d.index.isin(list(sessions))]
            first_ever = d.index[0]
            rows[name] = _metrics(part, part.index != first_ever, self.cfg)
        return pd.DataFrame(rows).T

    def to_markdown(
        self, periods: dict[str, Sequence] | None = None, title: str | None = None
    ) -> str:
        cfg = self.cfg
        cols = [
            ("ann_return_pct", "ann. return %", ".2f"),
            ("ann_vol_pct", "ann. vol %", ".2f"),
            ("vol_vs_target", "vol / target", ".2f"),
            ("sharpe", "Sharpe", ".2f"),
            ("max_drawdown_pct", "max DD %", ".2f"),
            ("worst_day_pct", "worst day %", ".2f"),
            ("z_std", "std(ret/forecast vol)", ".2f"),
            ("turnover_x", "turnover x/yr", ".1f"),
            ("costs_pct", "costs %/yr", ".2f"),
            ("orders_per_month", "orders/month", ".1f"),
            ("avg_gross", "avg invested", ".2f"),
        ]
        lines = [
            title or "## Paper trading: held basket, volatility-managed (simulated, delivery)",
            "",
            f"Target {100 * cfg.target_annual_vol:.0f}% annualized (close to close), capital "
            f"Rs {cfg.capital_inr:,.0f}, at most {cfg.max_gross:g}x invested, the rest in cash "
            f"at 0%. A stock is traded at the open only when its weight is more than "
            f"{100 * cfg.band:.0f}% away from its target. Zerodha delivery charges + slippage. "
            f"`{BUY_HOLD}` = equal amounts bought on day one and never traded.",
        ]
        for label, sessions in (periods or {"": None}).items():
            m = self.metrics(sessions)
            n = int(m["sessions"].max()) if len(m) else 0
            lines += ["", f"### {label} ({n} sessions)" if label else f"{n} sessions.", ""]
            lines += [
                "| strategy | " + " | ".join(c[1] for c in cols) + " |",
                "|---|" + "|".join("---" for _ in cols) + "|",
            ]
            for name, row in m.iterrows():
                cells = [format(row[c], f) if np.isfinite(row[c]) else "-" for c, _, f in cols]
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
        late = {
            name: int((d["gross"].to_numpy() > 0).argmax())
            for name, d in self.detail.items()
            if (d["gross"] > 0).any() and d["gross"].iloc[0] == 0
        }
        if late:
            lines += [
                "",
                "Started late (no forecasts yet, so in cash; returns not comparable): "
                + ", ".join(f"`{name}` after {n} sessions" for name, n in late.items())
                + ".",
            ]
        lines += [
            "",
            "Volatility forecasts say nothing about direction, and a long-only basket earns",
            "what the market gives: read the risk and cost columns, not the returns.",
            "`turnover x/yr` is value traded / equity after the first purchase; `costs %/yr`",
            "includes the first purchase but no final sale. Dividends are not counted.",
        ]
        return "\n".join(lines)


def _metrics(d: pd.DataFrame, rebalancing: np.ndarray, cfg: VolManagedConfig) -> dict:
    r = d["ret"]
    n, per_year = len(r), cfg.sessions_per_year
    equity = np.concatenate([[1.0], (1.0 + r).cumprod().to_numpy()])
    drawdown = equity / np.maximum.accumulate(equity) - 1.0
    ann_vol = r.std() * np.sqrt(per_year)
    z = d["z"].to_numpy()
    z = z[np.isfinite(z)]
    return {
        "sessions": n,
        "ann_return_pct": 100 * r.mean() * per_year,
        "ann_vol_pct": 100 * ann_vol,
        "vol_vs_target": ann_vol / cfg.target_annual_vol,
        "sharpe": r.mean() / r.std() * np.sqrt(per_year) if r.std() > 0 else np.nan,
        "max_drawdown_pct": 100 * drawdown.min(),
        "worst_day_pct": 100 * r.min(),
        "z_std": float(np.std(z)) if len(z) >= 20 else np.nan,
        "turnover_x": d["traded_frac"].to_numpy()[rebalancing].sum() * per_year / max(n, 1),
        "costs_pct": 100 * d["cost_frac"].sum() * per_year / max(n, 1),
        "orders_per_month": d["orders"].to_numpy()[rebalancing].sum() * 21 / max(n, 1),
        "avg_gross": float(d["gross"].mean()),
    }


def vol_managed_backtest(
    panel: pd.DataFrame,
    forecasts: pd.DataFrame,
    cfg: VolManagedConfig | None = None,
    costs: CostModel | None = None,
    bands: Sequence[float] | None = None,
) -> ManagedResult:
    """Simulate the held basket for every model in ``forecasts`` (next-session horizon:
    one forecast per session and symbol) plus a buy-and-hold benchmark.

    With ``bands``, every model is run once per band and named ``"<model> @ <band>%"``.
    """
    cfg = cfg or VolManagedConfig()
    costs = costs or CostModel.from_yaml(DELIVERY_COSTS)
    data = session_data(panel)
    position = {day: i for i, day in enumerate(data.sessions)}
    wanted = sorted(set(forecasts["session"]) & set(position))
    if not wanted:
        raise ValueError("no forecast session is in the panel")
    span = range(position[wanted[0]], position[wanted[-1]] + 1)
    days = data.sessions[span.start : span.stop]
    s = len(data.symbols)
    ratio = overnight_ratio(data, cfg.min_history)[span.start : span.stop]
    cc = close_to_close(data)
    corrs = {
        t: trailing_correlation(cc, t, cfg.corr_window, cfg.corr_shrink, cfg.min_history)
        for t in span
    }

    hold = np.full((len(span), s), np.nan)
    hold_cfg = replace(cfg, band=0.0, max_gross=min(cfg.max_gross, 1.0))  # never borrows
    hold[0] = hold_cfg.max_gross / s
    detail = {BUY_HOLD: _simulate(data, span, hold, None, corrs, hold_cfg, costs)}
    for model, f in forecasts.groupby("model", sort=False):
        fc = f.pivot_table(index="session", columns="symbol", values="yhat", dropna=False)
        fc = fc.reindex(index=days, columns=list(data.symbols)).to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            vol = np.sqrt(np.where(fc > 0, fc, np.nan) * ratio) / 100.0
        if not np.isfinite(vol).any():
            log.warning("%s: no usable forecasts; left out of paper trading", model)
            continue
        targets = np.vstack([_target_weights(vol[j], corrs[t], cfg) for j, t in enumerate(span)])
        for band in bands if bands is not None else [cfg.band]:
            name = model if bands is None else f"{model} @ {100 * band:.0f}%"
            run_cfg = replace(cfg, band=band)
            detail[name] = _simulate(data, span, targets, vol, corrs, run_cfg, costs)
    return ManagedResult(detail=detail, cfg=cfg)
