"""The invest / stay-out study (PROGRESS.md, round 9): every rule on every instrument,
after delivery charges and income tax, on the development years or the lockbox."""

from __future__ import annotations

import csv
import datetime as dt
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.timing.account import AccountResult, run_account
from forecasting.timing.rules import signals
from forecasting.timing.tax import TaxRules

log = logging.getLogger(__name__)

CAPITAL = 1_000_000.0
MARKET = "NIFTYBEES"
LOCKBOX_YEARS = 3
RULES = {  # name -> (signal column, take_profit, stop_loss)
    "B": ("B", None, None),
    "T": ("T", None, None),
    "V": ("V", None, None),
    "TV": ("TV", None, None),
    "T+take-profit": ("T", 0.02, None),
    "T+stop-loss": ("T", None, 0.02),
}
LABELS = {
    "B": "Buy and hold",
    "T": "Trend (above 200-day average)",
    "V": "Volatility (skip the most volatile 20%)",
    "TV": "Trend and volatility",
    "T+take-profit": "Trend + sell at 2% net profit",
    "T+stop-loss": "Trend + 2% stop-loss",
}


def prepare(daily: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Drop weekend special sessions and bad rows."""
    out = {}
    for sym, df in daily.items():
        df = df[df.index.day_of_week < 5]
        df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
        out[sym] = df.sort_index()
    return out


def periods(data: dict[str, pd.DataFrame], sigs: dict[str, pd.DataFrame]):
    first_ok = max(s[["T", "V", "TV"]].dropna().index[0] for s in sigs.values())
    start = min(df.index[df.index > first_ok][0] for df in data.values())
    last = min(df.index[-1] for df in data.values())
    lockbox_start = last - pd.DateOffset(years=LOCKBOX_YEARS) + pd.Timedelta(days=1)
    return {
        "dev": (start, lockbox_start - pd.Timedelta(days=1)),
        "final": (lockbox_start, last),
    }


def metrics(res: AccountResult, rules: TaxRules) -> dict:
    v = res.value
    start, end = v.index[0], v.index[-1]
    years = max((end - start).days / 365.25, 1e-9)
    with_start = pd.concat([pd.Series([CAPITAL], index=[start - pd.Timedelta(days=1)]), v])
    monthly = with_start.resample("ME").last().pct_change().dropna()
    yearly = with_start.resample("YE").last().pct_change().dropna()
    drawdown = (with_start / with_start.cummax() - 1).min()
    per_trade = []
    for t in res.trades:
        rate = rules.ltcg_rate if rules.is_long_term(t.bought, t.sold) else rules.stcg_rate
        after_tax = t.gain - max(0.0, t.gain) * rate * (1 + rules.cess_rate)
        per_trade.append(after_tax / t.cost)
    per_trade = np.array(per_trade)
    return {
        "start": str(start.date()),
        "end": str(end.date()),
        "final_value": float(v.iloc[-1]),
        "cagr": float((v.iloc[-1] / CAPITAL) ** (1 / years) - 1),
        "max_drawdown": float(drawdown),
        "worst_year": float(yearly.min()) if len(yearly) else np.nan,
        "months_2pct": float((monthly >= 0.02).mean()) if len(monthly) else np.nan,
        "avg_month": float(monthly.mean()) if len(monthly) else np.nan,
        "trades_per_year": len(res.trades) / years,
        "time_invested": float(res.invested.mean()),
        "avg_trade_after_tax": float(per_trade.mean()) if len(per_trade) else np.nan,
        "trades_2pct": float((per_trade >= 0.02).mean()) if len(per_trade) else np.nan,
        "tax_paid": float(sum(t for _, t in res.taxes)),
        "charges_per_year": res.charges / years / CAPITAL,
    }


def judge(table: pd.DataFrame) -> pd.DataFrame:
    """Add the pass rule: (a) >= 2 points a year above buy and hold, or (b) no more than
    1 point below it with a largest fall at least a third smaller."""
    bh = table[table["rule"] == "B"].set_index("instrument")
    t = table.copy()
    t["gap"] = t["cagr"] - t["instrument"].map(bh["cagr"])
    t["dd_ratio"] = t["max_drawdown"] / t["instrument"].map(bh["max_drawdown"])
    t["pass_a"] = t["gap"] >= 0.02
    t["pass_b"] = (t["gap"] >= -0.01) & (t["dd_ratio"] <= 2 / 3)
    t["passes"] = (t["pass_a"] | t["pass_b"]) & (t["rule"] != "B")
    return t


def summarize(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for rule, g in t[t["rule"] != "B"].groupby("rule", sort=False):
        stocks = g[g["instrument"] != MARKET]
        market = g[g["instrument"] == MARKET]
        rows.append(
            {
                "rule": rule,
                "stocks_passing": int(stocks["passes"].sum()),
                "n_stocks": len(stocks),
                "market_passes": bool(market["passes"].any()),
                "instruments_passing": int(g["passes"].sum()),
                "median_gap": float(g["gap"].median()),
                "median_dd_ratio": float(g["dd_ratio"].median()),
                "trades_per_year": float(g["trades_per_year"].mean()),
                "time_invested": float(g["time_invested"].mean()),
                "months_2pct": float(g["months_2pct"].mean()),
            }
        )
    s = pd.DataFrame(rows)
    return s.sort_values(
        ["instruments_passing", "trades_per_year"], ascending=[False, True]
    ).reset_index(drop=True)


def run_split(
    data: dict[str, pd.DataFrame], split: str, rules_to_run: list[str] | None = None
) -> tuple[pd.DataFrame, dict]:
    data = prepare(data)
    sigs = {sym: signals(df) for sym, df in data.items()}
    span = periods(data, sigs)[split]
    tax_rules = TaxRules.from_yaml()
    rows = []
    for sym, df in data.items():
        for rule in ["B", *(rules_to_run or [r for r in RULES if r != "B"])]:
            col, tp, sl = RULES[rule]
            res = run_account(
                df, sigs[sym][col], *span, rules=tax_rules, take_profit=tp, stop_loss=sl
            )
            rows.append({"instrument": sym, "rule": rule, **metrics(res, tax_rules)})
        log.info("%s: simulated (%s years)", sym, split)
    return judge(pd.DataFrame(rows)), {"start": str(span[0].date()), "end": str(span[1].date())}


def run_study(daily_root: Path, out: Path, split: str) -> Path:
    from forecasting.data.kite_backfill import load_daily

    out.mkdir(parents=True, exist_ok=True)
    data = load_daily(daily_root)
    finalist_file = out / "finalist.json"
    if split == "final":
        if not finalist_file.exists():
            raise SystemExit("run the development split first: it picks the finalist")
        finalist = json.loads(finalist_file.read_text())["rule"]
        _log_opening(out.parent / "lockbox_log.csv", finalist)
        table, span = run_split(data, "final", [finalist])
    else:
        table, span = run_split(data, "dev")
    summary = summarize(table)
    table.to_csv(out / f"metrics_{split}.csv", index=False)
    if split == "dev":
        best = summary.iloc[0]["rule"]
        finalist_file.write_text(
            json.dumps({"rule": best, "picked": str(dt.date.today()), "dev": span}, indent=2)
        )
    path = out / ("lockbox.md" if split == "final" else "development.md")
    path.write_text(report(table, summary, split, span))
    return path


def _log_opening(path: Path, finalist: str) -> None:
    previous = 0
    if path.exists():
        with open(path) as fh:
            previous = sum(1 for r in csv.DictReader(fh) if r["experiment"] == "timing")
    if previous:
        log.warning("the timing lockbox was opened %d time(s) before", previous)
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["data", "opened_at", "experiment", "models"])
        w.writerow(
            ["real:kite_daily", dt.datetime.now().strftime("%Y-%m-%d %H:%M"), "timing", finalist]
        )


def _pct(x: float, plus: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return "-"
    return f"{100 * x:+.1f}%" if plus else f"{100 * x:.1f}%"


def report(table: pd.DataFrame, summary: pd.DataFrame, split: str, span: dict) -> str:
    name = "Lockbox" if split == "final" else "Development years"
    lines = [
        f"# When to invest: {name.lower()} ({span['start']} to {span['end']})",
        "",
        "Each instrument is a separate Rs 10 lakh account. Results are after Zerodha",
        "delivery charges, slippage and income tax (20% short-term, 12.5% long-term",
        "above Rs 1.25 lakh, 4% cess), valued as if everything were sold at the end.",
        "Cash earns nothing; dividends are not included (for any rule).",
        "",
        "## Rules compared with buy and hold",
        "",
        "| rule | stocks passing | NIFTYBEES passes | yearly return vs buy and hold "
        "(median) | largest fall vs buy and hold (median) | trades a year | time invested "
        "| months with 2%+ after tax |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in summary.itertuples():
        lines.append(
            f"| {LABELS[r.rule]} | {r.stocks_passing} of {r.n_stocks} | "
            f"{'yes' if r.market_passes else 'no'} | {100 * r.median_gap:+.1f} points | "
            f"{100 * r.median_dd_ratio:.0f}% as deep | {r.trades_per_year:.1f} | "
            f"{_pct(r.time_invested)} | {_pct(r.months_2pct)} |"
        )
    lines += [
        "",
        "A rule passes for an instrument if its yearly return is at least 2 points above",
        "buy and hold, or no more than 1 point below it with a largest fall at least a",
        "third smaller.",
        "",
        "## Every instrument",
        "",
        "| instrument | rule | yearly return | vs buy and hold | largest fall | worst year "
        "| months 2%+ | trades/yr | avg trade after tax | passes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.sort_values(["instrument", "rule"], key=_rule_order).itertuples():
        lines.append(
            f"| {r.instrument} | {LABELS[r.rule]} | {_pct(r.cagr)} | "
            f"{'' if r.rule == 'B' else f'{100 * r.gap:+.1f} pts'} | {_pct(r.max_drawdown)} | "
            f"{_pct(r.worst_year)} | {_pct(r.months_2pct)} | {r.trades_per_year:.1f} | "
            f"{_pct(r.avg_trade_after_tax, plus=True)} | "
            f"{'' if r.rule == 'B' else ('yes' if r.passes else 'no')} |"
        )
    return "\n".join(lines) + "\n"


def _rule_order(col: pd.Series) -> pd.Series:
    if col.name == "rule":
        return col.map({r: i for i, r in enumerate(RULES)})
    return col
