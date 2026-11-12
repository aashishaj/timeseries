"""Command line entry point: ``forecasting {synth,validate,backtest}``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from forecasting.backtest.engine import (
    development_panel,
    lockbox_walk_forward,
    run_walk_forward,
)
from forecasting.backtest.evaluate import evaluate
from forecasting.backtest.report import (
    LEGEND,
    append_leaderboard,
    summary_markdown,
    write_results,
)
from forecasting.config import DEFAULT_EXPERIMENT, ExperimentConfig, load_config
from forecasting.data.loader import load_candles
from forecasting.data.resample import BuildReport, build_bars
from forecasting.features.returns import add_returns
from forecasting.models import build_model

log = logging.getLogger("forecasting")


def load_panel(cfg: ExperimentConfig) -> tuple[pd.DataFrame, BuildReport]:
    log.info("reading candles from %s (read-only)", cfg.data_root)
    candles = load_candles(cfg.data_root, cfg.symbols, cfg.start, cfg.end, cfg.schema)
    if candles.empty:
        raise SystemExit(f"no candles found under {cfg.data_root} for {cfg.symbols}")
    panel, report = build_bars(
        candles,
        cfg.calendar,
        symbols=cfg.symbols,
        bar_minutes=cfg.bar_minutes,
        start=cfg.start,
        end=cfg.end,
        source_interval_minutes=cfg.schema.source_interval_minutes,
    )
    log.info(
        "%d sessions x %d symbols x %d bars",
        report.n_sessions,
        len(panel.attrs["symbols"]),
        panel.attrs["bars_per_session"],
    )
    return add_returns(panel), report


def cmd_synth(args: argparse.Namespace) -> int:
    from forecasting.data.synthetic import generate_candles, write_export

    cfg = load_config(args.config)
    symbols = cfg.symbols[: args.n_symbols] if args.n_symbols else cfg.symbols
    t0 = time.perf_counter()
    df = generate_candles(symbols, cfg.calendar, args.start, args.end, seed=args.seed)
    write_export(df, args.out)
    log.info(
        "wrote %d synthetic candles to %s in %.1fs", len(df), args.out, time.perf_counter() - t0
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, data_root=args.data_root)
    panel, report = load_panel(cfg)
    md = report.to_markdown(cfg.calendar)
    print(md)
    if args.report:
        Path(args.report).write_text(md + "\n")
    problems = report.unexpected_dates or report.empty_sessions or report.symbols_without_data
    return 1 if (problems and args.strict) else 0


def model_spec_string(spec: dict) -> str:
    params = "".join(f", {k}={v}" for k, v in sorted((spec.get("params") or {}).items()))
    return f"{spec['type']}{params}"


def data_label(root: Path) -> str:
    from forecasting.data.synthetic import MARKER

    return "synthetic" if (Path(root) / MARKER).exists() else f"real:{Path(root).name}"


def cmd_backtest(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, data_root=args.data_root)
    wf = cfg.walk_forward
    if args.max_folds is not None:
        wf = type(wf)(**{**wf.__dict__, "max_folds": args.max_folds})
    horizons = [h for h in cfg.horizons if not args.horizon or h.name in args.horizon]
    specs = [m for m in cfg.models if not args.model or m["name"] in args.model]
    if cfg.baseline not in {m["name"] for m in specs}:
        specs = [m for m in cfg.models if m["name"] == cfg.baseline] + specs

    panel, report = load_panel(cfg)
    label = data_label(cfg.data_root)
    if args.final:
        split = "final"
        wf = lockbox_walk_forward(panel, wf)
        _log_lockbox_opening(Path(args.leaderboard).parent, cfg.name, label, specs)
    else:
        split = "dev"
        panel = development_panel(panel, wf)
    outputs = []
    for h in horizons:
        for spec in specs:
            model = build_model(spec["type"], spec["name"], spec.get("params"))
            outputs.append(run_walk_forward(panel, model, h, wf))
    forecasts = pd.concat(outputs, ignore_index=True)
    hmap = {h.name: h for h in horizons}
    rungs = {m["name"]: m["rung"] for m in specs if m.get("rung") is not None}
    evaluation = evaluate(forecasts, baseline=cfg.baseline, horizons=hmap, rungs=rungs)

    print(summary_markdown(evaluation, hmap))
    print(LEGEND)
    out_dir = Path(args.out) if args.out else cfg.output_dir
    first, last = forecasts["session"].min(), forecasts["session"].max()
    meta = {
        "experiment": cfg.name,
        "data": f"{label} ({cfg.data_root})",
        "split": f"{split}: scored sessions {first} -> {last}"
        + (" (LOCKBOX)" if args.final else f"; last {wf.holdout_sessions} sessions held out"),
        "symbols": ", ".join(panel.attrs["symbols"]),
        "models": "; ".join(f"{m['name']} = {model_spec_string(m)}" for m in specs),
    }
    path = write_results(
        out_dir,
        title=args.title or f"{cfg.name} ({label}, {split})",
        forecasts=forecasts,
        evaluation=evaluation,
        horizons=hmap,
        walk_forward=wf,
        meta=meta,
        quality_markdown=report.to_markdown(cfg.calendar),
    )
    if args.leaderboard:
        append_leaderboard(
            args.leaderboard,
            evaluation,
            experiment=cfg.name,
            data=label,
            split=split,
            model_specs={m["name"]: model_spec_string(m) for m in specs},
            run_at=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
        )
    log.info("results written to %s", path)
    return 0


def _log_lockbox_opening(folder: Path, experiment: str, label: str, specs: list[dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "lockbox_log.csv"
    previous = pd.read_csv(path) if path.exists() else pd.DataFrame(columns=["data"])
    opened = int((previous["data"] == label).sum())
    if opened:
        log.warning(
            "the lockbox for %s has been opened %d time(s) before; every extra look "
            "makes it less of an honest out-of-sample test",
            label,
            opened,
        )
    row = pd.DataFrame(
        [
            {
                "opened_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M"),
                "experiment": experiment,
                "data": label,
                "models": "; ".join(f"{m['name']} = {model_spec_string(m)}" for m in specs),
            }
        ]
    )
    pd.concat([previous, row], ignore_index=True).to_csv(path, index=False)


def cmd_papertrade(args: argparse.Namespace) -> int:
    from forecasting.trading.costs import CostModel
    from forecasting.trading.vol_target import VolTargetConfig, vol_target_backtest

    cfg = load_config(args.config, data_root=args.data_root)
    results = Path(args.results)
    forecasts = pd.read_parquet(results / "forecasts.parquet")
    forecasts = forecasts[forecasts["horizon"] == args.horizon]
    if args.model:
        forecasts = forecasts[forecasts["model"].isin(args.model)]
    if forecasts.empty:
        raise SystemExit(f"no {args.horizon} forecasts in {results}/forecasts.parquet")
    panel, _ = load_panel(cfg)
    vt = VolTargetConfig(
        target_annual_vol=args.target_vol,
        capital_inr=args.capital,
        max_gross_leverage=args.max_gross,
    )
    result = vol_target_backtest(panel, forecasts, vt, CostModel.from_yaml(args.costs))
    md = result.to_markdown(vt)
    print(md)
    out = Path(args.out) if args.out else results / "paper_trading.md"
    out.write_text(md + "\n")
    result.daily.to_csv(out.with_suffix(".csv"))
    log.info("paper trading report written to %s", out)
    return 0


def cmd_run_all(args: argparse.Namespace) -> int:
    """Validate -> development backtest of every model -> paper trading -> REPORT.md."""
    from forecasting.backtest.evaluate import plain_accuracy

    cfg = load_config(args.config, data_root=args.data_root)
    out = Path(args.out) if args.out else Path("results") / f"run_{Path(cfg.data_root).name}"
    out.mkdir(parents=True, exist_ok=True)
    common = {"config": args.config, "data_root": args.data_root}
    cmd_validate(argparse.Namespace(**common, report=str(out / "data_quality.md"), strict=False))
    cmd_backtest(
        argparse.Namespace(
            **common,
            out=str(out),
            horizon=None,
            model=args.model,
            max_folds=args.max_folds,
            title=None,
            final=False,
            leaderboard=args.leaderboard,
        )
    )
    cmd_papertrade(
        argparse.Namespace(
            **common,
            results=str(out),
            horizon="next_session",
            model=None,
            target_vol=0.15,
            capital=1_000_000.0,
            max_gross=2.0,
            costs=None,
            out=None,
        )
    )
    metrics = pd.read_csv(out / "metrics.csv")
    accuracy = plain_accuracy(pd.read_parquet(out / "forecasts.parquet"))
    report = _run_report(cfg.name, Path(cfg.data_root), metrics, accuracy, out)
    (out / "REPORT.md").write_text(report)
    print(report)
    log.info("report written to %s", out / "REPORT.md")
    return 0


def _run_report(name: str, root: Path, metrics, accuracy, out: Path) -> str:
    lines = [
        f"# {name} on {data_label(root)}",
        "",
        "Development split only: the lockbox (last `holdout_sessions`) is still sealed.",
        "Lower QLIKE / naive is better; `vs best lower rung` < 1 with a small p means the",
        "model earned its place on the ladder.",
        "",
    ]
    for hname, m in metrics.groupby("horizon", sort=False):
        acc = accuracy[accuracy["horizon"] == hname].set_index("model")
        m = m.sort_values("QLIKE_vs_naive")
        lines += [
            f"## {hname}",
            "",
            "| model | rung | QLIKE / naive | vs best lower rung (p) | explained | "
            "median vol error | within +-25% |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in m.itertuples():
            vs = (
                f"{r.QLIKE_vs_lower_rungs:.3f} ({r.DM_p_vs_lower_rungs:.2g})"
                if pd.notna(r.QLIKE_vs_lower_rungs)
                else "-"
            )
            a = acc.loc[r.model]
            lines.append(
                f"| {r.model} | {r.rung if pd.notna(r.rung) else '-'} | {r.QLIKE_vs_naive:.3f} | "
                f"{vs} | {100 * a.explained:.0f}% | {100 * a.median_vol_error:.0f}% | "
                f"{100 * a.within_25pct:.0f}% |"
            )
        lines.append("")
    lines += [(out / "paper_trading.md").read_text(), "", (out / "data_quality.md").read_text()]
    return "\n".join(lines) + "\n"


def cmd_kite_login(args: argparse.Namespace) -> int:
    from forecasting.data.kite_backfill import interactive_login

    path = interactive_login(
        Path(args.token_file) if args.token_file else None,
        port=args.port,
        open_browser=not args.no_browser,
    )
    print(f"Access token saved to {path} (readable only by you). It expires daily.")
    return 0


def cmd_backfill_kite(args: argparse.Namespace) -> int:
    import datetime as dt

    from forecasting.data.kite_backfill import backfill, make_client

    cfg = load_config(args.config)
    symbols = cfg.symbols[: args.n_symbols] if args.n_symbols else cfg.symbols
    end = dt.date.fromisoformat(args.end) if args.end else dt.date.today()
    start = dt.date.fromisoformat(args.start) if args.start else end - dt.timedelta(days=365)
    client = make_client(Path(args.token_file) if args.token_file else None)
    written = backfill(client, symbols, start, end, args.out)
    log.info("backfill done: %s", written)
    print(f"Wrote {sum(written.values())} new symbol-days to {args.out}. Next:")
    print(f"  uv run forecasting run-all --data-root {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="forecasting", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("synth", help="write a synthetic candle export (for tests/demos)")
    p.add_argument("--config", default=DEFAULT_EXPERIMENT)
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--n-symbols", type=int, default=None)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("validate", help="data-quality report for a candle export")
    p.add_argument("--config", default=DEFAULT_EXPERIMENT)
    p.add_argument("--data-root", default=None, help="overrides data.root in the config")
    p.add_argument("--report", default=None, help="also write the report to this file")
    p.add_argument("--strict", action="store_true", help="exit 1 if calendar/data disagree")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("backtest", help="walk-forward backtest + results table")
    p.add_argument("--config", default=DEFAULT_EXPERIMENT)
    p.add_argument("--data-root", default=None, help="overrides data.root in the config")
    p.add_argument("--out", default=None, help="output directory (default: config output_dir)")
    p.add_argument("--horizon", action="append", help="only these horizons (repeatable)")
    p.add_argument("--model", action="append", help="only these models (repeatable)")
    p.add_argument("--max-folds", type=int, default=None, help="cap folds for a quick run")
    p.add_argument("--title", default=None)
    p.add_argument(
        "--final",
        action="store_true",
        help="score the held-out lockbox sessions (do this rarely; every opening is logged)",
    )
    p.add_argument(
        "--leaderboard",
        default="results/leaderboard.csv",
        help="trial log to append to ('' to disable)",
    )
    p.set_defaults(func=cmd_backtest)

    p = sub.add_parser(
        "papertrade", help="simulate volatility-targeted intraday trading from backtest forecasts"
    )
    p.add_argument("--config", default=DEFAULT_EXPERIMENT)
    p.add_argument("--data-root", default=None, help="overrides data.root in the config")
    p.add_argument("--results", required=True, help="backtest output dir (forecasts.parquet)")
    p.add_argument("--horizon", default="next_session")
    p.add_argument("--model", action="append", help="only these models (repeatable)")
    p.add_argument("--target-vol", type=float, default=0.15, help="annualized, e.g. 0.15")
    p.add_argument("--capital", type=float, default=1_000_000.0, help="INR")
    p.add_argument("--max-gross", type=float, default=2.0, help="gross exposure cap")
    p.add_argument("--costs", default=None, help="cost YAML (default configs/costs.yaml)")
    p.add_argument("--out", default=None)
    p.set_defaults(func=cmd_papertrade)

    p = sub.add_parser("run-all", help="data check + every model + paper trading + REPORT.md")
    p.add_argument("--config", default=str(DEFAULT_EXPERIMENT.parent / "rv_ladder.yaml"))
    p.add_argument("--data-root", default=None, help="overrides data.root in the config")
    p.add_argument("--out", default=None, help="default: results/run_<data folder name>")
    p.add_argument("--model", action="append", help="only these models (repeatable)")
    p.add_argument("--max-folds", type=int, default=None)
    p.add_argument("--leaderboard", default="results/leaderboard.csv")
    p.set_defaults(func=cmd_run_all)

    p = sub.add_parser("kite-login", help="log in to Kite and save today's access token privately")
    p.add_argument("--token-file", default=None)
    p.add_argument("--port", type=int, default=8766, help="port of your app's redirect URL")
    p.add_argument("--no-browser", action="store_true", help="only print the login URL")
    p.set_defaults(func=cmd_kite_login)

    p = sub.add_parser("backfill-kite", help="download 1-min candles via the Kite historical API")
    p.add_argument("--config", default=DEFAULT_EXPERIMENT, help="for the symbol list")
    p.add_argument("--out", default="data/kite")
    p.add_argument("--start", default=None, help="YYYY-MM-DD (default: one year ago)")
    p.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    p.add_argument("--n-symbols", type=int, default=None)
    p.add_argument("--token-file", default=None)
    p.set_defaults(func=cmd_backfill_kite)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
