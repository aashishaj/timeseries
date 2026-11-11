"""Command line entry point: ``forecasting {synth,validate,backtest}``."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

from forecasting.backtest.engine import run_walk_forward
from forecasting.backtest.evaluate import evaluate
from forecasting.backtest.report import LEGEND, summary_markdown, write_results
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
    outputs = []
    for h in horizons:
        for spec in specs:
            model = build_model(spec["type"], spec["name"], spec.get("params"))
            outputs.append(run_walk_forward(panel, model, h, wf))
    forecasts = pd.concat(outputs, ignore_index=True)
    hmap = {h.name: h for h in horizons}
    evaluation = evaluate(forecasts, baseline=cfg.baseline, horizons=hmap)

    print(summary_markdown(evaluation, hmap))
    print(LEGEND)
    out_dir = Path(args.out) if args.out else cfg.output_dir
    meta = {
        "experiment": cfg.name,
        "data_root": str(cfg.data_root),
        "symbols": ", ".join(panel.attrs["symbols"]),
        "sessions": f"{report.n_sessions} ({report.first_session} -> {report.last_session})",
        "models": ", ".join(
            f"{m['name']} ({m['type']} {m.get('params') or ''})".strip() for m in specs
        ),
    }
    path = write_results(
        out_dir,
        title=args.title or f"{cfg.name}: realized volatility baselines",
        forecasts=forecasts,
        evaluation=evaluation,
        horizons=hmap,
        walk_forward=wf,
        meta=meta,
        quality_markdown=report.to_markdown(cfg.calendar),
    )
    log.info("results written to %s", path)
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
    p.set_defaults(func=cmd_backtest)
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
