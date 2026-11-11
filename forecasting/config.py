"""Experiment configuration (YAML) -> typed objects."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from forecasting.backtest.engine import Horizon, WalkForward
from forecasting.data.calendar import NSECalendar, to_date
from forecasting.data.loader import ExportSchema

CONFIG_DIR = Path(__file__).resolve().parent / "configs"
DEFAULT_EXPERIMENT = CONFIG_DIR / "rv_baselines.yaml"


@dataclass
class ExperimentConfig:
    name: str
    description: str
    data_root: Path
    schema: ExportSchema
    start: dt.date | None
    end: dt.date | None
    symbols: list[str]
    calendar: NSECalendar
    bar_minutes: int
    horizons: list[Horizon]
    walk_forward: WalkForward
    models: list[dict] = field(default_factory=list)
    baseline: str = "naive"
    output_dir: Path = Path("results")


def _read_yaml(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def _resolve(ref: str | Path, base: Path) -> Path:
    ref = Path(ref)
    return ref if ref.is_absolute() else base / ref


def load_symbols(path: str | Path) -> list[str]:
    symbols = _read_yaml(Path(path)).get("symbols") or []
    if not symbols or len(set(symbols)) != len(symbols):
        raise ValueError(f"{path}: need a non-empty list of unique symbols")
    return [str(s) for s in symbols]


def load_config(
    path: str | Path | None = None, *, data_root: str | Path | None = None
) -> ExperimentConfig:
    """Load an experiment config. Paths to other config files resolve relative to it;
    ``data.root`` resolves relative to the current working directory."""
    path = Path(path) if path else DEFAULT_EXPERIMENT
    raw = _read_yaml(path)
    base = path.parent
    calendar = NSECalendar.from_yaml(_resolve(raw.get("calendar", "nse_calendar.yaml"), base))
    symbols = load_symbols(_resolve(raw.get("symbols", "symbols.yaml"), base))
    data = raw.get("data", "data.yaml")
    data = _read_yaml(_resolve(data, base)) if isinstance(data, str) else dict(data)
    bar_minutes = int(raw.get("bar_minutes", 5))
    per_session = calendar.bars_per_session(bar_minutes)

    horizons = []
    for h in raw.get("horizons") or []:
        h = dict(h)
        if h.get("bars") == "session":
            h["bars"] = per_session
        horizons.append(Horizon(**h))
    if not horizons:
        raise ValueError(f"{path}: no horizons configured")
    models = list(raw.get("models") or [])
    names = [m["name"] for m in models]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: model names must be unique, got {names}")
    baseline = raw.get("baseline", "naive")
    if baseline not in names:
        raise ValueError(f"{path}: baseline {baseline!r} is not one of the models {names}")

    return ExperimentConfig(
        name=raw.get("name", path.stem),
        description=raw.get("description", ""),
        data_root=Path(data_root if data_root is not None else data.get("root", "data/export")),
        schema=ExportSchema.from_dict(data.get("schema")),
        start=to_date(data["start"]) if data.get("start") else None,
        end=to_date(data["end"]) if data.get("end") else None,
        symbols=symbols,
        calendar=calendar,
        bar_minutes=bar_minutes,
        horizons=horizons,
        walk_forward=WalkForward(**(raw.get("walk_forward") or {})),
        models=models,
        baseline=baseline,
        output_dir=Path(raw.get("output_dir", f"results/{path.stem}")),
    )
