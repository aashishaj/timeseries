"""Model families. Register new families in MODEL_REGISTRY so configs can refer to them."""

from __future__ import annotations

from forecasting.models.base import ForecastModel
from forecasting.models.baselines import HistoricalMean, RandomWalk, SeasonalNaive

MODEL_REGISTRY: dict[str, type[ForecastModel]] = {
    "random_walk": RandomWalk,
    "historical_mean": HistoricalMean,
    "seasonal_naive": SeasonalNaive,
}


def build_model(type_: str, name: str | None = None, params: dict | None = None) -> ForecastModel:
    try:
        cls = MODEL_REGISTRY[type_]
    except KeyError as exc:
        raise KeyError(f"unknown model type {type_!r}; known: {sorted(MODEL_REGISTRY)}") from exc
    kwargs = dict(params or {})
    if name is not None:
        kwargs["name"] = name
    return cls(**kwargs)


__all__ = [
    "MODEL_REGISTRY",
    "ForecastModel",
    "HistoricalMean",
    "RandomWalk",
    "SeasonalNaive",
    "build_model",
]
