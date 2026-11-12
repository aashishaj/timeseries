"""Model families. Register new families in MODEL_REGISTRY so configs can refer to them."""

from __future__ import annotations

from forecasting.models.base import ForecastModel
from forecasting.models.baselines import HistoricalMean, RandomWalk, SeasonalNaive
from forecasting.models.classical import HAR, ARMALogRV, SeasonalEWMA
from forecasting.models.deep import ChronosRVModel, NBEATSRVModel
from forecasting.models.ensemble import Combination
from forecasting.models.garch import SeasonalGARCH
from forecasting.models.ml import LightGBMRV

MODEL_REGISTRY: dict[str, type[ForecastModel]] = {
    "random_walk": RandomWalk,
    "historical_mean": HistoricalMean,
    "seasonal_naive": SeasonalNaive,
    "ewma": SeasonalEWMA,
    "har": HAR,
    "arma_logrv": ARMALogRV,
    "garch": SeasonalGARCH,
    "lightgbm": LightGBMRV,
    "nbeats": NBEATSRVModel,
    "chronos": ChronosRVModel,
    "combination": Combination,
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
    "ARMALogRV",
    "ChronosRVModel",
    "Combination",
    "MODEL_REGISTRY",
    "ForecastModel",
    "HAR",
    "HistoricalMean",
    "LightGBMRV",
    "NBEATSRVModel",
    "RandomWalk",
    "SeasonalEWMA",
    "SeasonalGARCH",
    "SeasonalNaive",
    "build_model",
]
