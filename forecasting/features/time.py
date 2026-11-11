"""Calendar / time-of-day features. Known in advance, so trivially leakage-free."""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.data.panel import bars_per_session


def time_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Slot, cyclical time-of-day encoding and day of week for every panel row."""
    b = bars_per_session(panel)
    slot = panel["slot"].to_numpy()
    angle = 2 * np.pi * slot / b
    return pd.DataFrame(
        {
            "slot": slot,
            "tod_sin": np.sin(angle),
            "tod_cos": np.cos(angle),
            "day_of_week": panel["ts"].dt.day_of_week.to_numpy(),
        },
        index=panel.index,
    )
