"""Rung 4: deep and foundation models for realized-variance forecasting.

Both models extend DailyComponentModel so they inherit the time-of-day profile
scaling and optional intraday correction that make classical models work on
intraday horizons.  The only thing each class provides is a daily-level forecast
of the next session's RV.

Dependencies (optional — install before using these models):
    uv pip install torch                            # N-BEATS
    uv pip install torch chronos-forecasting        # Chronos

The models import torch lazily; the rest of the package works without it.
"""

from __future__ import annotations

import os

# LightGBM and PyTorch both bundle an OpenMP runtime. On macOS this causes a
# crash when both are loaded in the same process. Setting this env var before
# torch is imported suppresses the duplicate-library check.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from typing import Self

import numpy as np
import pandas as pd

from forecasting.models.common import daily_rv, seasonal_profile
from forecasting.models.classical import DailyComponentModel
from forecasting.data.panel import n_symbols, panel_symbols


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for deep models.\n"
            "Install with:  uv pip install torch"
        ) from exc


def _qlike_log_loss(log_mu: "torch.Tensor", log_y: "torch.Tensor") -> "torch.Tensor":
    """QLIKE in log space: E[exp(log_y - log_mu) - (log_y - log_mu) - 1].

    Equivalent to y/mu - log(y/mu) - 1 when mu = exp(log_mu), y = exp(log_y).
    Gradient w.r.t. log_mu is (-y/mu + 1), which pushes mu toward y.
    """
    d = log_y - log_mu
    return (d.exp() - d - 1.0).mean()


# ---------------------------------------------------------------------------
# N-BEATS
# ---------------------------------------------------------------------------

class _NBEATSNet:
    """Thin wrapper around a PyTorch N-BEATS network (no nn.Module dependency at import)."""

    def __init__(self, lookback: int, n_blocks: int, width: int) -> None:
        torch = _require_torch()
        import torch.nn as nn

        class Block(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Sequential(
                    nn.Linear(lookback, width), nn.ReLU(),
                    nn.Linear(width, width), nn.ReLU(),
                )
                self.backcast = nn.Linear(width, lookback)
                self.forecast = nn.Linear(width, 1)

            def forward(self, x):
                h = self.fc(x)
                return self.backcast(h), self.forecast(h)

        class NBEATS(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])

            def forward(self, x):
                fc_total = torch.zeros(x.shape[0], 1, device=x.device)
                residual = x
                for blk in self.blocks:
                    bc, fc = blk(residual)
                    residual = residual - bc
                    fc_total = fc_total + fc
                return fc_total.squeeze(-1)

        self.net = NBEATS()
        self._torch = torch

    def train(
        self,
        x: np.ndarray,
        y: np.ndarray,
        n_epochs: int,
        lr: float,
        batch_size: int,
        seed: int,
    ) -> None:
        torch = self._torch
        torch.manual_seed(seed)
        self.net.train()
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        xt = torch.tensor(x, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.float32)
        n = len(xt)
        rng = np.random.default_rng(seed)
        for _ in range(n_epochs):
            idx = rng.permutation(n)
            for start in range(0, n, batch_size):
                b = torch.tensor(idx[start : start + batch_size])
                opt.zero_grad()
                loss = _qlike_log_loss(self.net(xt[b]), yt[b])
                loss.backward()
                opt.step()

    def predict(self, x: np.ndarray) -> np.ndarray:
        torch = self._torch
        self.net.eval()
        with torch.no_grad():
            return self.net(torch.tensor(x, dtype=torch.float32)).numpy()


class NBEATSRVModel(DailyComponentModel):
    """N-BEATS (Oreshkin et al. 2020) on log daily RV, pooled across symbols.

    Uses a generic (non-interpretable) stack with QLIKE loss in log space.
    The forecast is in log-RV space; the parent class converts it to variance
    and applies the time-of-day profile + optional intraday correction.
    """

    def __init__(
        self,
        name: str = "nbeats",
        lookback_days: int = 30,
        n_blocks: int = 4,
        width: int = 64,
        n_epochs: int = 100,
        lr: float = 1e-3,
        batch_size: int = 128,
        min_train_windows: int = 200,
        seed: int = 0,
        intraday_beta: float = 0.0,
        pooled_seasonality: bool = True,
        min_coverage: float = 0.8,
    ) -> None:
        super().__init__(name, intraday_beta, pooled_seasonality, min_coverage)
        self.lookback_days = lookback_days
        self.n_blocks = n_blocks
        self.width = width
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.min_train_windows = min_train_windows
        self.seed = seed
        self.history_days = lookback_days

    def _fit_daily(self, rv: np.ndarray) -> None:
        """Train on sliding windows of log daily RV, pooled across symbols."""
        d, s = rv.shape
        log_rv = np.log(np.where(rv > 0, rv, np.nan))

        # Build (lookback → target) pairs; skip windows with any NaN.
        windows_x, windows_y = [], []
        for j in range(s):
            seq = log_rv[:, j]
            for t in range(self.lookback_days, d):
                x_win = seq[t - self.lookback_days : t]
                y_val = seq[t]
                if np.isfinite(x_win).all() and np.isfinite(y_val):
                    # Normalize by window mean so the network sees de-meaned series.
                    mu = x_win.mean()
                    windows_x.append(x_win - mu)
                    windows_y.append(y_val - mu)

        if len(windows_x) < self.min_train_windows:
            self.net_ = None
            return

        x_arr = np.array(windows_x, dtype=np.float32)
        y_arr = np.array(windows_y, dtype=np.float32)
        self.net_ = _NBEATSNet(self.lookback_days, self.n_blocks, self.width)
        self.net_.train(x_arr, y_arr, self.n_epochs, self.lr, self.batch_size, self.seed)

    def _forecast_daily(self, rv: np.ndarray) -> np.ndarray:
        s = rv.shape[1]
        if self.net_ is None:
            return np.full(s, np.nan)
        log_rv = np.log(np.where(rv > 0, rv, np.nan))
        out = np.full(s, np.nan)
        for j in range(s):
            seq = log_rv[-self.lookback_days :, j]
            finite = seq[np.isfinite(seq)]
            if len(finite) < max(5, self.lookback_days // 3):
                continue
            mu = finite.mean()
            # Mean-impute NaN positions; network was trained on clean windows.
            seq_filled = np.where(np.isfinite(seq), seq, mu)
            x_norm = (seq_filled - mu).reshape(1, -1).astype(np.float32)
            log_fc = float(self.net_.predict(x_norm)[0]) + mu
            out[j] = np.exp(log_fc)
        return out


# ---------------------------------------------------------------------------
# Chronos (zero-shot foundation model)
# ---------------------------------------------------------------------------

class ChronosRVModel(DailyComponentModel):
    """Zero-shot Chronos-T5 on log daily RV.

    Downloads model weights on first ``fit`` call (cached by HuggingFace hub).
    No training: ``_fit_daily`` only stores the model variant and sets up the
    pipeline. ``_forecast_daily`` runs inference on each symbol independently.

    Requires:  uv pip install torch chronos-forecasting
    """

    def __init__(
        self,
        name: str = "chronos",
        model_id: str = "amazon/chronos-t5-tiny",
        num_samples: int = 100,
        context_days: int = 60,
        intraday_beta: float = 0.0,
        pooled_seasonality: bool = True,
        min_coverage: float = 0.8,
    ) -> None:
        super().__init__(name, intraday_beta, pooled_seasonality, min_coverage)
        self.model_id = model_id
        self.num_samples = num_samples
        self.context_days = context_days
        self.history_days = context_days

    def fit(self, train_df: pd.DataFrame) -> Self:
        """Load the Chronos pipeline and store the seasonal profile."""
        try:
            from chronos import BaseChronosPipeline
        except ImportError as exc:
            raise ImportError(
                "chronos-forecasting is required for ChronosRVModel.\n"
                "Install with:  uv pip install torch chronos-forecasting"
            ) from exc

        torch = _require_torch()
        self.profile_ = seasonal_profile(train_df, self.pooled_seasonality)
        self.symbols_ = panel_symbols(train_df)
        self.pipeline_ = BaseChronosPipeline.from_pretrained(
            self.model_id,
            device_map="cpu",
            torch_dtype=torch.float32,
        )
        rv, _ = daily_rv(train_df, self.profile_, self.min_coverage)
        self._fit_daily(rv)
        self.cache_ = None
        return self

    def _fit_daily(self, rv: np.ndarray) -> None:
        pass  # Chronos is zero-shot; nothing to fit

    def _forecast_daily(self, rv: np.ndarray) -> np.ndarray:
        torch = _require_torch()
        s = rv.shape[1]
        log_rv = np.log(np.where(rv > 0, rv, np.nan))
        contexts = []
        valid = []
        for j in range(s):
            seq = log_rv[-self.context_days :, j]
            finite = seq[np.isfinite(seq)]
            if len(finite) < 5:
                valid.append(False)
                contexts.append(torch.zeros(1))
            else:
                valid.append(True)
                contexts.append(torch.tensor(finite, dtype=torch.float32))

        out = np.full(s, np.nan)
        active = [c for c, v in zip(contexts, valid) if v]
        active_idx = [j for j, v in enumerate(valid) if v]
        if not active:
            return out

        # chronos.predict returns (n_series, num_samples, prediction_length)
        samples = self.pipeline_.predict(
            context=active,
            prediction_length=1,
            num_samples=self.num_samples,
        )
        medians = samples[:, :, 0].median(dim=1).values.numpy()
        for k, j in enumerate(active_idx):
            out[j] = float(np.exp(medians[k]))
        return out
