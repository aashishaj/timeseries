"""Rung 4: deep and foundation models for realized-variance forecasting.

Both models extend DailyComponentModel so they inherit the time-of-day profile
scaling and optional intraday correction that make classical models work on
intraday horizons.  The only thing each class provides is a daily-level forecast
of the next session's RV.

Dependencies (optional — install before using these models):
    uv pip install torch                            # N-BEATS
    uv pip install torch chronos-forecasting        # Chronos

The models import torch lazily; the rest of the package works without it.

macOS OpenMP note
-----------------
LightGBM and PyTorch both bundle an OpenMP runtime. Loading both in the same
process causes a hang (even with KMP_DUPLICATE_LIB_OK=TRUE). The fix is to run
all torch operations in a dedicated subprocess spawned before lgbm is imported.
_get_torch_pool() returns a persistent ProcessPoolExecutor(spawn) for this.
_nbeats_train_job and _nbeats_predict_job are module-level so they are
picklable with the 'spawn' start method.
"""

from __future__ import annotations

import os
import concurrent.futures as cf
import multiprocessing as mp
from typing import Self

import numpy as np
import pandas as pd

from forecasting.models.common import daily_rv, seasonal_profile
from forecasting.models.classical import DailyComponentModel
from forecasting.data.panel import n_symbols, panel_symbols


# ---------------------------------------------------------------------------
# Subprocess pool: torch runs here, isolated from lgbm's OpenMP runtime.
# ---------------------------------------------------------------------------

_TORCH_POOL: cf.ProcessPoolExecutor | None = None


def _get_torch_pool() -> cf.ProcessPoolExecutor:
    global _TORCH_POOL
    if _TORCH_POOL is None:
        ctx = mp.get_context("spawn")
        _TORCH_POOL = cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx)
    return _TORCH_POOL


# ---------------------------------------------------------------------------
# Module-level worker functions (must be top-level to be picklable with spawn)
# ---------------------------------------------------------------------------

def _nbeats_train_job(
    x: np.ndarray,
    y: np.ndarray,
    lookback: int,
    n_blocks: int,
    width: int,
    n_epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Train N-BEATS in the subprocess; returns state dict as numpy arrays."""
    import os as _os
    _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import torch
    import torch.nn as nn
    import numpy as _np

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(lookback, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
            )
            self.bc = nn.Linear(width, lookback)
            self.fo = nn.Linear(width, 1)

        def forward(self, z):
            h = self.fc(z)
            return self.bc(h), self.fo(h)

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])

        def forward(self, z):
            fc = torch.zeros(z.shape[0], 1)
            r = z
            for blk in self.blocks:
                bc, f = blk(r)
                r = r - bc
                fc = fc + f
            return fc.squeeze(-1)

    torch.manual_seed(seed)
    net = _Net()
    net.train()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt = torch.tensor(x, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    n = len(xt)
    rng = _np.random.default_rng(seed)
    for _ in range(n_epochs):
        idx = rng.permutation(n)
        for start in range(0, n, batch_size):
            b = torch.tensor(idx[start : start + batch_size])
            opt.zero_grad()
            d = yt[b] - net(xt[b])  # log_y - log_mu
            (d.exp() - d - 1.0).mean().backward()  # QLIKE in log space
            opt.step()
    return {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}


def _nbeats_predict_job(
    x: np.ndarray,
    state_numpy: dict[str, np.ndarray],
    lookback: int,
    n_blocks: int,
    width: int,
) -> np.ndarray:
    """Run N-BEATS inference in the subprocess; returns predictions as numpy."""
    import os as _os
    _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import torch
    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(lookback, width), nn.ReLU(),
                nn.Linear(width, width), nn.ReLU(),
            )
            self.bc = nn.Linear(width, lookback)
            self.fo = nn.Linear(width, 1)

        def forward(self, z):
            h = self.fc(z)
            return self.bc(h), self.fo(h)

    class _Net(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([_Block() for _ in range(n_blocks)])

        def forward(self, z):
            fc = torch.zeros(z.shape[0], 1)
            r = z
            for blk in self.blocks:
                bc, f = blk(r)
                r = r - bc
                fc = fc + f
            return fc.squeeze(-1)

    net = _Net()
    net.load_state_dict({k: torch.tensor(v) for k, v in state_numpy.items()})
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(x, dtype=torch.float32)).cpu().numpy()


def _chronos_predict_job(
    contexts: list[np.ndarray],
    model_id: str,
    num_samples: int,
) -> np.ndarray:
    """Run Chronos inference in the subprocess; returns medians as numpy."""
    import os as _os
    _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    import torch
    from chronos import BaseChronosPipeline

    pipeline = BaseChronosPipeline.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float32
    )
    tensors = [torch.tensor(c, dtype=torch.float32) for c in contexts]
    samples = pipeline.predict(tensors, prediction_length=1, num_samples=num_samples)
    return samples[:, :, 0].median(dim=1).values.numpy()


# ---------------------------------------------------------------------------
# N-BEATS model
# ---------------------------------------------------------------------------

class NBEATSRVModel(DailyComponentModel):
    """N-BEATS (Oreshkin et al. 2020) on log daily RV, pooled across symbols.

    Uses a generic (non-interpretable) stack with QLIKE loss in log space.
    Training and inference run in a dedicated subprocess to avoid an OpenMP
    conflict with LightGBM on macOS (see module docstring).
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
        d, s = rv.shape
        log_rv = np.log(np.where(rv > 0, rv, np.nan))

        windows_x, windows_y = [], []
        for j in range(s):
            seq = log_rv[:, j]
            for t in range(self.lookback_days, d):
                x_win = seq[t - self.lookback_days : t]
                y_val = seq[t]
                if np.isfinite(x_win).all() and np.isfinite(y_val):
                    mu = x_win.mean()
                    windows_x.append(x_win - mu)
                    windows_y.append(y_val - mu)

        if len(windows_x) < self.min_train_windows:
            self.state_dict_: dict[str, np.ndarray] | None = None
            return

        x_arr = np.array(windows_x, dtype=np.float32)
        y_arr = np.array(windows_y, dtype=np.float32)
        try:
            self.state_dict_ = _get_torch_pool().submit(
                _nbeats_train_job,
                x_arr, y_arr,
                self.lookback_days, self.n_blocks, self.width,
                self.n_epochs, self.lr, self.batch_size, self.seed,
            ).result()
        except Exception:
            self.state_dict_ = None

    def _forecast_daily(self, rv: np.ndarray) -> np.ndarray:
        s = rv.shape[1]
        if self.state_dict_ is None:
            return np.full(s, np.nan)
        log_rv = np.log(np.where(rv > 0, rv, np.nan))
        out = np.full(s, np.nan)
        for j in range(s):
            seq = log_rv[-self.lookback_days :, j]
            finite = seq[np.isfinite(seq)]
            if len(finite) < max(5, self.lookback_days // 3):
                continue
            mu = finite.mean()
            seq_filled = np.where(np.isfinite(seq), seq, mu)
            x_norm = (seq_filled - mu).reshape(1, -1).astype(np.float32)
            try:
                log_fc = float(_get_torch_pool().submit(
                    _nbeats_predict_job,
                    x_norm, self.state_dict_,
                    self.lookback_days, self.n_blocks, self.width,
                ).result()[0]) + mu
                out[j] = np.exp(log_fc)
            except Exception:
                pass
        return out


# ---------------------------------------------------------------------------
# Chronos (zero-shot foundation model)
# ---------------------------------------------------------------------------

class ChronosRVModel(DailyComponentModel):
    """Zero-shot Chronos-T5 on log daily RV.

    Downloads model weights on first ``fit`` call (cached by HuggingFace hub).
    No training: ``_fit_daily`` is a no-op. Inference runs in the same torch
    subprocess as N-BEATS to avoid the macOS OpenMP conflict.

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
        self.profile_ = seasonal_profile(train_df, self.pooled_seasonality)
        self.symbols_ = panel_symbols(train_df)
        rv, _ = daily_rv(train_df, self.profile_, self.min_coverage)
        self._fit_daily(rv)
        self.cache_ = None
        return self

    def _fit_daily(self, rv: np.ndarray) -> None:
        pass  # zero-shot: nothing to train

    def _forecast_daily(self, rv: np.ndarray) -> np.ndarray:
        s = rv.shape[1]
        log_rv = np.log(np.where(rv > 0, rv, np.nan))
        contexts, valid_idx = [], []
        for j in range(s):
            seq = log_rv[-self.context_days :, j]
            finite = seq[np.isfinite(seq)]
            if len(finite) >= 5:
                valid_idx.append(j)
                contexts.append(finite)

        out = np.full(s, np.nan)
        if not contexts:
            return out
        try:
            medians = _get_torch_pool().submit(
                _chronos_predict_job, contexts, self.model_id, self.num_samples
            ).result()
            for k, j in enumerate(valid_idx):
                out[j] = float(np.exp(medians[k]))
        except Exception:
            pass
        return out
