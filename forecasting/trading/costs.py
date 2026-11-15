"""Transaction costs for NSE equity trades: intraday (MIS) by default, delivery (CNC)
with ``configs/costs_delivery.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

DEFAULT_COSTS = Path(__file__).resolve().parents[1] / "configs" / "costs.yaml"
DELIVERY_COSTS = DEFAULT_COSTS.with_name("costs_delivery.yaml")


@dataclass(frozen=True)
class CostModel:
    brokerage_rate: float = 0.0003
    brokerage_cap_inr: float = 20.0
    stt_sell_rate: float = 0.00025
    exchange_rate: float = 0.0000297
    sebi_rate: float = 0.000001
    stamp_buy_rate: float = 0.00003
    gst_rate: float = 0.18
    slippage_bps: float = 2.0
    stt_buy_rate: float = 0.0  # delivery only: STT is charged on the buy side too
    dp_sell_inr: float = 0.0  # delivery only: depository fee per stock sold per day

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> CostModel:
        with open(path or DEFAULT_COSTS) as fh:
            return cls(**yaml.safe_load(fh))

    def side(self, notional: np.ndarray, buy: bool) -> np.ndarray:
        """Charges in INR for one executed order of ``notional`` INR."""
        n = np.abs(np.asarray(notional, dtype=float))
        brokerage = np.minimum(self.brokerage_rate * n, self.brokerage_cap_inr) * (n > 0)
        exchange = self.exchange_rate * n
        sebi = self.sebi_rate * n
        gst = self.gst_rate * (brokerage + exchange + sebi)
        if buy:
            tax = (self.stamp_buy_rate + self.stt_buy_rate) * n
        else:
            tax = self.stt_sell_rate * n + self.dp_sell_inr * (n > 0)
        slippage = self.slippage_bps * 1e-4 * n
        return brokerage + exchange + sebi + gst + tax + slippage

    def round_trip(self, notional: np.ndarray) -> np.ndarray:
        """Buy at the open and sell at the close (or short and cover) for ``notional``."""
        return self.side(notional, buy=True) + self.side(notional, buy=False)
