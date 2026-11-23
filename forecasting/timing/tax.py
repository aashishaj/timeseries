"""Capital gains tax on listed shares, one financial year (April-March) at a time.

Set-off follows the Income Tax Act for capital gains:
* within a year, short-term losses can reduce short-term or long-term gains, long-term
  losses only long-term gains;
* losses left over are carried forward (8 years) under the same restriction,
  oldest first;
* the yearly long-term exemption applies after set-off.
Losses are used against the gains taxed at the higher rate first, which is allowed
and is what a taxpayer would choose.
"""

from __future__ import annotations

import calendar
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_TAXES = Path(__file__).resolve().parents[1] / "configs" / "taxes.yaml"


@dataclass(frozen=True)
class TaxRules:
    stcg_rate: float = 0.20
    ltcg_rate: float = 0.125
    ltcg_exempt_inr: float = 125_000.0
    cess_rate: float = 0.04
    long_term_after_months: int = 12
    carry_forward_years: int = 8

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> TaxRules:
        with open(path or DEFAULT_TAXES) as fh:
            return cls(**yaml.safe_load(fh))

    def is_long_term(self, bought: dt.date, sold: dt.date) -> bool:
        """Held for more than ``long_term_after_months`` calendar months."""
        months = bought.month - 1 + self.long_term_after_months
        year, month = bought.year + months // 12, months % 12 + 1
        day = min(bought.day, calendar.monthrange(year, month)[1])  # 29 Feb -> 28 Feb
        return sold > dt.date(year, month, day)


def financial_year(day: dt.date) -> int:
    """FY 2024 = 1 April 2024 to 31 March 2025."""
    return day.year if day.month >= 4 else day.year - 1


@dataclass
class TaxLedger:
    """Realized gains of the current financial year plus losses carried forward."""

    rules: TaxRules = field(default_factory=TaxRules)
    st: float = 0.0  # net short-term result so far this year (negative = loss)
    lt: float = 0.0  # net long-term result so far this year
    carried_st: list[list] = field(default_factory=list)  # [[fy, amount > 0], ...]
    carried_lt: list[list] = field(default_factory=list)

    def record(self, gain: float, long_term: bool) -> None:
        if long_term:
            self.lt += gain
        else:
            self.st += gain

    def tax_for(self, fy: int, extra_st: float = 0.0, extra_lt: float = 0.0) -> float:
        """Tax due for ``fy`` if it closed now with these extra gains (no changes made)."""
        tax, _, _ = self._settle(fy, self.st + extra_st, self.lt + extra_lt)
        return tax

    def close_year(self, fy: int) -> float:
        """Tax due for ``fy``; carries remaining losses forward and starts a new year."""
        tax, carried_st, carried_lt = self._settle(fy, self.st, self.lt)
        self.carried_st, self.carried_lt = carried_st, carried_lt
        self.st = self.lt = 0.0
        return tax

    def _settle(self, fy: int, st: float, lt: float) -> tuple[float, list, list]:
        r = self.rules
        keep = fy - r.carry_forward_years
        carried_st = [[y, a] for y, a in self.carried_st if y >= keep and a > 0]
        carried_lt = [[y, a] for y, a in self.carried_lt if y >= keep and a > 0]
        # this year's losses
        if st < 0 and lt > 0:  # short-term loss may reduce long-term gains
            used = min(-st, lt)
            st, lt = st + used, lt - used
        # older losses, oldest first: short-term ones against short-term gains first
        for pool in carried_st:
            for target in ("st", "lt"):
                gain = st if target == "st" else lt
                if gain > 0 and pool[1] > 0:
                    used = min(pool[1], gain)
                    pool[1] -= used
                    if target == "st":
                        st -= used
                    else:
                        lt -= used
        for pool in carried_lt:
            if lt > 0 and pool[1] > 0:
                used = min(pool[1], lt)
                pool[1] -= used
                lt -= used
        new_st = [[y, a] for y, a in carried_st if a > 0]
        new_lt = [[y, a] for y, a in carried_lt if a > 0]
        if st < 0:
            new_st.append([fy, -st])
        if lt < 0:
            new_lt.append([fy, -lt])
        taxable_lt = max(0.0, lt - r.ltcg_exempt_inr)
        tax = (max(0.0, st) * r.stcg_rate + taxable_lt * r.ltcg_rate) * (1 + r.cess_rate)
        return tax, new_st, new_lt
