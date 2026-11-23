"""Day-by-day simulation of one Rs-denominated delivery account trading one instrument.

* The decision made at the close of day t (``signal[t]``) is executed at the open of
  day t+1, in whole shares, all in or all out, paying delivery charges.
* Lots are matched first in, first out. STT is not deductible from gains; the other
  charges are (cost of acquisition / expenses of transfer).
* Tax for a financial year is paid on its first trading day after 31 March, selling
  shares if the cash is short (those sales count in the new year).
* ``value`` is what the account is worth after tax each evening: cash, plus the
  holding sold at the close net of charges, minus the tax this year would owe with
  that sale included. The last day's value is the final, all-in result.
* Optional overlays: ``take_profit`` sells once the close is high enough that selling
  would leave that fraction of profit after charges and short-term tax; ``stop_loss``
  sells after the close falls that fraction below the buy price. After either, the
  account waits for the signal's next fresh entry (False at one close, True at the
  next) before buying again.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from forecasting.timing.tax import TaxLedger, TaxRules, financial_year
from forecasting.trading.costs import CostModel


@dataclass
class Lot:
    bought: dt.date
    qty: int
    basis: float  # deductible cost of the whole lot: price x qty + non-STT buy charges


@dataclass
class Trade:
    bought: dt.date
    sold: dt.date
    cost: float  # everything paid to open (price x qty + all buy charges)
    proceeds: float  # everything received on close (price x qty - all sell charges)
    exit: str  # "signal", "take-profit", "stop-loss", "end"

    @property
    def gain(self) -> float:
        return self.proceeds - self.cost


@dataclass
class AccountResult:
    value: pd.Series  # after-tax walk-away value each day
    invested: pd.Series  # holding shares at that day's close
    trades: list[Trade] = field(default_factory=list)
    taxes: list[tuple[int, float]] = field(default_factory=list)  # (FY, tax paid)
    charges: float = 0.0


class Account:
    def __init__(self, capital: float, costs: CostModel, rules: TaxRules) -> None:
        self.cash = float(capital)
        self.costs, self.rules = costs, rules
        self.lots: list[Lot] = []
        self.ledger = TaxLedger(rules)
        self.charges = 0.0
        self.opened: tuple[dt.date, float] | None = None  # (date, total cost) of position
        self.entry_price = 0.0

    @property
    def qty(self) -> int:
        return sum(lot.qty for lot in self.lots)

    def _charges(self, notional: float, buy: bool) -> tuple[float, float]:
        """(all charges, STT part) for one order."""
        total = float(self.costs.side(np.array([notional]), buy=buy)[0])
        stt = notional * (self.costs.stt_buy_rate if buy else self.costs.stt_sell_rate)
        return total, stt

    def buy_all(self, day: dt.date, price: float) -> None:
        qty = math.floor(self.cash / (price * 1.003))
        while qty > 0 and qty * price + self._charges(qty * price, True)[0] > self.cash:
            qty -= 1
        while (qty + 1) * price + self._charges((qty + 1) * price, True)[0] <= self.cash:
            qty += 1
        if qty <= 0:
            return
        notional = qty * price
        total, stt = self._charges(notional, True)
        self.cash -= notional + total
        self.charges += total
        self.lots.append(Lot(day, qty, notional + total - stt))
        self.opened = (day, notional + total)
        self.entry_price = price

    def sell(self, day: dt.date, price: float, qty: int | None = None) -> float:
        """Sell ``qty`` shares (all if None), oldest lots first; returns net proceeds."""
        held = self.qty
        qty = held if qty is None else min(qty, held)
        if qty <= 0:
            return 0.0
        if qty < held and self.opened:  # partial sale (to pay tax): shrink the position's cost
            self.opened = (self.opened[0], self.opened[1] * (held - qty) / held)
        notional = qty * price
        total, stt = self._charges(notional, False)
        deductible = total - stt
        left = qty
        while left:
            lot = self.lots[0]
            take = min(left, lot.qty)
            basis = lot.basis * take / lot.qty
            gain = take * price - deductible * take / qty - basis
            self.ledger.record(gain, self.rules.is_long_term(lot.bought, day))
            lot.basis -= basis
            lot.qty -= take
            left -= take
            if lot.qty == 0:
                self.lots.pop(0)
        proceeds = notional - total
        self.cash += proceeds
        self.charges += total
        return proceeds

    def walk_away_value(self, fy: int, price: float, day: dt.date) -> float:
        qty = self.qty
        if qty == 0:
            return self.cash - self.ledger.tax_for(fy)
        notional = qty * price
        total, stt = self._charges(notional, False)
        extra_st = extra_lt = 0.0
        for lot in self.lots:
            gain = lot.qty * price - (total - stt) * lot.qty / qty - lot.basis
            if self.rules.is_long_term(lot.bought, day):
                extra_lt += gain
            else:
                extra_st += gain
        return self.cash + notional - total - self.ledger.tax_for(fy, extra_st, extra_lt)

    def take_profit_price(self, target: float) -> float:
        """Lowest price at which selling the whole position leaves ``target`` x cost
        after charges and short-term tax (the worst case for a fresh position)."""
        qty, cost = self.qty, self.opened[1]
        basis = sum(lot.basis for lot in self.lots)
        rate = self.rules.stcg_rate * (1 + self.rules.cess_rate)

        def after_tax_profit(p: float) -> float:
            total, stt = self._charges(qty * p, False)
            taxable = qty * p - (total - stt) - basis
            return qty * p - total - cost - max(0.0, taxable) * rate

        lo, hi = cost / qty * 0.5, cost / qty * 3.0
        for _ in range(60):
            mid = (lo + hi) / 2
            if after_tax_profit(mid) >= target * cost:
                hi = mid
            else:
                lo = mid
        return hi


def run_account(
    px: pd.DataFrame,
    signal: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    capital: float = 1_000_000.0,
    costs: CostModel | None = None,
    rules: TaxRules | None = None,
    take_profit: float | None = None,
    stop_loss: float | None = None,
) -> AccountResult:
    """Simulate ``signal`` (decided at each close) on ``px`` from ``start`` to ``end``."""
    from forecasting.trading.costs import DELIVERY_COSTS

    costs = costs or CostModel.from_yaml(DELIVERY_COSTS)
    rules = rules or TaxRules()
    acct = Account(capital, costs, rules)
    dates = px.index
    sig = signal.reindex(dates)
    opens, closes = px["open"].to_numpy(), px["close"].to_numpy()
    positions = np.flatnonzero((dates >= start) & (dates <= end))
    if len(positions) == 0 or positions[0] == 0:
        raise ValueError("the period needs at least one day of history before it")
    values, invested, trades, taxes = [], [], [], []
    fy = financial_year(dates[positions[0]].date())
    blocked, exit_now, exit_kind, tp_price = False, False, "", math.inf
    for i in positions:
        day = dates[i].date()
        # tax for a finished financial year, paid at this open
        if financial_year(day) != fy:
            tax = acct.ledger.close_year(fy)
            taxes.append((fy, tax))
            fy = financial_year(day)
            if tax > acct.cash and acct.qty:
                need = tax - acct.cash
                acct.sell(day, opens[i], min(acct.qty, math.ceil(need / (opens[i] * 0.997)) + 1))
            acct.cash -= tax
        prev = sig.iloc[i - 1]
        want = bool(prev) if pd.notna(prev) else acct.qty > 0  # no decision: stay as is
        if blocked:
            fresh = bool(prev == 1) and i >= 2 and bool(sig.iloc[i - 2] == 0)
            blocked = not fresh
            want = want and not blocked
        if exit_now:
            want, blocked, exit_now = False, True, False
        if want and acct.qty == 0:
            acct.buy_all(day, opens[i])
            if take_profit is not None and acct.qty:
                tp_price = acct.take_profit_price(take_profit)
        elif not want and acct.qty:
            opened, cost = acct.opened
            proceeds = acct.sell(day, opens[i])
            trades.append(Trade(opened, day, cost, proceeds, exit_kind or "signal"))
            exit_kind = ""
        if acct.qty:
            if take_profit is not None and closes[i] >= tp_price:
                exit_now, exit_kind = True, "take-profit"
            if stop_loss is not None and closes[i] <= acct.entry_price * (1 - stop_loss):
                exit_now, exit_kind = True, "stop-loss"
        values.append(acct.walk_away_value(fy, closes[i], day))
        invested.append(acct.qty > 0)
    if acct.qty:  # count the open position as a trade closed at the last close
        opened, cost = acct.opened
        notional = acct.qty * closes[positions[-1]]
        proceeds = notional - acct._charges(notional, False)[0]
        trades.append(Trade(opened, dates[positions[-1]].date(), cost, proceeds, "end"))
    index = dates[positions]
    return AccountResult(
        value=pd.Series(values, index=index),
        invested=pd.Series(invested, index=index),
        trades=trades,
        taxes=taxes,
        charges=acct.charges,
    )
