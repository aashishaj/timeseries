import datetime as dt

import pytest

from forecasting.timing.tax import TaxLedger, TaxRules, financial_year

R = TaxRules()  # 20% short, 12.5% long above 1.25 lakh, 4% cess
CESS = 1.04


def test_financial_year_runs_april_to_march():
    assert financial_year(dt.date(2025, 3, 31)) == 2024
    assert financial_year(dt.date(2025, 4, 1)) == 2025


def test_long_term_means_held_more_than_twelve_months():
    assert not R.is_long_term(dt.date(2024, 1, 10), dt.date(2025, 1, 10))
    assert R.is_long_term(dt.date(2024, 1, 10), dt.date(2025, 1, 11))
    assert not R.is_long_term(dt.date(2024, 2, 29), dt.date(2025, 2, 28))  # leap day
    assert R.is_long_term(dt.date(2024, 2, 29), dt.date(2025, 3, 1))
    assert TaxRules.from_yaml() == R  # the config file matches the defaults


def test_short_term_gain_taxed_at_20_percent_plus_cess():
    led = TaxLedger(R)
    led.record(100_000, long_term=False)
    assert led.close_year(2024) == pytest.approx(20_000 * CESS)


def test_long_term_gain_exempt_up_to_one_and_a_quarter_lakh():
    led = TaxLedger(R)
    led.record(100_000, long_term=True)
    assert led.close_year(2024) == 0.0
    led.record(225_000, long_term=True)
    assert led.close_year(2025) == pytest.approx(100_000 * 0.125 * CESS)


def test_short_term_loss_reduces_long_term_gain_but_not_the_other_way():
    led = TaxLedger(R)
    led.record(-50_000, long_term=False)
    led.record(300_000, long_term=True)
    assert led.close_year(2024) == pytest.approx((250_000 - 125_000) * 0.125 * CESS)
    led = TaxLedger(R)
    led.record(-50_000, long_term=True)
    led.record(100_000, long_term=False)
    assert led.close_year(2024) == pytest.approx(100_000 * 0.20 * CESS)  # no set-off
    assert led.carried_lt == [[2024, 50_000]]


def test_losses_carry_forward_eight_years_then_lapse():
    led = TaxLedger(R)
    led.record(-80_000, long_term=False)
    assert led.close_year(2015) == 0.0
    led.record(100_000, long_term=False)
    assert led.close_year(2016) == pytest.approx(20_000 * 0.20 * CESS)  # 80k used up

    led = TaxLedger(R)
    led.record(-80_000, long_term=False)
    led.close_year(2015)
    led.record(100_000, long_term=False)
    assert led.close_year(2024) == pytest.approx(100_000 * 0.20 * CESS)  # lapsed after 8


def test_carried_short_term_loss_used_on_short_term_gains_first():
    led = TaxLedger(R)
    led.record(-100_000, long_term=False)
    led.close_year(2020)
    led.record(60_000, long_term=False)
    led.record(200_000, long_term=True)
    # 60k of the loss wipes the short-term gain, 40k reduces the long-term gain
    expected = (200_000 - 40_000 - 125_000) * 0.125 * CESS
    assert led.close_year(2021) == pytest.approx(expected)


def test_what_if_tax_does_not_change_the_ledger():
    led = TaxLedger(R)
    led.record(50_000, long_term=False)
    assert led.tax_for(2024, extra_st=50_000) == pytest.approx(20_000 * CESS)
    assert led.st == 50_000 and led.carried_st == []
