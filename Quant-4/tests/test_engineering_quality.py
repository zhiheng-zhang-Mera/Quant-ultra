import sys
from pathlib import Path

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Main.financial_invariants import validate_long_only_weights, validate_nav_accounting, validate_pit_prefix
from Main.performance_engineering import benchmark_alternative_events, benchmark_full_pool


@given(st.lists(st.floats(min_value=0, max_value=1, allow_nan=False, allow_infinity=False), min_size=1, max_size=30))
@settings(max_examples=80, deadline=None)
def test_property_normalized_long_only_portfolios_never_create_leverage(values):
    raw = np.asarray(values, dtype=float)
    weights = raw / max(raw.sum(), 1.0)
    result = validate_long_only_weights(pd.DataFrame([weights]))
    assert result["passed"]
    assert result["max_observed_gross"] <= 1.0 + 1e-10


@given(st.lists(st.floats(min_value=-0.2, max_value=0.2, allow_nan=False, allow_infinity=False), min_size=1, max_size=100),
       st.floats(min_value=0, max_value=0.01, allow_nan=False, allow_infinity=False))
@settings(max_examples=80, deadline=None)
def test_property_nav_identity_holds_for_valid_return_ledgers(returns, cost):
    gross = pd.Series(returns, dtype=float)
    costs = pd.Series(float(cost), index=gross.index)
    nav = (1.0 + gross - costs).cumprod()
    assert validate_nav_accounting(nav, gross, costs)["passed"]


def test_financial_invariants_fail_closed_on_financial_logic_errors():
    assert validate_long_only_weights(pd.DataFrame([[0.8, 0.4]]))["failures"] == ["gross_leverage"]
    result = validate_nav_accounting(pd.Series([1.0, 1.5]), pd.Series([0.0, 0.0]), pd.Series([0.0, 0.0]))
    assert result["failures"] == ["nav_identity"]


def test_pit_prefix_gate_detects_future_leakage():
    index = pd.date_range("2024-01-01", periods=8)
    before = {"factor": pd.Series(range(8), index=index, dtype=float)}
    after = {"factor": before["factor"].copy()}
    after["factor"].iloc[2] = 99.0
    assert not validate_pit_prefix(before, after, cutoff="2024-01-04")["passed"]
    after["factor"] = before["factor"].copy()
    after["factor"].iloc[-1] = 99.0
    assert validate_pit_prefix(before, after, cutoff="2024-01-04")["passed"]


def test_performance_fingerprints_are_stable_and_budgets_are_enforced():
    first = benchmark_full_pool(assets=200, days=260)
    second = benchmark_full_pool(assets=200, days=260)
    assert first["status"] == "PASS"
    assert first["result_sha256"] == second["result_sha256"]
    alt_first = benchmark_alternative_events(events=1000)
    alt_second = benchmark_alternative_events(events=1000)
    assert alt_first["status"] == "PASS"
    assert alt_first["result_sha256"] == alt_second["result_sha256"]
