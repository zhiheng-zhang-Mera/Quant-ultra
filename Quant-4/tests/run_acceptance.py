"""Minimal acceptance runner for embedded Python builds without unittest."""
from test_improvements import (
    test_data_quality_proves_valid_and_rejects_bad,
    test_fast_math_and_metrics,
    test_risk_parity_invariants,
    test_stock_and_etf_normalization,
    test_recommendation_has_required_four_outputs,
    test_holding_advice_reconciles_cash_position_and_action,
)

TESTS = [
    test_data_quality_proves_valid_and_rejects_bad,
    test_fast_math_and_metrics,
    test_risk_parity_invariants,
    test_stock_and_etf_normalization,
    test_recommendation_has_required_four_outputs,
    test_holding_advice_reconciles_cash_position_and_action,
]

if __name__ == "__main__":
    for test in TESTS:
        test()
        print(f"PASS {test.__name__}")
    print(f"ACCEPTANCE_PASS={len(TESTS)}")
