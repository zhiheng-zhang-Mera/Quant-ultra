from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Main.generalization_validation import (
    FrozenResearchLogic, MarketValidationContext, classify_market_regimes,
    evaluate_regime_robustness, evaluate_temporal_windows, validate_external_market,
)


def _frozen():
    return FrozenResearchLogic("CN_A", "2024-01-01", "git:abc", "params:123", "2023-12-31", "factor/v3", "risk/v2", "execution/cn-v2")


def _market_context(**changes):
    values = dict(market="US", data_version="us-prices:sha256", parameter_hash="params:123",
                  validation_start="2024-01-02", validation_end="2025-04-01", calendar="XNYS",
                  currency="USD", transaction_cost_model="us-equity/v1",
                  trading_constraints="T+0,no-price-limit,round-lot-1", pit_enforced=True, target_search_trials=0)
    values.update(changes)
    return MarketValidationContext(**values)


def test_external_market_validation_freezes_logic_and_decomposes_transfer():
    rng = np.random.default_rng(19)
    dates = pd.bdate_range("2024-01-02", periods=300)
    factor = rng.normal(.0003, .004, len(dates))
    risk = rng.normal(.0001, .001, len(dates))
    cost = np.full(len(dates), .00005)
    frame = pd.DataFrame({"factor_return": factor, "risk_return": risk, "execution_cost": cost,
                          "strategy_return": factor + risk - cost,
                          "benchmark_return": rng.normal(.00015, .006, len(dates))}, index=dates)
    result = validate_external_market(frame, _market_context(), _frozen())
    assert result["status"] == "COMPLETE"
    assert result["market_context"]["target_search_trials"] == 0
    assert result["transfer_verdict"] in {"TRANSFER_SUPPORTED", "TRANSFER_MIXED", "TRANSFER_FAILED"}
    assert set(result["transfer_checks"]) == {"positive_sharpe", "return_not_worse", "sharpe_not_worse", "drawdown_not_worse"}
    assert result["decomposition"]["execution_cost_annual"] > 0
    assert len(result["evidence_sha256"]) == 64


def test_external_validation_rejects_target_tuning_or_parameter_drift():
    dates = pd.bdate_range("2024-01-02", periods=300)
    frame = pd.DataFrame(0.0, index=dates, columns=["factor_return", "risk_return", "execution_cost", "strategy_return", "benchmark_return"])
    with pytest.raises(ValueError, match="parameter search"):
        validate_external_market(frame, _market_context(target_search_trials=1), _frozen())
    with pytest.raises(ValueError, match="parameters differ"):
        validate_external_market(frame, _market_context(parameter_hash="params:tuned"), _frozen())


def _regime_returns():
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2020-01-01", periods=360)
    benchmark = np.concatenate([rng.normal(.003, .002, 120), rng.normal(-.003, .002, 120), rng.normal(0, .025, 120)])
    benchmark = pd.Series(benchmark, index=dates)
    strategy = benchmark * .55 + .0004
    factor = benchmark * .15 + .0002
    unprotected = benchmark * 1.2 + .0002
    liquidity = pd.Series(0.0, index=dates)
    liquidity.iloc[300:310] = -.40
    return strategy, benchmark, factor, unprotected, liquidity


def test_regime_audit_identifies_advantage_weakness_factors_and_risk_effect():
    strategy, benchmark, factor, unprotected, liquidity = _regime_returns()
    classified = classify_market_regimes(benchmark, liquidity)
    assert {"BULL", "BEAR", "HIGH_VOL", "LIQUIDITY_SHOCK"} <= set(classified["primary_regime"])
    result = evaluate_regime_robustness(
        strategy, benchmark, factor_returns={"core_factor": factor},
        unprotected_returns=unprotected, liquidity_change=liquidity,
    )
    assert result["status"] == "PASS", result["reasons"]
    assert result["best_regime"] and result["worst_regime"]
    assert "core_factor" in result["factor_by_regime"]
    assert result["risk_effect_by_regime"]


def test_temporal_windows_require_predefined_nonoverlapping_periods():
    strategy, *_ = _regime_returns()
    windows = {"early": ("2020-01-01", "2020-06-30"), "late": ("2020-07-01", "2021-05-18")}
    result = evaluate_temporal_windows(strategy, windows)
    assert result["status"] == "PASS"
    assert result["best_window"] != result["worst_window"]
    with pytest.raises(ValueError, match="non-overlapping"):
        evaluate_temporal_windows(strategy, {"a": ("2020-01-01", "2020-12-31"), "b": ("2020-06-01", "2021-01-01")})
