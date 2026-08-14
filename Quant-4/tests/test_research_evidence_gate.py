"""Pre-registered OOS/risk/cost/multiple-testing gate tests."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Main.research_evidence_gate import GATE_SPEC, GATE_SPEC_SHA256, evaluate_research_gate


def _returns(seed=17):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=756)
    return pd.DataFrame({"strategy_return": rng.normal(0.00055, 0.0025, len(dates))}, index=dates)


def _summary(**overrides):
    result = {
        "max_drawdown": -0.05,
        "calmar": 1.50,
        "avg_rebalance_turnover": 0.20,
        "total_cost_fraction": 0.05,
    }
    result.update(overrides)
    return result


def test_strong_evidence_passes_pre_registered_gate():
    result = evaluate_research_gate(_returns(), _summary(), num_trials=5)
    assert result["passed"] is True
    assert result["status"] == "PASS"
    assert result["action"] == "RESEARCH_ELIGIBLE"
    assert result["spec"] == GATE_SPEC
    assert result["spec_sha256"] == GATE_SPEC_SHA256
    assert result["metrics"]["dsr_evidence_status"] == "CALCULATED"


def test_missing_trial_evidence_fails_closed_even_with_strong_returns():
    result = evaluate_research_gate(_returns(), _summary(), num_trials=None)
    assert result["status"] == "HOLD_FOR_REVIEW"
    assert result["action"] == "OBSERVATION_ONLY"
    assert {"num_trials", "dsr"}.issubset(result["reasons"])
    assert result["metrics"]["dsr_evidence_status"] == "MISSING_NUM_TRIALS"


def test_each_material_risk_metric_is_an_independent_veto():
    cases = {
        "max_drawdown": {"max_drawdown": -0.11},
        "calmar": {"calmar": 0.99},
        "turnover": {"avg_rebalance_turnover": 0.51},
        "cost": {"total_cost_fraction": 0.151},
    }
    for reason, overrides in cases.items():
        result = evaluate_research_gate(_returns(), _summary(**overrides), num_trials=5)
        assert result["passed"] is False
        assert reason in result["reasons"]


def test_short_oos_window_cannot_pass_on_good_point_estimates():
    short = _returns().iloc[:200]
    result = evaluate_research_gate(short, _summary(), num_trials=5)
    assert result["status"] == "HOLD_FOR_REVIEW"
    assert "oos_observations" in result["reasons"]
    assert "dsr" in result["reasons"]
