from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Main.statistical_validation_v2 import VALIDATION_SPEC_SHA256, holm_adjust, validate_candidate


def _evidence(seed=819):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=1000)
    candidate = pd.Series(rng.normal(.0010, .004, len(index)), index=index)
    trials = pd.DataFrame({
        "candidate": rng.normal(.0010, .004, len(index)),
        "weak_1": rng.normal(.0000, .006, len(index)),
        "weak_2": rng.normal(-.0001, .006, len(index)),
    }, index=index)
    return candidate, trials


def test_statistical_validation_records_all_mandatory_evidence():
    returns, trials = _evidence()
    result = validate_candidate(
        returns, oos_returns=returns.iloc[-400:], trial_returns=trials,
        raw_pvalues=[.001, .20, .30], parameter_neighbor_passes={f"neighbor_{i}": i < 4 for i in range(5)},
    )
    assert result["status"] == "PASS"
    assert result["spec_sha256"] == VALIDATION_SPEC_SHA256
    assert all(result["checks"].values())
    assert result["metrics"]["num_trials"] == 3
    assert len(result["metrics"]["bootstrap_ci"]["sharpe"]) == 2


def test_missing_trials_neighbors_and_short_oos_hold_closed():
    returns, _ = _evidence()
    result = validate_candidate(
        returns, oos_returns=returns.iloc[-50:], trial_returns=pd.DataFrame(),
        raw_pvalues=[], parameter_neighbor_passes={},
    )
    assert result["status"] == "HOLD"
    assert result["action"] == "OBSERVATION_ONLY"
    assert {"multiple_testing", "backtest_overfitting", "parameter_stability", "oos_observations"} <= set(result["reasons"])


def test_holm_adjustment_is_monotone_in_sorted_order():
    raw = [.04, .01, .03]
    adjusted = holm_adjust(raw)
    ordered = [adjusted[i] for i in np.argsort(raw)]
    assert ordered == sorted(ordered)
    assert all(0 <= item <= 1 for item in adjusted)
