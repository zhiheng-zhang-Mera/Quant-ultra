"""Adapters for existing factor and preregistered statistical validators."""
from __future__ import annotations

from Research_OS.contracts.verification import StatisticalEvidence


def statistical_evidence_from_kernel(experiment_id: str, result: dict, *, registered_trials: int) -> StatisticalEvidence:
    metrics = dict(result.get("metrics", result))
    checks = result.get("checks", {})
    required = ("probabilistic_sharpe", "multiple_testing", "backtest_overfitting", "bootstrap", "oos_sharpe")
    missing = tuple(key for key in required if key not in checks)
    return StatisticalEvidence(
        schema_version="statistical-evidence/v1", experiment_id=experiment_id,
        metrics={key: value for key, value in metrics.items() if isinstance(value, (int, float)) or value is None},
        registered_trials=registered_trials, multiple_testing_method="Holm/DSR registered-trial accounting",
        statistically_significant=not missing and all(checks.get(key, False) for key in required),
        economically_significant=bool(result.get("passed", False)), missing_required=missing,
    )
