"""Pre-registered statistical validation for production candidates."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

VALIDATION_SPEC = {
    "version": "statistical-validation/v2", "registered_at": "2026-08-19",
    "frequency": 252, "sharpe_threshold": 0.50, "min_psr_probability": 0.95,
    "familywise_alpha": 0.05, "max_pbo": 0.20, "bootstrap_samples": 1000,
    "bootstrap_block_size": 20, "bootstrap_confidence": 0.95,
    "min_parameter_stability_ratio": 0.80, "min_oos_observations": 252,
    "min_oos_sharpe": 0.50, "seed": 20260819,
}
VALIDATION_SPEC_SHA256 = hashlib.sha256(
    json.dumps(VALIDATION_SPEC, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _clean(values: pd.Series | Sequence[float]) -> np.ndarray:
    series = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return series.to_numpy(dtype=float)


def _sharpe(values: np.ndarray) -> float:
    if len(values) < 2:
        return float("nan")
    std = float(np.std(values, ddof=1))
    return float(np.mean(values) / std * math.sqrt(VALIDATION_SPEC["frequency"])) if std > 0 else float("nan")


def probabilistic_sharpe_probability(values: Sequence[float], threshold: float | None = None) -> float | None:
    clean = _clean(values)
    if len(clean) < 3:
        return None
    sr = _sharpe(clean)
    daily_sr = sr / math.sqrt(VALIDATION_SPEC["frequency"])
    skew = float(pd.Series(clean).skew())
    kurt = float(pd.Series(clean).kurtosis()) + 3.0
    variance = 1.0 - skew * daily_sr + ((kurt - 1.0) / 4.0) * daily_sr**2
    if not np.isfinite(variance) or variance <= 0:
        return None
    target = VALIDATION_SPEC["sharpe_threshold"] if threshold is None else float(threshold)
    z_score = (sr - target) * math.sqrt(len(clean) - 1) / math.sqrt(variance * VALIDATION_SPEC["frequency"])
    probability = _normal_cdf(z_score)
    return float(probability) if np.isfinite(probability) else None


def holm_adjust(pvalues: Sequence[float]) -> list[float]:
    values = np.asarray(pvalues, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("pvalues must be finite probabilities")
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(values) - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def probability_of_backtest_overfitting(trial_returns: pd.DataFrame) -> float | None:
    numeric = trial_returns.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if numeric.shape[0] < 8 or numeric.shape[1] < 2:
        return None
    split_points = [numeric.shape[0] // 2]
    if numeric.shape[0] >= 16:
        split_points.extend([numeric.shape[0] // 4, numeric.shape[0] * 3 // 4])
    failures = 0
    for split in split_points:
        left, right = numeric.iloc[:split], numeric.iloc[split:]
        winner = left.mean().idxmax()
        failures += int(float(right.mean().rank(method="average", pct=True)[winner]) <= 0.5)
    return failures / len(split_points)


def circular_block_bootstrap_ci(values: Sequence[float]) -> dict | None:
    clean = _clean(values)
    if len(clean) < max(20, VALIDATION_SPEC["bootstrap_block_size"]):
        return None
    rng = np.random.default_rng(VALIDATION_SPEC["seed"])
    size, block = len(clean), min(VALIDATION_SPEC["bootstrap_block_size"], len(clean))
    estimates = []
    for _ in range(VALIDATION_SPEC["bootstrap_samples"]):
        starts = rng.integers(0, size, size=math.ceil(size / block))
        sample = np.concatenate([clean[(start + np.arange(block)) % size] for start in starts])[:size]
        estimates.append([float(np.mean(sample) * VALIDATION_SPEC["frequency"]), _sharpe(sample)])
    alpha = 1.0 - VALIDATION_SPEC["bootstrap_confidence"]
    array = np.asarray(estimates)
    return {
        "annual_mean": [float(x) for x in np.quantile(array[:, 0], [alpha / 2, 1 - alpha / 2])],
        "sharpe": [float(x) for x in np.quantile(array[:, 1], [alpha / 2, 1 - alpha / 2])],
    }


def validate_candidate(
    returns: pd.Series, *, oos_returns: pd.Series, trial_returns: pd.DataFrame,
    raw_pvalues: Sequence[float], parameter_neighbor_passes: Mapping[str, bool],
) -> dict:
    """Run every mandatory test and fail closed when evidence is unavailable."""
    clean, oos = _clean(returns), _clean(oos_returns)
    psr = probabilistic_sharpe_probability(clean)
    adjusted = holm_adjust(raw_pvalues) if raw_pvalues else []
    pbo = probability_of_backtest_overfitting(trial_returns)
    bootstrap = circular_block_bootstrap_ci(clean)
    stability = (sum(bool(value) for value in parameter_neighbor_passes.values()) / len(parameter_neighbor_passes)
                 if parameter_neighbor_passes else None)
    oos_sharpe = _sharpe(oos)
    checks = {
        "probabilistic_sharpe": psr is not None and psr >= VALIDATION_SPEC["min_psr_probability"],
        "multiple_testing": bool(adjusted) and min(adjusted) <= VALIDATION_SPEC["familywise_alpha"],
        "backtest_overfitting": pbo is not None and pbo <= VALIDATION_SPEC["max_pbo"],
        "bootstrap": bootstrap is not None and bootstrap["annual_mean"][0] > 0,
        "parameter_stability": stability is not None and stability >= VALIDATION_SPEC["min_parameter_stability_ratio"],
        "oos_observations": len(oos) >= VALIDATION_SPEC["min_oos_observations"],
        "oos_sharpe": np.isfinite(oos_sharpe) and oos_sharpe >= VALIDATION_SPEC["min_oos_sharpe"],
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not reasons, "status": "PASS" if not reasons else "HOLD",
        "action": "PRODUCTION_CANDIDATE" if not reasons else "OBSERVATION_ONLY",
        "spec": dict(VALIDATION_SPEC), "spec_sha256": VALIDATION_SPEC_SHA256,
        "checks": checks, "reasons": reasons,
        "metrics": {
            "observations": len(clean), "sharpe": _sharpe(clean), "psr_probability": psr,
            "raw_pvalues": list(raw_pvalues), "holm_adjusted_pvalues": adjusted,
            "num_trials": int(trial_returns.shape[1]), "pbo": pbo, "bootstrap_ci": bootstrap,
            "parameter_stability_ratio": stability, "parameter_neighbors": len(parameter_neighbor_passes),
            "oos_observations": len(oos), "oos_sharpe": oos_sharpe,
        },
    }
