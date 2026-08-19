"""Fail-closed financial invariants shared by tests and research gates."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd


def validate_long_only_weights(weights: pd.DataFrame, *, max_gross: float = 1.0, tolerance: float = 1e-10) -> dict:
    numeric = weights.apply(pd.to_numeric, errors="coerce")
    failures: list[str] = []
    if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy()).all():
        failures.append("non_finite_weight")
    if (numeric < -tolerance).any().any():
        failures.append("short_position")
    gross = numeric.abs().sum(axis=1)
    if (gross > max_gross + tolerance).any():
        failures.append("gross_leverage")
    return {"passed": not failures, "failures": failures, "max_observed_gross": float(gross.max()) if len(gross) else 0.0}


def validate_nav_accounting(
    nav: pd.Series, gross_returns: pd.Series, costs: pd.Series, *, initial_nav: float = 1.0, tolerance: float = 1e-10,
) -> dict:
    frame = pd.concat({"nav": nav, "gross": gross_returns, "cost": costs}, axis=1)
    failures: list[str] = []
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        failures.append("non_finite_ledger")
    expected = (1.0 + frame["gross"] - frame["cost"]).cumprod() * float(initial_nav)
    error = float((frame["nav"] - expected).abs().max()) if len(frame) else 0.0
    if error > tolerance:
        failures.append("nav_identity")
    if (frame["cost"] < -tolerance).any():
        failures.append("negative_cost")
    return {"passed": not failures, "failures": failures, "max_absolute_error": error}


def validate_pit_prefix(before: Mapping[str, pd.Series], after: Mapping[str, pd.Series], *, cutoff: object) -> dict:
    failures = []
    cutoff_ts = pd.Timestamp(cutoff)
    for name in sorted(set(before) | set(after)):
        if name not in before or name not in after:
            failures.append(f"missing_feature:{name}")
            continue
        left, right = before[name].loc[:cutoff_ts], after[name].loc[:cutoff_ts]
        if not left.index.equals(right.index) or not np.allclose(left.to_numpy(), right.to_numpy(), equal_nan=True):
            failures.append(f"future_mutation_changed_prefix:{name}")
    return {"passed": not failures, "failures": failures, "cutoff": str(cutoff_ts)}
