"""N-version semantic comparison across independently produced implementations."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from Research_OS.contracts.composite import CrossImplementationComparison, IndependentImplementationManifest

LEVEL_KEYS = {
    "L1_SCHEMA": "schema", "L2_FACTOR_OUTPUT": "factors", "L3_POSITION_WEIGHT": "weights",
    "L4_TRADE_PATH": "trades", "L5_ACCOUNTING": "accounting", "L6_METRICS": "metrics",
}


def _numeric_agreement(left: Any, right: Any, tolerance: float) -> tuple[bool, float]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return False, 0.0
        results = [_numeric_agreement(left[key], right[key], tolerance) for key in sorted(left)]
    elif isinstance(left, Sequence) and isinstance(right, Sequence) and not isinstance(left, (str, bytes)):
        if len(left) != len(right):
            return False, 0.0
        results = [_numeric_agreement(a, b, tolerance) for a, b in zip(left, right, strict=True)]
    elif isinstance(left, (int, float)) and isinstance(right, (int, float)):
        difference = abs(float(left) - float(right))
        scale = max(abs(float(left)), abs(float(right)), 1.0)
        return math.isfinite(difference) and difference <= tolerance, max(0.0, 1.0 - difference / scale)
    else:
        return left == right, 1.0 if left == right else 0.0
    return (all(item[0] for item in results), sum(item[1] for item in results) / len(results)) if results else (True, 1.0)


def compare_implementations(manifest_a: IndependentImplementationManifest,
                            manifest_b: IndependentImplementationManifest,
                            outputs_a: dict[str, Any], outputs_b: dict[str, Any], *,
                            tolerances: dict[str, float] | None = None,
                            critical: bool = True) -> CrossImplementationComparison:
    if manifest_a.experiment_id != manifest_b.experiment_id or manifest_a.spec_sha256 != manifest_b.spec_sha256:
        raise ValueError("implementations must use the same frozen experiment specification")
    independent = all((manifest_a.implementer_agent_id != manifest_b.implementer_agent_id,
                       manifest_a.workspace_id != manifest_b.workspace_id,
                       manifest_a.code_sha256 != manifest_b.code_sha256,
                       not manifest_b.source_implementation_visible,
                       not manifest_b.primary_metrics_visible))
    limits = tolerances or {level: 1e-9 for level in LEVEL_KEYS}
    level_results, metrics, diagnostics = {}, {}, []
    for level, key in LEVEL_KEYS.items():
        if key not in outputs_a or key not in outputs_b:
            level_results[level] = False
            metrics[level] = 0.0
            diagnostics.append(f"{level}: missing {key} output")
            continue
        passed, agreement = _numeric_agreement(outputs_a[key], outputs_b[key], limits.get(level, 0.0))
        level_results[level], metrics[level] = passed, agreement
        if not passed:
            diagnostics.append(f"{level}: unexplained semantic divergence")
    if not independent:
        status = "INDEPENDENT_IMPLEMENTATION_HOLD"
        diagnostics.append("identity/workspace/code/information-firewall independence not proven")
    elif all(level_results.values()):
        status = "INDEPENDENTLY_CONFIRMED"
    else:
        status = "DIVERGENT_IMPLEMENTATION"
    return CrossImplementationComparison(
        schema_version="cross-implementation-comparison/v1", experiment_id=manifest_a.experiment_id,
        implementation_a_id=manifest_a.implementation_id, implementation_b_id=manifest_b.implementation_id,
        level_results=level_results, agreement_metrics=metrics, tolerances=limits,
        divergence_diagnostics=tuple(diagnostics), status=status, critical=critical,
    )
