"""N-version semantic comparison across independently produced implementations."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from Research_OS.contracts.composite import CrossImplementationComparison, IndependentImplementationManifest

LEVEL_KEYS = {
    "L1_SCHEMA": "schema", "L2_FACTOR_OUTPUT": "factors", "L3_POSITION_WEIGHT": "weights",
    "L4_TRADE_PATH": "trades", "L5_ACCOUNTING": "accounting", "L6_METRICS": "metrics",
}


@dataclass(frozen=True)
class IndependentImplementationProfile:
    require_distinct_agent: bool = True
    require_distinct_workspace: bool = True
    require_source_blindness: bool = True
    require_primary_metric_blindness: bool = True
    require_distinct_provider_family: bool = False
    require_distinct_model_family: bool = False


STRICT_INDEPENDENCE_PROFILE = IndependentImplementationProfile(
    require_distinct_provider_family=True, require_distinct_model_family=True,
)


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


class SchemaComparator:
    def compare(self, left: Any, right: Any, tolerance: float) -> tuple[bool, float, str]:
        passed = left == right
        return passed, 1.0 if passed else 0.0, "schema mismatch" if not passed else ""


class FactorOutputComparator:
    def compare(self, left: Any, right: Any, tolerance: float) -> tuple[bool, float, str]:
        passed, agreement = _numeric_agreement(left, right, tolerance)
        return passed, agreement, "factor asset/index/value divergence" if not passed else ""


class WeightComparator:
    def compare(self, left: Any, right: Any, tolerance: float) -> tuple[bool, float, str]:
        passed, agreement = _numeric_agreement(left, right, tolerance)
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            numeric_left = [float(value) for value in left.values() if isinstance(value, (int, float))]
            numeric_right = [float(value) for value in right.values() if isinstance(value, (int, float))]
            invariant = (all(math.isfinite(value) for value in numeric_left + numeric_right)
                         and abs(sum(numeric_left)) <= 1.0 + tolerance and abs(sum(numeric_right)) <= 1.0 + tolerance)
            passed = passed and invariant
        return passed, agreement if passed else 0.0, "weight alignment or exposure invariant failed" if not passed else ""


class TradePathComparator:
    REQUIRED = {"timestamp", "asset", "side", "quantity"}

    def compare(self, left: Any, right: Any, tolerance: float) -> tuple[bool, float, str]:
        if not isinstance(left, Sequence) or isinstance(left, (str, bytes)) or not isinstance(right, Sequence):
            return False, 0.0, "trade path must be a sequence"
        if any(not isinstance(row, Mapping) or not self.REQUIRED <= set(row) for row in list(left) + list(right)):
            return False, 0.0, "trade path schema is incomplete"
        passed, agreement = _numeric_agreement(left, right, tolerance)
        return passed, agreement, "trade timestamp/asset/side/order path diverged" if not passed else ""


class AccountingComparator:
    def compare(self, left: Any, right: Any, tolerance: float) -> tuple[bool, float, str]:
        passed, agreement = _numeric_agreement(left, right, tolerance)
        for value in (left, right):
            if isinstance(value, Mapping) and {"nav", "cash", "positions_value"} <= set(value):
                identity_error = abs(float(value["nav"]) - float(value["cash"]) - float(value["positions_value"]))
                passed = passed and identity_error <= tolerance
            if isinstance(value, Mapping) and any(float(value.get(key, 0.0)) < 0 for key in ("fees", "turnover")):
                passed = False
        return passed, agreement if passed else 0.0, "NAV/cash/fee/turnover accounting identity failed" if not passed else ""


class MetricComparator:
    def compare(self, left: Any, right: Any, tolerance: float) -> tuple[bool, float, str]:
        passed, agreement = _numeric_agreement(left, right, tolerance)
        return passed, agreement, "metric-specific tolerance exceeded" if not passed else ""


COMPARATORS: dict[str, Any] = {
    "L1_SCHEMA": SchemaComparator(), "L2_FACTOR_OUTPUT": FactorOutputComparator(),
    "L3_POSITION_WEIGHT": WeightComparator(), "L4_TRADE_PATH": TradePathComparator(),
    "L5_ACCOUNTING": AccountingComparator(), "L6_METRICS": MetricComparator(),
}


def compare_implementations(manifest_a: IndependentImplementationManifest,
                            manifest_b: IndependentImplementationManifest,
                            outputs_a: dict[str, Any], outputs_b: dict[str, Any], *,
                            tolerances: dict[str, float] | None = None,
                            critical: bool = True,
                            profile: IndependentImplementationProfile | None = None) -> CrossImplementationComparison:
    if manifest_a.experiment_id != manifest_b.experiment_id or manifest_a.spec_sha256 != manifest_b.spec_sha256:
        raise ValueError("implementations must use the same frozen experiment specification")
    selected = profile or IndependentImplementationProfile()
    independence_checks = {
        "distinct agent": not selected.require_distinct_agent or manifest_a.implementer_agent_id != manifest_b.implementer_agent_id,
        "distinct workspace": not selected.require_distinct_workspace or manifest_a.workspace_id != manifest_b.workspace_id,
        "source blindness": not selected.require_source_blindness or not manifest_b.source_implementation_visible,
        "primary-metric blindness": not selected.require_primary_metric_blindness or not manifest_b.primary_metrics_visible,
        "distinct provider family": not selected.require_distinct_provider_family or manifest_a.provider_family != manifest_b.provider_family,
        "distinct model family": not selected.require_distinct_model_family or manifest_a.model_family != manifest_b.model_family,
    }
    independent = all(independence_checks.values())
    limits = tolerances or {level: 1e-9 for level in LEVEL_KEYS}
    level_results, metrics, diagnostics = {}, {}, []
    for level, key in LEVEL_KEYS.items():
        if key not in outputs_a or key not in outputs_b:
            level_results[level] = False
            metrics[level] = 0.0
            diagnostics.append(f"{level}: missing {key} output")
            continue
        passed, agreement, detail = COMPARATORS[level].compare(outputs_a[key], outputs_b[key], limits.get(level, 0.0))
        level_results[level], metrics[level] = passed, agreement
        if not passed:
            diagnostics.append(f"{level}: {detail}")
    if not independent:
        status = "INDEPENDENT_IMPLEMENTATION_HOLD"
        diagnostics.append("independence not proven: " + ", ".join(name for name, passed in independence_checks.items() if not passed))
    elif manifest_a.code_sha256 == manifest_b.code_sha256:
        status = "INDEPENDENT_IMPLEMENTATION_HOLD"
        diagnostics.append("IDENTICAL_CODE_REVIEW_REQUIRED")
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
