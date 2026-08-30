"""Fail-closed data quality and point-in-time gate."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

CRITICAL_CHECKS = (
    "coverage", "missingness", "duplicate_timestamps", "monotonic_time_index", "calendar_alignment",
    "corporate_actions", "survivorship", "publication_lag", "revision_contamination", "lookahead", "source_consistency",
)


@dataclass(frozen=True)
class DataGateResult:
    status: str
    action: str
    checks: dict[str, bool]
    reasons: tuple[str, ...]


def evaluate_data_gate(checks: Mapping[str, bool | None], *, research_limited: bool = False) -> DataGateResult:
    missing = tuple(key for key in CRITICAL_CHECKS if checks.get(key) is None)
    failed = tuple(key for key in CRITICAL_CHECKS if checks.get(key) is False)
    normalized = {key: bool(checks.get(key, False)) for key in CRITICAL_CHECKS}
    if failed:
        return DataGateResult("DATA_REJECT", "REJECTED", normalized, failed)
    if missing or research_limited:
        reasons = missing + (("research_limited",) if research_limited else ())
        return DataGateResult("DATA_HOLD", "RESEARCH_ONLY", normalized, reasons)
    return DataGateResult("DATA_PASS", "PROCEED", normalized, ())
