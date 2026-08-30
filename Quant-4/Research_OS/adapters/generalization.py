"""Research OS wrapper for Main.generalization_validation."""
from __future__ import annotations

from Research_OS.contracts.verification import GeneralizationEvidence


def generalization_evidence(experiment_id: str, result: dict) -> GeneralizationEvidence:
    context = result.get("market_context", {})
    frozen = result.get("frozen_logic", {})
    return GeneralizationEvidence(
        schema_version="generalization-evidence/v1", experiment_id=experiment_id,
        verdict=result.get("transfer_verdict", "TRANSFER_FAILED"),
        target_search_trials=int(context.get("target_search_trials", -1)),
        source_parameter_hash=str(frozen.get("parameter_hash", "")),
        target_parameter_hash=str(context.get("parameter_hash", "")),
        pit_enforced=bool(context.get("pit_enforced", False)),
        transfer_matrix={key: ("PASS" if value else "FAIL") for key, value in result.get("transfer_checks", {}).items()},
    )
