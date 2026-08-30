"""Independent reasoning, holdout, risk, transfer and policy integrity gates."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from Research_OS.contracts.common import LifecycleStatus, sha256
from Research_OS.contracts.composite import (
    GeneralizationMatrix,
    HoldoutVaultManifest,
    PolicyManifest,
    ReasoningIndependenceEvidence,
    RiskVerificationEvidence,
)
from Research_OS.registry.storage import HashChainStore


def assess_reasoning_independence(experiment_id: str, profiles: tuple[dict[str, Any], ...]) -> ReasoningIndependenceEvidence:
    def diversity(key: str) -> float:
        values = {str(profile.get(key, "")) for profile in profiles if profile.get(key)}
        return len(values) / max(1, len(profiles))

    firewalls = bool(profiles) and all(not profile.get("peer_conclusions_visible", True) for profile in profiles)
    minimum = len(profiles) >= 2 and min(diversity("provider_family"), diversity("context_hash"),
                                         diversity("evidence_set_hash")) >= 0.5
    return ReasoningIndependenceEvidence(
        ReasoningIndependenceEvidence.SCHEMA, experiment_id, profiles,
        diversity("provider_family"), diversity("model_family"), diversity("context_hash"),
        diversity("evidence_set_hash"), 0.0, 0.0, firewalls, True,
        LifecycleStatus.PASS if firewalls and minimum else LifecycleStatus.HOLD,
    )


class HoldoutVault:
    def __init__(self, path: str | Path):
        self.store = HashChainStore(path, schema_version="holdout-vault-events/v1")

    def expose(self, manifest: HoldoutVaultManifest, *, role: str, actor: str, purpose: str) -> HoldoutVaultManifest:
        if role not in manifest.authorized_roles:
            raise PermissionError(f"role {role!r} is not authorized for holdout access")
        if manifest.exposure_count >= manifest.exposure_budget:
            raise PermissionError("holdout exposure budget exhausted; create a new generation")
        row = self.store.append("HOLDOUT_EXPOSED", {"experiment_family_id": manifest.experiment_family_id,
                                                     "generation_id": manifest.generation_id, "role": role,
                                                     "purpose": purpose}, actor=actor)
        return replace(manifest, exposure_count=manifest.exposure_count + 1, clean_holdout=False,
                       exposure_event_ids=manifest.exposure_event_ids + (row["event_hash"],))


def verify_risk_plane(experiment_id: str, protected: dict[str, float], unprotected: dict[str, float],
                      *, capacity_profile: dict[str, float], stress_passed: bool) -> RiskVerificationEvidence:
    required = {"max_drawdown", "expected_shortfall", "turnover", "concentration"}
    missing = sorted(required - protected.keys())
    checks = {name: LifecycleStatus.PASS for name in required - set(missing)}
    checks.update({name: LifecycleStatus.HOLD for name in missing})
    checks["stress"] = LifecycleStatus.PASS if stress_passed else LifecycleStatus.REJECT
    failures = tuple(missing + ([] if stress_passed else ["stress"]))
    attribution: dict[str, float | str] = {
        key: protected[key] - unprotected[key] for key in protected.keys() & unprotected.keys()
    }
    return RiskVerificationEvidence(RiskVerificationEvidence.SCHEMA, experiment_id, checks, protected,
                                    unprotected, attribution, capacity_profile, failures,
                                    LifecycleStatus.PASS if not failures else LifecycleStatus.HOLD)


def build_generalization_matrix(experiment_id: str, profile: str, axes: dict[str, str], *,
                                frozen_parameter_hash: str, target_parameter_hashes: dict[str, str],
                                target_search_trials: dict[str, int], pit_metadata_complete: dict[str, bool],
                                evidence_refs: dict[str, tuple[str, ...]] | None = None) -> GeneralizationMatrix:
    for axis, target_hash in target_parameter_hashes.items():
        if target_hash != frozen_parameter_hash or target_search_trials.get(axis, 0) != 0:
            axes[axis] = "FAILED"
    for axis, complete in pit_metadata_complete.items():
        if not complete:
            axes[axis] = "FAILED"
    return GeneralizationMatrix(GeneralizationMatrix.SCHEMA, experiment_id, profile, dict(axes),
                                evidence_refs or {}, frozen_parameter_hash, target_parameter_hashes,
                                target_search_trials, pit_metadata_complete)


def compare_policy(old: PolicyManifest, new: PolicyManifest) -> dict[str, Any]:
    return {"old_hash": old.policy_hash, "new_hash": new.policy_hash,
            "changed": old.policy_hash != new.policy_hash,
            "mandatory_added": sorted(set(new.mandatory_dimensions) - set(old.mandatory_dimensions)),
            "mandatory_removed": sorted(set(old.mandatory_dimensions) - set(new.mandatory_dimensions)),
            "thresholds_changed": sha256(old.threshold_profile) != sha256(new.threshold_profile)}
