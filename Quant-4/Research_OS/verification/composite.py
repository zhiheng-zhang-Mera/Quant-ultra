"""Independent reasoning, holdout, risk, transfer and policy integrity gates."""
from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from Research_OS.contracts.common import LifecycleStatus, sha256
from Research_OS.contracts.composite import (
    GeneralizationMatrix,
    HoldoutVaultManifest,
    PolicyManifest,
    ReasoningIndependenceEvidence,
    ResearchGeneration,
    RiskVerificationEvidence,
)
from Research_OS.registry.storage import HashChainStore


def assess_reasoning_independence(experiment_id: str, profiles: tuple[dict[str, Any], ...]) -> ReasoningIndependenceEvidence:
    def diversity(key: str) -> float:
        values = {str(profile.get(key, "")) for profile in profiles if profile.get(key)}
        return len(values) / max(1, len(profiles))

    def pairwise_overlap(key: str) -> float:
        if len(profiles) < 2:
            return 1.0
        sets = [set(re.findall(r"[\w.-]+", str(profile.get(key, "")).casefold())) for profile in profiles]
        values = [len(left & right) / max(1, len(left | right))
                  for index, left in enumerate(sets) for right in sets[index + 1:]]
        return sum(values) / len(values)

    firewalls = bool(profiles) and all(not profile.get("peer_conclusions_visible", True) for profile in profiles)
    minimum = len(profiles) >= 2 and min(diversity("provider_family"), diversity("context_hash"),
                                         diversity("evidence_set_hash")) >= 0.5
    return ReasoningIndependenceEvidence(
        schema_version=ReasoningIndependenceEvidence.SCHEMA, experiment_id=experiment_id, profiles=profiles,
        provider_diversity=diversity("provider_family"), model_family_diversity=diversity("model_family"),
        prompt_family_diversity=diversity("prompt_family"), context_diversity=diversity("context_hash"),
        evidence_set_diversity=diversity("evidence_set_hash"), conclusion_similarity=pairwise_overlap("conclusion"),
        failure_mode_overlap=pairwise_overlap("failure_codes"), information_firewalls_passed=firewalls,
        informational_only=True, status=LifecycleStatus.PASS if firewalls and minimum else LifecycleStatus.HOLD,
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

    def expose_generation(self, generation: ResearchGeneration, *, actor: str, purpose: str) -> ResearchGeneration:
        self.store.append("GENERATION_HOLDOUT_EXPOSED", {"family_id": generation.family_id,
                                                          "generation_id": generation.generation_id,
                                                          "purpose": purpose}, actor=actor)
        return replace(generation, exposure_count=generation.exposure_count + 1, clean_holdout=False)

    def child_generation(self, generation: ResearchGeneration, *, generation_id: str,
                         reason: str) -> ResearchGeneration:
        self.store.append("GENERATION_CREATED", {"family_id": generation.family_id,
                                                  "generation_id": generation_id,
                                                  "parent_generation_id": generation.generation_id,
                                                  "reason": reason}, actor="HoldoutVault")
        return ResearchGeneration(ResearchGeneration.SCHEMA, generation_id, generation.generation_id,
                                  generation.family_id, True, 0, reason)


@dataclass(frozen=True)
class RiskMetricDefinition:
    name: str
    better: str = "lower"

    def improvement(self, protected: float, unprotected: float) -> float:
        if self.better == "lower":
            return unprotected - protected
        if self.better == "higher":
            return protected - unprotected
        if self.better == "closer_to_zero":
            return abs(unprotected) - abs(protected)
        raise ValueError(f"unknown risk metric semantics: {self.better}")


@dataclass(frozen=True)
class RiskVerificationProfile:
    required_metrics: tuple[str, ...] = ("max_drawdown", "expected_shortfall", "turnover", "concentration")
    require_counterfactual: bool = True
    required_capacity_metrics: tuple[str, ...] = ("estimated_capacity",)
    require_stress: bool = True


def verify_risk_plane(experiment_id: str, protected: dict[str, float], unprotected: dict[str, float],
                      *, capacity_profile: dict[str, float], stress_passed: bool,
                      profile: RiskVerificationProfile | None = None,
                      metric_definitions: dict[str, RiskMetricDefinition] | None = None) -> RiskVerificationEvidence:
    selected = profile or RiskVerificationProfile()
    required = set(selected.required_metrics)
    missing = sorted(required - protected.keys())
    counterfactual_missing = sorted(required - unprotected.keys()) if selected.require_counterfactual else []
    capacity_missing = sorted(set(selected.required_capacity_metrics) - capacity_profile.keys())
    checks = {name: LifecycleStatus.PASS for name in required - set(missing)}
    checks.update({name: LifecycleStatus.HOLD for name in missing})
    checks["counterfactual"] = LifecycleStatus.PASS if not counterfactual_missing else LifecycleStatus.HOLD
    checks["capacity"] = LifecycleStatus.PASS if not capacity_missing else LifecycleStatus.HOLD
    checks["stress"] = LifecycleStatus.PASS if stress_passed or not selected.require_stress else LifecycleStatus.REJECT
    failures = tuple(missing + [f"counterfactual:{key}" for key in counterfactual_missing]
                     + [f"capacity:{key}" for key in capacity_missing]
                     + ([] if stress_passed or not selected.require_stress else ["stress"]))
    definitions = metric_definitions or {name: RiskMetricDefinition(name) for name in required}
    attribution: dict[str, float | str] = {}
    for key in protected.keys() & unprotected.keys():
        definition = definitions.get(key)
        attribution[key] = definition.improvement(protected[key], unprotected[key]) if definition else "UNKNOWN_SEMANTICS"
    return RiskVerificationEvidence(RiskVerificationEvidence.SCHEMA, experiment_id, checks, protected,
                                    unprotected, attribution, capacity_profile, failures,
                                    LifecycleStatus.PASS if not failures else LifecycleStatus.HOLD)


def build_generalization_matrix(experiment_id: str, profile: str, axes: dict[str, str], *,
                                frozen_parameter_hash: str, target_parameter_hashes: dict[str, str],
                                target_search_trials: dict[str, int], pit_metadata_complete: dict[str, bool],
                                evidence_refs: dict[str, tuple[str, ...]] | None = None) -> GeneralizationMatrix:
    required_by_profile = {
        "FACTOR": {"time", "sector", "regime"},
        "PORTFOLIO_STRATEGY": {"time", "liquidity", "capital", "execution", "regime"},
        "CROSS_MARKET": {"market", "vendor", "time"},
    }
    resolved_axes = dict(axes)
    for axis in required_by_profile.get(profile, set()):
        resolved_axes.setdefault(axis, "MISSING")
    for axis, target_hash in target_parameter_hashes.items():
        if target_hash != frozen_parameter_hash or target_search_trials.get(axis, 0) != 0:
            resolved_axes[axis] = "FAILED"
    for axis, complete in pit_metadata_complete.items():
        if not complete:
            resolved_axes[axis] = "FAILED"
    return GeneralizationMatrix(GeneralizationMatrix.SCHEMA, experiment_id, profile, resolved_axes,
                                evidence_refs or {}, frozen_parameter_hash, target_parameter_hashes,
                                target_search_trials, pit_metadata_complete)


def compare_policy(old: PolicyManifest, new: PolicyManifest) -> dict[str, Any]:
    return {"old_hash": old.policy_hash, "new_hash": new.policy_hash,
            "changed": old.policy_hash != new.policy_hash,
            "mandatory_added": sorted(set(new.mandatory_dimensions) - set(old.mandatory_dimensions)),
            "mandatory_removed": sorted(set(old.mandatory_dimensions) - set(new.mandatory_dimensions)),
            "thresholds_changed": sha256(old.threshold_profile) != sha256(new.threshold_profile)}
