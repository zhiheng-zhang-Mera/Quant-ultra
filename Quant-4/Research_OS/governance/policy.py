"""Deterministic, fail-closed admission. Agent votes are deliberately ignored."""
from __future__ import annotations

from enum import Enum

from Research_OS.contracts.common import AdmissionAction, LifecycleStatus
from Research_OS.contracts.composite import VERIFICATION_LEVELS, CompositeVerificationMatrix, PolicyManifest
from Research_OS.contracts.governance import AgentAssessment, EvidenceMatrix, GovernanceDecision


class ProductionInvariantError(ValueError):
    pass


DEFAULT_MANDATORY = ("mechanism", "source", "data", "implementation", "reproducibility", "statistics", "risk", "robustness", "generalization")
DEFAULT_CRITICAL = ("data", "implementation", "reproducibility")


class AdmissionProfile(str, Enum):
    LEGACY_RESEARCH = "LEGACY_RESEARCH"
    STANDARD_RESEARCH = "STANDARD_RESEARCH"
    CRITICAL_CANDIDATE = "CRITICAL_CANDIDATE"
    CROSS_MARKET_CANDIDATE = "CROSS_MARKET_CANDIDATE"


COMPOSITE_REQUIRED = {
    AdmissionProfile.STANDARD_RESEARCH: tuple(
        level for level in VERIFICATION_LEVELS if not level.startswith(("V9_", "V11_"))
    ),
    AdmissionProfile.CRITICAL_CANDIDATE: VERIFICATION_LEVELS,
    AdmissionProfile.CROSS_MARKET_CANDIDATE: VERIFICATION_LEVELS,
}


class GovernancePolicy:
    version = "governance-policy/v1"

    @property
    def manifest(self) -> PolicyManifest:
        return PolicyManifest(PolicyManifest.SCHEMA, self.version, DEFAULT_MANDATORY, DEFAULT_CRITICAL,
                              {"pass": LifecycleStatus.PASS.value, "fail_closed": True}, True)

    def decide(self, experiment_id: str, matrix: EvidenceMatrix,
               assessments: tuple[AgentAssessment, ...] = ()) -> GovernanceDecision:
        mandatory = matrix.mandatory_dimensions or DEFAULT_MANDATORY
        critical = matrix.critical_dimensions or DEFAULT_CRITICAL
        missing = tuple(key for key in mandatory if key not in matrix.dimensions)
        critical_failures = tuple(key for key in critical if matrix.dimensions.get(key) in {LifecycleStatus.REJECT, LifecycleStatus.FAILED})
        rejected = tuple(key for key, value in matrix.dimensions.items() if value in {LifecycleStatus.REJECT, LifecycleStatus.FAILED})
        holds = tuple(key for key in mandatory if matrix.dimensions.get(key) in {LifecycleStatus.HOLD, LifecycleStatus.PENDING, LifecycleStatus.RUNNING})
        if critical_failures or rejected:
            action = AdmissionAction.REJECTED
            reasons = tuple(f"deterministic gate failed: {key}" for key in sorted(set(critical_failures + rejected)))
        elif missing or holds:
            action = AdmissionAction.HOLD_FOR_REVIEW
            reasons = tuple(f"mandatory evidence unavailable: {key}" for key in sorted(set(missing + holds)))
        elif all(matrix.dimensions[key] == LifecycleStatus.PASS for key in mandatory):
            action = AdmissionAction.PRODUCTION_CANDIDATE
            reasons = ("all mandatory deterministic evidence gates passed", "human authorization is still required")
        else:
            action = AdmissionAction.RESEARCH_ONLY
            reasons = ("evidence supports research use only",)
        refs = tuple(sorted({ref for values in matrix.evidence_refs.values() for ref in values}))
        dissent = tuple(item for item in assessments if item.status != LifecycleStatus.PASS)
        return GovernanceDecision(schema_version="governance-decision/v1", experiment_id=experiment_id,
                                  action=action, reasons=reasons, evidence_refs=refs,
                                  dissenting_assessments=dissent, policy_version=self.version,
                                  policy_hash=self.manifest.policy_hash)

    def decide_composite(self, experiment_id: str, matrix: CompositeVerificationMatrix, *,
                         profile: AdmissionProfile = AdmissionProfile.CRITICAL_CANDIDATE,
                         agent_assessments: tuple[AgentAssessment, ...] = (),
                         execution_mode: str = "UNKNOWN_LEGACY") -> GovernanceDecision:
        required = COMPOSITE_REQUIRED.get(profile, ())
        if profile == AdmissionProfile.LEGACY_RESEARCH:
            raise ValueError("LEGACY_RESEARCH must use decide()")
        rejected = tuple(level for level in required if matrix.levels.get(level) in {
            LifecycleStatus.REJECT, LifecycleStatus.FAILED
        })
        unavailable = tuple(level for level in required if matrix.levels.get(level, LifecycleStatus.HOLD) in {
            LifecycleStatus.HOLD, LifecycleStatus.PENDING, LifecycleStatus.RUNNING, LifecycleStatus.SKIPPED
        })
        if rejected:
            action = AdmissionAction.REJECTED
            reasons = tuple(f"composite critical gate failed: {level}" for level in rejected)
        elif execution_mode == "DEMO_OFFLINE":
            action = AdmissionAction.RESEARCH_ONLY
            reasons = ("demo/reference execution cannot become a production candidate",)
        elif unavailable:
            action = AdmissionAction.HOLD_FOR_REVIEW
            reasons = tuple(f"composite evidence unavailable: {level}" for level in unavailable)
        else:
            action = AdmissionAction.PRODUCTION_CANDIDATE
            reasons = ("all profile-required composite gates passed", "human authorization is still required")
        evidence_ids = tuple(sorted({ref for level in required for ref in matrix.evidence_refs.get(level, ())}))
        dissent = tuple(item for item in agent_assessments if item.status != LifecycleStatus.PASS)
        manifest = PolicyManifest(PolicyManifest.SCHEMA, f"{self.version}/{profile.value}", required, required,
                                  {"profile": profile.value, "fail_closed": True}, True)
        return GovernanceDecision(GovernanceDecision.SCHEMA, experiment_id, action, reasons, evidence_ids,
                                  dissent, manifest.policy_version, manifest.policy_hash)

    def assert_candidate_invariants(self, profile: AdmissionProfile, matrix: CompositeVerificationMatrix,
                                    metadata: dict[str, object]) -> None:
        required = COMPOSITE_REQUIRED.get(profile, VERIFICATION_LEVELS)
        failures = [level for level in required if matrix.levels.get(level) != LifecycleStatus.PASS]
        checks = {"locked spec": bool(metadata.get("locked_spec")),
                  "valid holdout": bool(metadata.get("valid_holdout")),
                  "no target search": metadata.get("target_search_trials") == 0,
                  "frozen parameters": bool(metadata.get("parameter_hash_match")),
                  "production defaults unchanged": not bool(metadata.get("production_config_mutated")),
                  "AI has no production permission": not bool(metadata.get("actor_has_production_permission"))}
        failures.extend(name for name, passed in checks.items() if not passed)
        if failures:
            raise ProductionInvariantError("; ".join(failures))

    @staticmethod
    def assert_production_invariants(*, locked_spec: bool, pit_passed: bool, reproducible: bool,
                                     critical_findings: int, trial_count_known: bool,
                                     target_search_trials: int, parameter_hash_match: bool,
                                     production_config_mutated: bool, actor_has_production_permission: bool) -> None:
        checks = {
            "locked experiment": locked_spec,
            "PIT gate": pit_passed,
            "reproduction": reproducible,
            "critical findings": critical_findings == 0,
            "trial accounting": trial_count_known,
            "target-market tuning": target_search_trials == 0,
            "frozen parameter hash": parameter_hash_match,
            "production defaults unchanged": not production_config_mutated,
            "AI has no production permission": not actor_has_production_permission,
        }
        failures = [name for name, valid in checks.items() if not valid]
        if failures:
            raise ProductionInvariantError("; ".join(failures))
