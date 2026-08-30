"""Deterministic, fail-closed admission. Agent votes are deliberately ignored."""
from __future__ import annotations

from Research_OS.contracts.common import AdmissionAction, LifecycleStatus
from Research_OS.contracts.governance import AgentAssessment, EvidenceMatrix, GovernanceDecision


class ProductionInvariantError(ValueError):
    pass


DEFAULT_MANDATORY = ("mechanism", "source", "data", "implementation", "reproducibility", "statistics", "risk", "robustness", "generalization")
DEFAULT_CRITICAL = ("data", "implementation", "reproducibility")


class GovernancePolicy:
    version = "governance-policy/v1"

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
                                  dissenting_assessments=dissent, policy_version=self.version)

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
