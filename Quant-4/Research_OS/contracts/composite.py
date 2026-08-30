"""Composite verification v2 contracts.

These contracts describe independent evidence dimensions. They deliberately do
not expose an aggregate verification score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from .common import Contract, LifecycleStatus, ValidationError, utc_now

INDEPENDENT_IMPLEMENTATION_STATUSES = frozenset({
    "INDEPENDENTLY_CONFIRMED", "SEMANTICALLY_EQUIVALENT",
    "DIVERGENT_IMPLEMENTATION", "INDEPENDENT_IMPLEMENTATION_HOLD",
})
TRANSFER_AXIS_STATUSES = frozenset({"SUPPORTED", "MIXED", "FAILED", "NOT_REQUIRED", "MISSING"})
VERIFICATION_LEVELS = (
    "V1_SOURCE_PROVENANCE", "V2_SOURCE_INDEPENDENCE", "V3_TEMPORAL_PIT",
    "V4_IMPLEMENTATION_SAFETY", "V5_INDEPENDENT_IMPLEMENTATION",
    "V6_REPRODUCTION_ENVIRONMENT", "V7_STATISTICS", "V8_MECHANISM",
    "V9_RISK_CAPACITY", "V10_ROBUSTNESS", "V11_GENERALIZATION",
    "V12_GOVERNANCE_INTEGRITY",
)


@dataclass(frozen=True)
class SourceIndependenceEvidence(Contract):
    SCHEMA: ClassVar[str] = "source-independence-evidence/v1"
    experiment_id: str = ""
    raw_source_count: int = 0
    independent_origin_count: int = 0
    source_family_count: int = 0
    syndication_adjusted_diversity: float = 0.0
    origin_groups: dict[str, tuple[str, ...]] = field(default_factory=dict)
    near_duplicate_clusters: tuple[tuple[str, ...], ...] = ()
    reconciliation_findings: tuple[dict[str, Any], ...] = ()
    status: LifecycleStatus = LifecycleStatus.HOLD
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if min(self.raw_source_count, self.independent_origin_count, self.source_family_count) < 0:
            raise ValidationError("source counts cannot be negative")
        if self.independent_origin_count > self.raw_source_count:
            raise ValidationError("independent origins cannot exceed raw sources")
        if not 0 <= self.syndication_adjusted_diversity <= 1:
            raise ValidationError("adjusted diversity must be in [0, 1]")


@dataclass(frozen=True)
class IndependentImplementationManifest(Contract):
    SCHEMA: ClassVar[str] = "independent-implementation-manifest/v1"
    implementation_id: str = ""
    experiment_id: str = ""
    spec_sha256: str = ""
    implementer_agent_id: str = ""
    provider_family: str = ""
    model_family: str = ""
    workspace_id: str = ""
    workspace_sha256: str = ""
    code_sha256: str = ""
    source_implementation_visible: bool = True
    primary_metrics_visible: bool = True
    output_hashes: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not all((self.implementation_id, self.experiment_id, self.spec_sha256, self.implementer_agent_id,
                    self.workspace_id, self.workspace_sha256, self.code_sha256)):
            raise ValidationError("independent implementation manifest is incomplete")


@dataclass(frozen=True)
class CrossImplementationComparison(Contract):
    SCHEMA: ClassVar[str] = "cross-implementation-comparison/v1"
    experiment_id: str = ""
    implementation_a_id: str = ""
    implementation_b_id: str = ""
    level_results: dict[str, bool] = field(default_factory=dict)
    agreement_metrics: dict[str, float] = field(default_factory=dict)
    tolerances: dict[str, float] = field(default_factory=dict)
    divergence_diagnostics: tuple[str, ...] = ()
    status: str = "INDEPENDENT_IMPLEMENTATION_HOLD"
    critical: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.status not in INDEPENDENT_IMPLEMENTATION_STATUSES:
            raise ValidationError("invalid independent implementation status")


@dataclass(frozen=True)
class ExecutionEnvironmentManifest(Contract):
    SCHEMA: ClassVar[str] = "execution-environment-manifest/v1"
    workspace_id: str = ""
    workspace_path_hash: str = ""
    python_version: str = ""
    dependency_lock_hashes: dict[str, str] = field(default_factory=dict)
    os_name: str = ""
    architecture: str = ""
    timezone: str = ""
    locale: str = ""
    random_seeds: dict[str, int] = field(default_factory=dict)
    cpu: str = ""
    gpu: str = ""
    blas: str = ""
    git_sha: str = ""
    git_state: str = "UNKNOWN"
    environment_whitelist_hash: str = ""
    data_manifest_hash: str = ""
    experiment_spec_hash: str = ""
    shared_writable_cache: bool = True
    run_classification: str = "HOLD"

    @property
    def production_candidate_eligible(self) -> bool:
        return self.git_state == "CLEAN" and not self.shared_writable_cache and self.run_classification == "CLEAN_RESEARCH_RUN"


@dataclass(frozen=True)
class ReasoningIndependenceEvidence(Contract):
    SCHEMA: ClassVar[str] = "reasoning-independence-evidence/v1"
    experiment_id: str = ""
    profiles: tuple[dict[str, Any], ...] = ()
    provider_diversity: float = 0.0
    model_family_diversity: float = 0.0
    context_diversity: float = 0.0
    evidence_set_diversity: float = 0.0
    conclusion_similarity: float = 0.0
    failure_mode_overlap: float = 0.0
    information_firewalls_passed: bool = False
    informational_only: bool = True
    status: LifecycleStatus = LifecycleStatus.HOLD


@dataclass(frozen=True)
class RiskVerificationEvidence(Contract):
    SCHEMA: ClassVar[str] = "risk-verification-evidence/v1"
    experiment_id: str = ""
    checks: dict[str, LifecycleStatus] = field(default_factory=dict)
    protected_metrics: dict[str, float] = field(default_factory=dict)
    unprotected_metrics: dict[str, float] = field(default_factory=dict)
    counterfactual_attribution: dict[str, float | str] = field(default_factory=dict)
    capacity_profile: dict[str, float] = field(default_factory=dict)
    critical_failures: tuple[str, ...] = ()
    status: LifecycleStatus = LifecycleStatus.HOLD


@dataclass(frozen=True)
class HoldoutVaultManifest(Contract):
    SCHEMA: ClassVar[str] = "holdout-vault-manifest/v1"
    experiment_family_id: str = ""
    generation_id: str = ""
    parent_generation_id: str | None = None
    holdout_sha256: str = ""
    exposure_count: int = 0
    exposure_budget: int = 1
    clean_holdout: bool = True
    exposure_event_ids: tuple[str, ...] = ()
    authorized_roles: tuple[str, ...] = ("governance", "human_reviewer")

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.exposure_count < 0 or self.exposure_budget < 1:
            raise ValidationError("invalid holdout exposure accounting")
        if self.clean_holdout != (self.exposure_count == 0):
            raise ValidationError("clean_holdout must reflect exposure history")


@dataclass(frozen=True)
class PolicyManifest(Contract):
    SCHEMA: ClassVar[str] = "governance-policy-manifest/v1"
    policy_version: str = ""
    mandatory_dimensions: tuple[str, ...] = ()
    critical_dimensions: tuple[str, ...] = ()
    threshold_profile: dict[str, Any] = field(default_factory=dict)
    human_authorization_required: bool = True

    @property
    def policy_hash(self) -> str:
        return self.record_sha256


@dataclass(frozen=True)
class GeneralizationMatrix(Contract):
    SCHEMA: ClassVar[str] = "generalization-matrix/v1"
    experiment_id: str = ""
    profile: str = ""
    axes: dict[str, str] = field(default_factory=dict)
    evidence_refs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    frozen_parameter_hash: str = ""
    target_parameter_hashes: dict[str, str] = field(default_factory=dict)
    target_search_trials: dict[str, int] = field(default_factory=dict)
    pit_metadata_complete: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        invalid = set(self.axes.values()) - TRANSFER_AXIS_STATUSES
        if invalid:
            raise ValidationError(f"invalid transfer statuses: {sorted(invalid)}")


@dataclass(frozen=True)
class CompositeVerificationMatrix(Contract):
    SCHEMA: ClassVar[str] = "composite-verification-matrix/v1"
    experiment_id: str = ""
    levels: dict[str, LifecycleStatus] = field(default_factory=dict)
    critical_levels: tuple[str, ...] = ()
    evidence_refs: dict[str, tuple[str, ...]] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        super().__post_init__()
        unknown = set(self.levels) - set(VERIFICATION_LEVELS)
        if unknown:
            raise ValidationError(f"unknown verification levels: {sorted(unknown)}")

    @property
    def production_ready(self) -> bool:
        return bool(self.levels) and not any(
            self.levels.get(level, LifecycleStatus.HOLD) != LifecycleStatus.PASS
            for level in self.critical_levels
        )
