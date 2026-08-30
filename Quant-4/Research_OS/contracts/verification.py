"""Independent verification evidence contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

from .common import Contract, LifecycleStatus


@dataclass(frozen=True)
class VerificationRecord(Contract):
    SCHEMA: ClassVar[str] = "verification-record/v1"
    verification_level: str = "V1"
    status: LifecycleStatus = LifecycleStatus.HOLD
    critical: bool = True
    reasons: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    spec_version: str = ""


@dataclass(frozen=True)
class ReproductionManifest(Contract):
    SCHEMA: ClassVar[str] = "reproduction-manifest/v1"
    experiment_id: str = ""
    primary_run_id: str = ""
    reproduction_run_id: str = ""
    reproduction_agent_id: str = ""
    fresh_process: bool = False
    clean_cache: bool = False
    matching_hashes: bool = False
    metric_differences: dict[str, float] = field(default_factory=dict)
    tolerances: dict[str, float] = field(default_factory=dict)
    status: str = "REPRODUCTION_HOLD"


@dataclass(frozen=True)
class StatisticalEvidence(Contract):
    SCHEMA: ClassVar[str] = "statistical-evidence/v1"
    experiment_id: str = ""
    metrics: dict[str, float | None] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    registered_trials: int = 0
    multiple_testing_method: str = ""
    statistically_significant: bool = False
    economically_significant: bool = False
    missing_required: tuple[str, ...] = ()


@dataclass(frozen=True)
class RobustnessEvidence(Contract):
    SCHEMA: ClassVar[str] = "robustness-evidence/v1"
    experiment_id: str = ""
    attacks: dict[str, dict[str, Any]] = field(default_factory=dict)
    critical_failures: tuple[str, ...] = ()
    failure_mechanisms: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class GeneralizationEvidence(Contract):
    SCHEMA: ClassVar[str] = "generalization-evidence/v1"
    experiment_id: str = ""
    verdict: str = "TRANSFER_FAILED"
    target_search_trials: int = 0
    source_parameter_hash: str = ""
    target_parameter_hash: str = ""
    pit_enforced: bool = False
    transfer_matrix: dict[str, str] = field(default_factory=dict)
