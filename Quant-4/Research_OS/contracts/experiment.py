"""Preregistered experiment and execution manifests."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from .common import Contract, LockState, ValidationError, sha256, utc_now, validate_stable_id


@dataclass(frozen=True)
class ExperimentDesign(Contract):
    SCHEMA: ClassVar[str] = "experiment-design/v1"
    universe: tuple[str, ...] = ()
    market: str = ""
    data_cutoff: str = ""
    train_window: tuple[str, str] = ("", "")
    validation_window: tuple[str, str] = ("", "")
    oos_window: tuple[str, str] = ("", "")
    walk_forward: bool = True
    embargo_days: int = 0
    benchmark: str = ""
    transaction_cost_model: str = ""
    primary_metrics: tuple[str, ...] = ()
    secondary_metrics: tuple[str, ...] = ()
    robustness_tests: tuple[str, ...] = ()
    subgroup_tests: tuple[str, ...] = ()
    acceptance_rules: dict[str, Any] = field(default_factory=dict)
    rejection_rules: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if not all((self.universe, self.market, self.data_cutoff, self.benchmark, self.transaction_cost_model, self.primary_metrics)):
            raise ValidationError("experiment design is incomplete")
        windows = [self.train_window, self.validation_window, self.oos_window]
        if any(not start or not end or start >= end for start, end in windows):
            raise ValidationError("experiment windows must be non-empty and ordered")
        if not (windows[0][1] < windows[1][0] < windows[1][1] < windows[2][0]):
            raise ValidationError("train, validation and OOS windows must not overlap")
        if self.embargo_days < 0 or not self.acceptance_rules or not self.rejection_rules:
            raise ValidationError("embargo and outcome rules must be preregistered")


@dataclass(frozen=True)
class ExperimentSpec(Contract):
    SCHEMA: ClassVar[str] = "experiment-spec/v1"
    experiment_id: str = ""
    hypothesis_id: str = ""
    experiment_family_id: str = ""
    max_trials: int = 1
    design: ExperimentDesign | None = None
    frozen_code_scope: tuple[str, ...] = ()
    frozen_data_scope: tuple[str, ...] = ()
    frozen_metric_definitions: dict[str, str] = field(default_factory=dict)
    parent_experiment_id: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_stable_id(self.experiment_id, "EXP")
        validate_stable_id(self.hypothesis_id, "HYP")
        if self.parent_experiment_id:
            validate_stable_id(self.parent_experiment_id, "EXP")
        if self.max_trials < 1 or self.design is None or not self.frozen_metric_definitions:
            raise ValidationError("experiment must define a positive trial budget and frozen design")

    @property
    def spec_sha256(self) -> str:
        return self.content_sha256


@dataclass(frozen=True)
class ExperimentLock(Contract):
    SCHEMA: ClassVar[str] = "experiment-lock/v1"
    experiment_id: str = ""
    state: LockState = LockState.DRAFT
    spec_sha256: str = ""
    locked_at: datetime | None = None
    actor: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_stable_id(self.experiment_id, "EXP")
        if self.state in {LockState.PREREGISTERED, LockState.LOCKED, LockState.EXECUTING, LockState.COMPLETED}:
            if not self.spec_sha256 or self.locked_at is None:
                raise ValidationError("locked experiment requires hash and timestamp")


@dataclass(frozen=True)
class DataManifest(Contract):
    SCHEMA: ClassVar[str] = "data-manifest/v1"
    experiment_id: str = ""
    data_cutoff: str = ""
    sources: tuple[str, ...] = ()
    source_versions: dict[str, str] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)
    pit_enforced: bool = False
    quality_checks: dict[str, bool] = field(default_factory=dict)
    retrieval_failures: tuple[str, ...] = ()

    @property
    def data_sha256(self) -> str:
        return self.content_sha256


@dataclass(frozen=True)
class ImplementationManifest(Contract):
    SCHEMA: ClassVar[str] = "implementation-manifest/v1"
    experiment_id: str = ""
    code_paths: tuple[str, ...] = ()
    code_sha256: str = ""
    dependencies: tuple[str, ...] = ()
    developer_agent_id: str = ""
    generated_tests: tuple[str, ...] = ()
    implemented_at: datetime = field(default_factory=utc_now)


@dataclass(frozen=True)
class KernelRunManifest(Contract):
    SCHEMA: ClassVar[str] = "kernel-run-manifest/v1"
    run_id: str = ""
    experiment_id: str = ""
    spec_sha256: str = ""
    code_sha256: str = ""
    data_sha256: str = ""
    git_sha: str = ""
    environment_sha256: str = ""
    report_hashes: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=utc_now)

    def fingerprint(self) -> str:
        return sha256({"spec": self.spec_sha256, "code": self.code_sha256, "data": self.data_sha256, "git": self.git_sha})
