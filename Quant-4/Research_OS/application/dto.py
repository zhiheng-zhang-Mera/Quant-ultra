"""Versioned, dependency-light application read DTOs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class RunSummaryDTO:
    run_id: str
    question: str
    execution_mode: str
    lifecycle_status: str
    admission_action: str | None
    cancel_state: str
    last_event_sequence: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LifecycleNodeDTO:
    stage_id: str
    name: str
    status: str
    critical: bool


@dataclass(frozen=True)
class LifecycleEdgeDTO:
    source_stage_id: str
    target_stage_id: str


@dataclass(frozen=True)
class VerificationRowDTO:
    level: str
    status: str
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GovernanceDTO:
    admission_action: str | None
    policy_hash: str
    evidence_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceNodeDTO:
    evidence_id: str
    level: str
    status: str
    record_type: str


@dataclass(frozen=True)
class ProviderHealthDTO:
    provider_id: str
    status: str
    capabilities: tuple[str, ...] = ()
    detail: dict[str, Any] = field(default_factory=dict)
