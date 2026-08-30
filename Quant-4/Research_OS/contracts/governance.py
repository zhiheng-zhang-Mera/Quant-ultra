"""Agent assessments and deterministic governance outputs."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

from .common import AdmissionAction, Contract, LifecycleStatus, utc_now


@dataclass(frozen=True)
class AgentAssessment(Contract):
    SCHEMA: ClassVar[str] = "agent-assessment/v1"
    agent_id: str = ""
    role: str = ""
    status: LifecycleStatus = LifecycleStatus.HOLD
    confidence: float = 0.0
    reasons: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceMatrix(Contract):
    SCHEMA: ClassVar[str] = "evidence-matrix/v1"
    dimensions: dict[str, LifecycleStatus] = field(default_factory=dict)
    critical_dimensions: tuple[str, ...] = ()
    mandatory_dimensions: tuple[str, ...] = ()
    evidence_refs: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class GovernanceDecision(Contract):
    SCHEMA: ClassVar[str] = "governance-decision/v1"
    experiment_id: str = ""
    action: AdmissionAction = AdmissionAction.HOLD_FOR_REVIEW
    reasons: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()
    dissenting_assessments: tuple[AgentAssessment, ...] = ()
    policy_version: str = "governance-policy/v1"
    decided_at: datetime = field(default_factory=utc_now)
    requires_human_authorization: bool = True


@dataclass(frozen=True)
class MemoryRecord(Contract):
    SCHEMA: ClassVar[str] = "memory-record/v1"
    memory_id: str = ""
    memory_type: str = "EPISODIC"
    experiment_id: str | None = None
    hypothesis_text: str = ""
    mechanism: str = ""
    features: tuple[str, ...] = ()
    parameters: dict[str, str] = field(default_factory=dict)
    universe: tuple[str, ...] = ()
    market: str = ""
    regimes: tuple[str, ...] = ()
    result: str = ""
    failure_reason: str = ""
    source_refs: tuple[str, ...] = ()
    parent_memory_id: str | None = None
