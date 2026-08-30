"""Multi-source, point-in-time evidence contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import ClassVar

from .common import Contract, ValidationError, utc_now, validate_stable_id


@dataclass(frozen=True)
class SourceRecord(Contract):
    SCHEMA: ClassVar[str] = "source-record/v1"
    source_id: str = ""
    name: str = ""
    source_type: str = ""
    version: str = ""
    reliability: float = 0.0
    content_sha256: str = ""
    license_or_usage_note: str = ""
    retrieved_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_stable_id(self.source_id, "SRC")
        if not self.name or not self.version or not 0 <= self.reliability <= 1:
            raise ValidationError("source metadata is incomplete")


@dataclass(frozen=True)
class EvidenceRecord(Contract):
    SCHEMA: ClassVar[str] = "evidence-record/v1"
    evidence_id: str = ""
    source_id: str = ""
    entity: str = ""
    event_type: str = ""
    claim: str = ""
    direction: str = "NEUTRAL"
    horizon: str = ""
    confidence: float = 0.0
    event_time: datetime | None = None
    publish_time: datetime | None = None
    availability_time: datetime | None = None
    ingest_time: datetime = field(default_factory=utc_now)
    content_sha256: str = ""
    state: str = "OBSERVED"

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_stable_id(self.evidence_id, "EVD")
        validate_stable_id(self.source_id, "SRC")
        if self.state not in {"OBSERVED", "DATA_MISSING", "NO_ELIGIBLE_EVENT"}:
            raise ValidationError("invalid evidence state")
        if not 0 <= self.confidence <= 1:
            raise ValidationError("confidence must be in [0, 1]")
        if self.state == "OBSERVED" and not all((self.entity, self.claim, self.publish_time, self.availability_time)):
            raise ValidationError("observed evidence is incomplete")


@dataclass(frozen=True)
class EvidenceConflict(Contract):
    SCHEMA: ClassVar[str] = "evidence-conflict/v1"
    conflict_id: str = ""
    evidence_ids: tuple[str, ...] = ()
    conflict_type: str = "SOURCE_DISAGREEMENT"
    description: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.conflict_type not in {"SEMANTIC", "TEMPORAL", "DIRECTION", "SOURCE_DISAGREEMENT"}:
            raise ValidationError("invalid conflict type")
        if len(set(self.evidence_ids)) < 2:
            raise ValidationError("conflict requires at least two evidence records")
