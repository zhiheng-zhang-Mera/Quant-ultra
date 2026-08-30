"""Small auditable evidence graph; no graph database is required."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from Research_OS.contracts.common import sha256, stable_id
from Research_OS.contracts.evidence import EvidenceConflict, EvidenceRecord, SourceRecord


@dataclass(frozen=True)
class PITValidation:
    valid: bool
    reasons: tuple[str, ...]


def validate_pit(record: EvidenceRecord, *, use_time: datetime | None = None) -> PITValidation:
    if record.state != "OBSERVED":
        return PITValidation(False, (record.state,))
    reasons: list[str] = []
    if record.event_time and record.publish_time and record.event_time > record.publish_time:
        reasons.append("event_after_publication")
    if record.publish_time and record.availability_time and record.publish_time > record.availability_time:
        reasons.append("availability_before_publication")
    if record.availability_time and record.ingest_time and record.availability_time > record.ingest_time:
        reasons.append("ingest_before_availability")
    if use_time and record.availability_time and record.availability_time > use_time:
        reasons.append("not_available_at_use_time")
    return PITValidation(not reasons, tuple(reasons))


class EvidenceGraph:
    def __init__(self):
        self.sources: dict[str, SourceRecord] = {}
        self.evidence: dict[str, EvidenceRecord] = {}
        self.conflicts: dict[str, EvidenceConflict] = {}

    def add_source(self, source: SourceRecord) -> None:
        prior = self.sources.get(source.source_id)
        if prior and prior.content_sha256 != source.content_sha256:
            raise ValueError("source id cannot be reused for different content")
        self.sources[source.source_id] = source

    def add_evidence(self, record: EvidenceRecord) -> bool:
        if record.source_id not in self.sources:
            raise ValueError("evidence source is not registered")
        duplicate = any(item.content_sha256 == record.content_sha256 and item.entity == record.entity for item in self.evidence.values())
        if duplicate:
            return False
        self.evidence[record.evidence_id] = record
        return True

    def detect_conflicts(self) -> tuple[EvidenceConflict, ...]:
        groups: dict[tuple[str, str, str], list[EvidenceRecord]] = {}
        for item in self.evidence.values():
            groups.setdefault((item.entity, item.event_type, item.horizon), []).append(item)
        for key, records in groups.items():
            directions = {item.direction for item in records if item.direction != "NEUTRAL"}
            sources = {item.source_id for item in records}
            if len(directions) > 1 and len(sources) > 1:
                evidence_ids = tuple(sorted(item.evidence_id for item in records))
                conflict_id = stable_id("EVD", "conflict", key, evidence_ids)
                self.conflicts[conflict_id] = EvidenceConflict(
                    schema_version="evidence-conflict/v1", conflict_id=conflict_id,
                    evidence_ids=evidence_ids, conflict_type="DIRECTION",
                    description=f"sources disagree about {key}",
                )
        return tuple(self.conflicts[key] for key in sorted(self.conflicts))

    def source_diversity(self) -> float:
        if not self.evidence:
            return 0.0
        return len({item.source_id for item in self.evidence.values()}) / len(self.evidence)

    def agreement_metric(self) -> dict[str, float | bool]:
        directions = [item.direction for item in self.evidence.values() if item.direction != "NEUTRAL"]
        agreement = max((directions.count(value) for value in set(directions)), default=0) / max(len(directions), 1)
        return {"agreement": agreement, "informational_only": True, "truth_probability": False}

    @property
    def graph_sha256(self) -> str:
        return sha256({"sources": self.sources, "evidence": self.evidence, "conflicts": self.conflicts})
