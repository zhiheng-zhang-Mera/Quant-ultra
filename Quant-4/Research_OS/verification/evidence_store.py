"""Authoritative immutable store for composite verification evidence."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Research_OS.contracts.common import LifecycleStatus, sha256, stable_id
from Research_OS.contracts.composite import VERIFICATION_LEVELS
from Research_OS.registry.storage import HashChainStore


@dataclass(frozen=True)
class VerificationEvidenceRecord:
    verification_evidence_id: str
    level: str
    run_id: str
    experiment_id: str
    record_type: str
    record_hash: str
    critical: bool
    status: LifecycleStatus
    evidence_refs: tuple[str, ...]
    producer: str
    created_at: str
    supersedes_evidence_id: str | None = None


class VerificationEvidenceStore:
    def __init__(self, path: str | Path):
        self.store = HashChainStore(path, schema_version="verification-evidence-store/v1")

    def append(self, *, level: str, run_id: str, experiment_id: str, record_type: str,
               record: Any, status: LifecycleStatus, critical: bool = True,
               evidence_refs: tuple[str, ...] = (), producer: str,
               supersedes_evidence_id: str | None = None) -> VerificationEvidenceRecord:
        if level not in VERIFICATION_LEVELS:
            raise ValueError(f"unknown verification level: {level}")
        record_hash = sha256(record)
        identity = stable_id("VEV", run_id, level, record_hash, supersedes_evidence_id)
        payload = {"verification_evidence_id": identity, "level": level, "run_id": run_id,
                   "experiment_id": experiment_id, "record_type": record_type, "record_hash": record_hash,
                   "critical": critical, "status": status.value, "evidence_refs": list(evidence_refs),
                   "producer": producer, "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                   "supersedes_evidence_id": supersedes_evidence_id}
        self.store.append("VERIFICATION_EVIDENCE", payload, actor=producer)
        return self._decode(payload)

    @staticmethod
    def _decode(payload: dict[str, Any]) -> VerificationEvidenceRecord:
        data = dict(payload)
        data["status"] = LifecycleStatus(data["status"])
        data["evidence_refs"] = tuple(data.get("evidence_refs", ()))
        return VerificationEvidenceRecord(**data)

    def list_by_run(self, run_id: str) -> list[VerificationEvidenceRecord]:
        return [self._decode(row["payload"]) for row in self.store.events(("VERIFICATION_EVIDENCE",))
                if row["payload"]["run_id"] == run_id]

    def list_by_experiment(self, experiment_id: str) -> list[VerificationEvidenceRecord]:
        return [record for row in self.store.events(("VERIFICATION_EVIDENCE",))
                if (record := self._decode(row["payload"])).experiment_id == experiment_id]

    def get(self, evidence_id: str) -> VerificationEvidenceRecord:
        matches = [record for row in self.store.events(("VERIFICATION_EVIDENCE",))
                   if (record := self._decode(row["payload"])).verification_evidence_id == evidence_id]
        if not matches:
            raise KeyError(evidence_id)
        return matches[-1]

    def latest_by_level(self, run_id: str) -> dict[str, VerificationEvidenceRecord]:
        latest: dict[str, VerificationEvidenceRecord] = {}
        for record in self.list_by_run(run_id):
            latest[record.level] = record
        return latest

    def history_by_level(self, run_id: str, level: str) -> list[VerificationEvidenceRecord]:
        return [record for record in self.list_by_run(run_id) if record.level == level]
