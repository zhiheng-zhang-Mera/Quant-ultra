"""Immutable policy-bound composite governance decision history."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Research_OS.contracts.common import stable_id
from Research_OS.contracts.composite import CompositeVerificationMatrix
from Research_OS.contracts.governance import GovernanceDecision
from Research_OS.registry.storage import HashChainStore


@dataclass(frozen=True)
class GovernanceDecisionRecord:
    decision_id: str
    experiment_id: str
    run_id: str
    matrix_hash: str
    policy_hash: str
    decision: str
    evidence_ids: tuple[str, ...]
    reasons: tuple[str, ...]
    created_at: str
    supersedes_decision_id: str | None = None


class GovernanceDecisionStore:
    def __init__(self, path: str | Path):
        self.store = HashChainStore(path, schema_version="governance-decision-history/v1")

    def append(self, run_id: str, matrix: CompositeVerificationMatrix, decision: GovernanceDecision,
               *, supersedes_decision_id: str | None = None) -> GovernanceDecisionRecord:
        identity = stable_id("DEC", run_id, matrix.record_sha256, decision.policy_hash, supersedes_decision_id)
        payload = {"decision_id": identity, "experiment_id": decision.experiment_id, "run_id": run_id,
                   "matrix_hash": matrix.record_sha256, "policy_hash": decision.policy_hash,
                   "decision": decision.action.value, "evidence_ids": list(decision.evidence_refs),
                   "reasons": list(decision.reasons),
                   "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                   "supersedes_decision_id": supersedes_decision_id}
        self.store.append("GOVERNANCE_DECISION", payload, actor="GovernancePolicy")
        return self._decode(payload)

    @staticmethod
    def _decode(payload: dict[str, Any]) -> GovernanceDecisionRecord:
        data = dict(payload)
        data["evidence_ids"] = tuple(data.get("evidence_ids", ()))
        data["reasons"] = tuple(data.get("reasons", ()))
        return GovernanceDecisionRecord(**data)

    def list_by_run(self, run_id: str) -> list[GovernanceDecisionRecord]:
        return [self._decode(row["payload"]) for row in self.store.events(("GOVERNANCE_DECISION",))
                if row["payload"]["run_id"] == run_id]

    def latest(self, run_id: str) -> GovernanceDecisionRecord | None:
        records = self.list_by_run(run_id)
        return records[-1] if records else None
