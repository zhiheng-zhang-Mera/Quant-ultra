"""The sole constructor of persisted-evidence composite matrices."""
from __future__ import annotations

from Research_OS.contracts.common import LifecycleStatus
from Research_OS.contracts.composite import VERIFICATION_LEVELS, CompositeVerificationMatrix

from .evidence_store import VerificationEvidenceStore


class VerificationMatrixBuilder:
    def __init__(self, evidence_store: VerificationEvidenceStore):
        self.evidence_store = evidence_store

    def build(self, run_id: str, *, experiment_id: str = "") -> CompositeVerificationMatrix:
        latest = self.evidence_store.latest_by_level(run_id)
        levels = {level: latest[level].status if level in latest else LifecycleStatus.HOLD
                  for level in VERIFICATION_LEVELS}
        refs = {level: ((latest[level].verification_evidence_id,) + latest[level].evidence_refs)
                if level in latest else () for level in VERIFICATION_LEVELS}
        critical = tuple(level for level in VERIFICATION_LEVELS if latest.get(level) is None or latest[level].critical)
        return CompositeVerificationMatrix(CompositeVerificationMatrix.SCHEMA, experiment_id or run_id,
                                           levels, critical, refs)
