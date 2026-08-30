"""Authoritative command/query boundary for Desktop, CLI and automation clients."""
from __future__ import annotations

from pathlib import Path
from threading import RLock
from typing import Any

from Research_OS.contracts.common import AdmissionAction, LifecycleStatus, stable_id
from Research_OS.governance import AdmissionProfile, GovernancePolicy
from Research_OS.orchestration import ResearchLifecycle
from Research_OS.orchestration.budget import ResearchBudget
from Research_OS.orchestration.research_dag import STAGE_NAMES
from Research_OS.verification import VerificationEvidenceStore, VerificationMatrixBuilder

from .dto import GovernanceDTO, LifecycleEdgeDTO, LifecycleNodeDTO, RunSummaryDTO, VerificationRowDTO
from .event_bus import PersistentEventBus
from .mode import DEMO_CAPABILITIES, ExecutionCapabilities, ResearchExecutionMode
from .run_store import ResearchRunStore


class ResearchApplicationService:
    def __init__(self, state_dir: str | Path, *, lifecycle: ResearchLifecycle | None = None,
                 capabilities: ExecutionCapabilities | None = None):
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lifecycle = lifecycle or ResearchLifecycle()
        self.capabilities = capabilities or DEMO_CAPABILITIES
        self.execution_mode = self.capabilities.mode
        self.run_store = ResearchRunStore(self.state_dir)
        self.events = PersistentEventBus(self.state_dir / "application-events.jsonl")
        self.verification_evidence = VerificationEvidenceStore(self.state_dir / "verification-evidence.jsonl")
        self.matrix_builder = VerificationMatrixBuilder(self.verification_evidence)
        self.governance = GovernancePolicy()
        self._cancelled: set[str] = set()
        self._lock = RLock()

    def create_research(self, intake: dict[str, Any], *, run_id: str | None = None) -> str:
        if not isinstance(intake, dict) or not str(intake.get("question", "")).strip():
            raise ValueError("intake.question is required")
        identity = run_id or stable_id("RUN", intake)
        self.run_store.create(identity, dict(intake), self.execution_mode)
        self._publish(identity, "RUN_CREATED", {"question": intake["question"],
                                                  "execution_mode": self.execution_mode.value})
        return identity

    def start_research(self, run_id: str, intake: dict[str, Any], *, budget: ResearchBudget | None = None) -> dict[str, Any]:
        with self._lock:
            state = self.run_store.get(run_id)
            if state["cancel_state"] in {"CANCEL_REQUESTED", "CANCELLED"}:
                raise ValueError("cancelled run cannot be started")
            self.run_store.update(run_id, lifecycle_status="RUNNING", inputs=dict(intake))
            self._publish(run_id, "RUN_STARTED", {"execution_mode": self.execution_mode.value})
            context = self.lifecycle.run(run_id, intake, budget=budget)
            results = self._serialize_results(context.results)
            failed = any(item.status in {LifecycleStatus.FAILED, LifecycleStatus.REJECT} for item in context.results.values())
            lifecycle_status = "FAILED" if failed else "COMPLETED"
            requested_action = str(context.results["R19"].output.get("action", "RESEARCH_ONLY")) \
                if "R19" in context.results else "RESEARCH_ONLY"
            admission_action = (AdmissionAction.RESEARCH_ONLY.value if self.execution_mode == ResearchExecutionMode.DEMO_OFFLINE
                                else requested_action)
            self.run_store.update(run_id, results=results, lifecycle_status=lifecycle_status,
                                  admission_action=admission_action, active_stage="")
            self._publish(run_id, "RUN_COMPLETED", {"lifecycle_status": lifecycle_status,
                                                     "admission_action": admission_action,
                                                     "execution_mode": self.execution_mode.value})
            return self.get_run(run_id)

    def resume_research(self, run_id: str, *, budget: ResearchBudget | None = None) -> dict[str, Any]:
        state = self.run_store.get(run_id)
        self._publish(run_id, "RUN_RESUMED", {})
        return self.start_research(run_id, dict(state["inputs"]), budget=budget)

    def cancel_research(self, run_id: str) -> None:
        self.run_store.get(run_id)
        self._cancelled.add(run_id)
        self.run_store.update(run_id, cancel_state="CANCEL_REQUESTED")
        self._publish(run_id, "CANCEL_REQUESTED", {})

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.run_store.get(run_id)

    def list_runs(self) -> list[dict[str, Any]]:
        return self.run_store.list()

    def list_run_summaries(self) -> list[RunSummaryDTO]:
        return [self._summary(state) for state in self.run_store.list()]

    def get_run_summary(self, run_id: str) -> RunSummaryDTO:
        return self._summary(self.run_store.get(run_id))

    def get_lifecycle_topology(self) -> tuple[tuple[LifecycleNodeDTO, ...], tuple[LifecycleEdgeDTO, ...]]:
        nodes = tuple(LifecycleNodeDTO(f"R{index}", name, "PENDING", index not in {1, 2, 20})
                      for index, name in enumerate(STAGE_NAMES))
        edges = tuple(LifecycleEdgeDTO(f"R{index - 1}", f"R{index}") for index in range(1, len(STAGE_NAMES)))
        return nodes, edges

    def get_lifecycle_state(self, run_id: str) -> tuple[LifecycleNodeDTO, ...]:
        state = self.run_store.get(run_id)
        return tuple(LifecycleNodeDTO(f"R{index}", name,
                                      state["results"].get(f"R{index}", {}).get("status", "PENDING"),
                                      index not in {1, 2, 20}) for index, name in enumerate(STAGE_NAMES))

    def get_verification_matrix(self, run_id: str) -> tuple[VerificationRowDTO, ...]:
        matrix = self.matrix_builder.build(run_id)
        return tuple(VerificationRowDTO(level, matrix.levels[level].value, matrix.evidence_refs[level])
                     for level in matrix.levels)

    def verification_matrix(self, run_id: str) -> list[dict[str, Any]]:
        return [{"level": row.level, "status": row.status, "evidence": list(row.evidence_ids)}
                for row in self.get_verification_matrix(run_id)]

    def get_governance_summary(self, run_id: str) -> GovernanceDTO:
        state = self.run_store.get(run_id)
        matrix = self.matrix_builder.build(run_id)
        decision = self.governance.decide_composite(run_id, matrix, profile=AdmissionProfile.CRITICAL_CANDIDATE,
                                                    execution_mode=state["execution_mode"])
        return GovernanceDTO(state.get("admission_action") or decision.action.value, decision.policy_hash,
                             decision.evidence_refs, decision.reasons)

    def _publish(self, run_id: str, event_type: str, payload: dict[str, Any], **metadata: Any) -> None:
        event = self.events.publish(run_id, event_type, payload, **metadata)
        self.run_store.update(run_id, last_event_sequence=event.sequence)

    @staticmethod
    def _serialize_results(results: dict[str, Any]) -> dict[str, Any]:
        return {key: {"stage_id": value.stage_id, "status": value.status.value, "output": value.output,
                      "evidence_refs": list(value.evidence_refs), "reasons": list(value.reasons)}
                for key, value in results.items()}

    @staticmethod
    def _summary(state: dict[str, Any]) -> RunSummaryDTO:
        return RunSummaryDTO(state["run_id"], str(state.get("inputs", {}).get("question", "")),
                             state.get("execution_mode", ResearchExecutionMode.UNKNOWN_LEGACY.value),
                             state.get("lifecycle_status", "UNKNOWN_LEGACY"), state.get("admission_action"),
                             state.get("cancel_state", "NONE"), int(state.get("last_event_sequence", 0)))
