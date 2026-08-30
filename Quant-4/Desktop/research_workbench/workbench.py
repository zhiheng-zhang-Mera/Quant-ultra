"""QML-facing projection of application DTOs and persisted events only."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PySide6.QtCore import Property, QObject, QSettings, Signal, Slot

from Research_OS.application import ResearchApplicationService, ResearchExecutionMode

from .models import EventStreamModel, GovernanceGateModel, ResearchGraphModel, ResearchRunListModel, VerificationMatrixModel


class WorkbenchViewModel(QObject):
    activeRunChanged = Signal()
    commandFailed = Signal(str)
    commandCompleted = Signal(str)
    eventArrived = Signal(object)

    def __init__(self, state_dir: str | Path):
        super().__init__()
        self.service = ResearchApplicationService(state_dir)
        self.settings = QSettings("QuantUltra", "ResearchWorkbench")
        self.runs = ResearchRunListModel()
        self.graph = ResearchGraphModel()
        self.verification = VerificationMatrixModel()
        self.events = EventStreamModel()
        self.governance = GovernanceGateModel()
        self._active_run = ""
        self._execution_mode = self.service.execution_mode.value
        self._lifecycle_status = "NO_RUN"
        self._admission_action = "NONE"
        self._verification_passed = 0
        self._verification_holds = 12
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-command")
        self.eventArrived.connect(self._reduce_event)
        self.commandCompleted.connect(lambda _run_id: self.hydrate())
        self._unsubscribe = self.service.events.subscribe(self._on_event, replay_recent=True)
        self.hydrate()

    @Property(str, notify=activeRunChanged)
    def activeRun(self) -> str:  # noqa: N802 - Qt property
        return self._active_run

    @Property(str, notify=activeRunChanged)
    def executionMode(self) -> str:  # noqa: N802 - Qt property
        return self._execution_mode

    @Property(bool, notify=activeRunChanged)
    def demoMode(self) -> bool:  # noqa: N802 - Qt property
        return self._execution_mode in {ResearchExecutionMode.DEMO_OFFLINE.value,
                                        ResearchExecutionMode.UNKNOWN_LEGACY.value}

    @Property(str, notify=activeRunChanged)
    def lifecycleStatus(self) -> str:  # noqa: N802 - Qt property
        return self._lifecycle_status

    @Property(str, notify=activeRunChanged)
    def admissionAction(self) -> str:  # noqa: N802 - Qt property
        return self._admission_action

    @Property(int, notify=activeRunChanged)
    def verificationPassed(self) -> int:  # noqa: N802 - Qt property
        return self._verification_passed

    @Property(int, notify=activeRunChanged)
    def verificationHolds(self) -> int:  # noqa: N802 - Qt property
        return self._verification_holds

    @Property(bool, notify=activeRunChanged)
    def reducedMotion(self) -> bool:  # noqa: N802 - Qt property
        return bool(self.settings.value("reducedMotion", False, bool))

    @Slot()
    def hydrate(self) -> None:
        summaries = self.service.list_run_summaries()
        self.runs.replace([{"runId": item.run_id, "question": item.question, "status": item.lifecycle_status,
                            "mode": item.execution_mode} for item in summaries])
        preferred = str(self.settings.value("selectedRun", ""))
        available = {item.run_id for item in summaries}
        self._active_run = preferred if preferred in available else (summaries[-1].run_id if summaries else "")
        if self._active_run:
            summary = self.service.get_run_summary(self._active_run)
            self._execution_mode = summary.execution_mode
            self._lifecycle_status = summary.lifecycle_status
            self._admission_action = summary.admission_action or "NONE"
            self.settings.setValue("selectedRun", self._active_run)
        else:
            self._execution_mode = self.service.execution_mode.value
            self._lifecycle_status = "NO_RUN"
            self._admission_action = "NONE"
        self._hydrate_active_models()
        self.activeRunChanged.emit()

    @Slot(str, result=str)
    def createResearch(self, question: str) -> str:  # noqa: N802 - Qt slot
        try:
            run_id = self.service.create_research({"question": question})
            self.settings.setValue("selectedRun", run_id)
            self.hydrate()
            return run_id
        except Exception as exc:
            self.commandFailed.emit(str(exc))
            return ""

    @Slot(str)
    def startResearch(self, question: str) -> None:  # noqa: N802 - Qt slot
        run_id = self._active_run or self.createResearch(question)
        if not run_id:
            return
        future = self._executor.submit(self.service.start_research, run_id, {"question": question})
        future.add_done_callback(lambda completed: self.commandCompleted.emit(run_id)
                                 if completed.exception() is None else self.commandFailed.emit(str(completed.exception())))

    @Slot()
    def cancelActiveResearch(self) -> None:  # noqa: N802 - Qt slot
        if self._active_run:
            self.service.cancel_research(self._active_run)

    @Slot(bool)
    def setReducedMotion(self, enabled: bool) -> None:  # noqa: N802 - Qt slot
        self.settings.setValue("reducedMotion", enabled)
        self.activeRunChanged.emit()

    @Slot()
    def shutdown(self) -> None:
        self._unsubscribe()
        self.service.shutdown()
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.settings.sync()

    def _hydrate_active_models(self) -> None:
        if not self._active_run:
            topology, _edges = self.service.get_lifecycle_topology()
            nodes = topology
            verification = ()
            governance = None
        else:
            nodes = self.service.get_lifecycle_state(self._active_run)
            verification = self.service.get_verification_matrix(self._active_run)
            governance = self.service.get_governance_summary(self._active_run)
        self.graph.replace([{"stageId": item.stage_id, "name": item.name, "status": item.status,
                             "x": (index % 4) * 230, "y": (index // 4) * 105, "critical": item.critical}
                            for index, item in enumerate(nodes)])
        self.verification.replace([{"level": item.level, "status": item.status,
                                    "evidenceCount": len(item.evidence_ids)} for item in verification])
        self._verification_passed = sum(item.status == "PASS" for item in verification)
        self._verification_holds = sum(item.status != "PASS" for item in verification) if verification else 12
        self.governance.replace([] if governance is None else [
            {"dimension": "Admission", "status": governance.admission_action or "HOLD",
             "reason": "; ".join(governance.reasons), "critical": True},
            {"dimension": "Policy", "status": "BOUND", "reason": governance.policy_hash, "critical": True},
        ])

    def _on_event(self, event: Any) -> None:
        self.eventArrived.emit(event)

    @Slot(object)
    def _reduce_event(self, event: Any) -> None:
        self.events.append({"sequence": event.sequence, "eventType": event.event_type, "runId": event.run_id,
                            "summary": str(event.payload), "occurredAt": event.occurred_at.isoformat()})
        if event.run_id != self._active_run or not event.stage_id or event.substage_id:
            return
        status_by_event = {"STAGE_STARTED": "RUNNING", "STAGE_COMPLETED": "PASS", "STAGE_HELD": "HOLD",
                           "STAGE_FAILED": "FAILED", "STAGE_SKIPPED": "SKIPPED"}
        if event.event_type in status_by_event:
            self.graph.update_where("stageId", event.stage_id, {"status": status_by_event[event.event_type]})
