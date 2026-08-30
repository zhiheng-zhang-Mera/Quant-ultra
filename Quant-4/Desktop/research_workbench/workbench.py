"""QML-facing view model. No registry, DAG or config object crosses this boundary."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PySide6.QtCore import Property, QObject, QSettings, Signal, Slot

from Research_OS.application import ResearchApplicationService
from Research_OS.orchestration.research_dag import STAGE_NAMES

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
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="research-command")
        self.eventArrived.connect(self._append_event)
        self.graph.replace([{"stageId": f"R{index}", "name": name, "status": "PENDING",
                             "x": (index % 4) * 230, "y": (index // 4) * 105,
                             "critical": index not in {1, 2, 20}} for index, name in enumerate(STAGE_NAMES)])
        self.service.events.subscribe(self._on_event, replay_recent=True)

    @Property(str, notify=activeRunChanged)
    def activeRun(self) -> str:  # noqa: N802 - Qt property
        return self._active_run

    @Property(bool, constant=True)
    def demoMode(self) -> bool:  # noqa: N802 - Qt property
        return False

    @Property(bool, notify=activeRunChanged)
    def reducedMotion(self) -> bool:  # noqa: N802 - Qt property
        return bool(self.settings.value("reducedMotion", False, bool))

    @Slot(str, result=str)
    def createResearch(self, question: str) -> str:  # noqa: N802 - Qt slot
        try:
            run_id = self.service.create_research({"question": question})
            self._active_run = run_id
            self.runs.append({"runId": run_id, "question": question, "status": "CREATED", "mode": "REAL"})
            self.verification.replace([{"level": f"V{index}", "status": "HOLD", "evidenceCount": 0}
                                       for index in range(1, 13)])
            self.activeRunChanged.emit()
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

    @Slot(bool)
    def setReducedMotion(self, enabled: bool) -> None:  # noqa: N802 - Qt slot
        self.settings.setValue("reducedMotion", enabled)
        self.activeRunChanged.emit()

    def _on_event(self, event: Any) -> None:
        self.eventArrived.emit(event)

    @Slot(object)
    def _append_event(self, event: Any) -> None:
        self.events.append({"sequence": event.sequence, "eventType": event.event_type, "runId": event.run_id,
                            "summary": str(event.payload), "occurredAt": event.occurred_at.isoformat()})
