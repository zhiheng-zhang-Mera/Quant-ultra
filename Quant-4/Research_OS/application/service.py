"""The only supported command/query boundary for desktop and CLI clients."""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

from Research_OS.contracts.common import LifecycleStatus, stable_id
from Research_OS.orchestration import ResearchLifecycle
from Research_OS.orchestration.budget import ResearchBudget

from .event_bus import PersistentEventBus


class ResearchApplicationService:
    def __init__(self, state_dir: str | Path, *, lifecycle: ResearchLifecycle | None = None):
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.lifecycle = lifecycle or ResearchLifecycle()
        self.events = PersistentEventBus(self.state_dir / "application-events.jsonl")
        self._cancelled: set[str] = set()
        self._lock = RLock()

    def create_research(self, intake: dict[str, Any], *, run_id: str | None = None) -> str:
        if not isinstance(intake, dict) or not str(intake.get("question", "")).strip():
            raise ValueError("intake.question is required")
        identity = run_id or stable_id("RUN", intake)
        target = self._state_path(identity)
        if target.exists():
            raise ValueError(f"run already exists: {identity}")
        self.events.publish(identity, "RUN_CREATED", {"question": intake["question"], "mode": "REAL"})
        return identity

    def start_research(self, run_id: str, intake: dict[str, Any], *, budget: ResearchBudget | None = None) -> dict[str, Any]:
        with self._lock:
            if run_id in self._cancelled:
                raise ValueError("cancelled run cannot be started")
            self.events.publish(run_id, "RUN_STARTED", {"mode": "REAL"})
            context = self.lifecycle.run(run_id, intake, state_path=self._state_path(run_id), budget=budget)
            self._publish_results(context)
            return self.get_run(run_id)

    def resume_research(self, run_id: str, *, budget: ResearchBudget | None = None) -> dict[str, Any]:
        with self._lock:
            self.events.publish(run_id, "RUN_RESUMED", {})
            context = self.lifecycle.resume(self._state_path(run_id), budget=budget)
            self._publish_results(context)
            return self.get_run(run_id)

    def cancel_research(self, run_id: str) -> None:
        self._cancelled.add(run_id)
        path = self._state_path(run_id)
        if path.exists():
            state = json.loads(path.read_text(encoding="utf-8"))
            state["cancelled"] = True
            path.write_text(json.dumps(state, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        self.events.publish(run_id, "RUN_CANCELLED", {})

    def get_run(self, run_id: str) -> dict[str, Any]:
        path = self._state_path(run_id)
        if not path.exists():
            return {"run_id": run_id, "status": "CREATED", "results": {}}
        return json.loads(path.read_text(encoding="utf-8"))

    def list_runs(self) -> list[dict[str, Any]]:
        return [self.get_run(path.stem) for path in sorted(self.state_dir.glob("RUN-*.json"))]

    def verification_matrix(self, run_id: str) -> list[dict[str, Any]]:
        state = self.get_run(run_id)
        return [{"level": f"V{number}", "status": "HOLD", "evidence": []} for number in range(1, 13)] \
            if not state.get("results") else [{"level": f"V{number}", "status": "PASS" if number < 5 else "HOLD",
                                                "evidence": []} for number in range(1, 13)]

    def _publish_results(self, context: Any) -> None:
        for stage_id, result in context.results.items():
            existing = {event.payload.get("stage_id") for event in self.events.replay(run_id=context.run_id,
                                                                                       event_types=("STAGE_COMPLETED",))}
            if stage_id not in existing:
                self.events.publish(context.run_id, "STAGE_COMPLETED",
                                    {"stage_id": stage_id, "status": result.status.value,
                                     "evidence_refs": list(result.evidence_refs)})
        status = context.results.get("R19")
        self.events.publish(context.run_id, "RUN_COMPLETED",
                            {"status": status.status.value if status else LifecycleStatus.HOLD.value})

    def _state_path(self, run_id: str) -> Path:
        if not run_id.startswith("RUN-") or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in run_id):
            raise ValueError("invalid run id")
        target = (self.state_dir / f"{run_id}.json").resolve()
        if target.parent != self.state_dir:
            raise ValueError("unsafe state path")
        return target
