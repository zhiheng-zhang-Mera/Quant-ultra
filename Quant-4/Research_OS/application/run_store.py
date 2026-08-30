"""Atomic v2 run-state persistence with explicit legacy migration."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

from Research_OS.contracts.common import canonical_json

from .mode import ResearchExecutionMode


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


class RunStateCorruptionError(ValueError):
    pass


class ResearchRunStore:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()

    def path_for(self, run_id: str) -> Path:
        if not run_id.startswith("RUN-") or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in run_id):
            raise ValueError("invalid run id")
        path = (self.directory / f"{run_id}.json").resolve()
        if path.parent != self.directory:
            raise ValueError("unsafe run path")
        return path

    def create(self, run_id: str, inputs: dict[str, Any], mode: ResearchExecutionMode) -> dict[str, Any]:
        with self._lock:
            if self.path_for(run_id).exists():
                raise ValueError(f"run already exists: {run_id}")
            timestamp = _now()
            state = {"schema_version": "research-run-state/v2", "run_id": run_id,
                     "execution_mode": mode.value, "lifecycle_status": "CREATED", "admission_action": None,
                     "inputs": inputs, "results": {}, "subresults": {}, "created_at": timestamp,
                     "updated_at": timestamp, "last_event_sequence": 0, "cancel_state": "NONE",
                     "active_stage": "", "active_substage": ""}
            self.write(state)
            return state

    def write(self, state: dict[str, Any]) -> None:
        with self._lock:
            payload = dict(state)
            payload["schema_version"] = "research-run-state/v2"
            payload["updated_at"] = _now()
            target = self.path_for(str(payload["run_id"]))
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(canonical_json(payload), encoding="utf-8")
            temporary.replace(target)

    def update(self, run_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            state = self.get(run_id)
            state.update(changes)
            self.write(state)
            return state

    def get(self, run_id: str) -> dict[str, Any]:
        path = self.path_for(run_id)
        if not path.exists():
            raise FileNotFoundError(run_id)
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError) as exc:
            raise RunStateCorruptionError(f"corrupted run state {run_id}: {exc}") from exc
        if state.get("run_id") != run_id or state.get("schema_version") not in {
            "research-run-state/v1", "research-run-state/v2"
        }:
            raise RunStateCorruptionError(f"invalid run state identity/schema: {run_id}")
        if state["schema_version"] == "research-run-state/v1":
            state = self._migrate_v1(state)
        return state

    def list(self) -> list[dict[str, Any]]:
        states = []
        for path in sorted(self.directory.glob("RUN-*.json")):
            states.append(self.get(path.stem))
        return states

    @staticmethod
    def _migrate_v1(state: dict[str, Any]) -> dict[str, Any]:
        migrated = dict(state)
        migrated.update({"schema_version": "research-run-state/v2",
                         "execution_mode": ResearchExecutionMode.UNKNOWN_LEGACY.value,
                         "lifecycle_status": "CANCELLED" if state.get("cancelled") else "UNKNOWN_LEGACY",
                         "admission_action": None, "subresults": {}, "created_at": state.get("updated_at", _now()),
                         "last_event_sequence": 0, "cancel_state": "CANCELLED" if state.get("cancelled") else "NONE",
                         "active_stage": "", "active_substage": ""})
        return migrated
