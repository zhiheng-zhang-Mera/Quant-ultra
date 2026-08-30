"""Preregistration, state history and multiple-testing budget accounting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from Research_OS.contracts.common import LockState
from Research_OS.contracts.experiment import ExperimentSpec

from .storage import HashChainStore

ALLOWED_TRANSITIONS = {
    LockState.DRAFT: {LockState.PREREGISTERED, LockState.INVALIDATED},
    LockState.PREREGISTERED: {LockState.LOCKED, LockState.INVALIDATED},
    LockState.LOCKED: {LockState.EXECUTING, LockState.INVALIDATED},
    LockState.EXECUTING: {LockState.COMPLETED, LockState.INVALIDATED},
    LockState.COMPLETED: set(),
    LockState.INVALIDATED: set(),
}


class ExperimentRegistry:
    def __init__(self, path: str | Path):
        self.store = HashChainStore(path, schema_version="research-experiment-registry/v1")

    def _registrations(self) -> dict[str, dict[str, Any]]:
        return {row["payload"]["experiment_id"]: row for row in self.store.events({"EXPERIMENT_REGISTERED"})}

    def register(self, spec: ExperimentSpec, *, actor: str) -> dict[str, Any]:
        registrations = self._registrations()
        if spec.experiment_id in registrations:
            raise ValueError("experiment is already registered")
        if spec.parent_experiment_id and spec.parent_experiment_id not in registrations:
            raise ValueError("parent experiment is not registered")
        family = [row["payload"] for row in registrations.values()
                  if row["payload"]["experiment_family_id"] == spec.experiment_family_id]
        if family:
            if any(row["max_trials"] != spec.max_trials for row in family):
                raise ValueError("experiment family trial budget is immutable")
            if len(family) >= spec.max_trials:
                raise ValueError("experiment trial budget is exhausted")
        return self.store.append("EXPERIMENT_REGISTERED", {
            "experiment_id": spec.experiment_id,
            "hypothesis_id": spec.hypothesis_id,
            "experiment_family_id": spec.experiment_family_id,
            "max_trials": spec.max_trials,
            "spec_sha256": spec.spec_sha256,
            "spec": spec.to_dict(),
            "parent_experiment_id": spec.parent_experiment_id,
        }, actor=actor)

    def state(self, experiment_id: str) -> LockState:
        if experiment_id not in self._registrations():
            raise KeyError(experiment_id)
        transitions = [row for row in self.store.events({"STATE_TRANSITION"}) if row["payload"]["experiment_id"] == experiment_id]
        return LockState(transitions[-1]["payload"]["new_state"]) if transitions else LockState.DRAFT

    def transition(self, experiment_id: str, new_state: LockState, *, actor: str, reason: str) -> dict[str, Any]:
        previous = self.state(experiment_id)
        if new_state not in ALLOWED_TRANSITIONS[previous]:
            raise ValueError(f"invalid experiment transition: {previous.value} -> {new_state.value}")
        if not reason.strip():
            raise ValueError("state transition reason is required")
        return self.store.append("STATE_TRANSITION", {
            "experiment_id": experiment_id, "previous_state": previous.value,
            "new_state": new_state.value, "reason": reason,
        }, actor=actor)

    def family_budget(self, family_id: str) -> dict[str, int]:
        registrations = [row["payload"] for row in self.store.events({"EXPERIMENT_REGISTERED"}) if row["payload"]["experiment_family_id"] == family_id]
        if not registrations:
            raise KeyError(family_id)
        budgets = {row["max_trials"] for row in registrations}
        if len(budgets) != 1:
            raise ValueError("experiment family contains inconsistent trial budgets")
        consumed = len(registrations)
        maximum = budgets.pop()
        return {"registered_trials": consumed, "consumed_trials": consumed,
                "remaining_trials": maximum - consumed, "max_trials": maximum}

    def assert_trial_available(self, family_id: str) -> None:
        if self.family_budget(family_id)["remaining_trials"] <= 0:
            raise ValueError("experiment trial budget is exhausted")

    def get(self, experiment_id: str) -> dict[str, Any]:
        registration = self._registrations().get(experiment_id)
        if registration is None:
            raise KeyError(experiment_id)
        return {"registration": registration, "state": self.state(experiment_id).value,
                "history": [row for row in self.store.read() if row["payload"].get("experiment_id") == experiment_id]}

    def list(self, *, state: LockState | None = None, tag: str | None = None) -> list[dict[str, Any]]:
        rows = [self.get(key) for key in sorted(self._registrations())]
        if state is not None:
            rows = [row for row in rows if row["state"] == state.value]
        if tag:
            rows = [row for row in rows if tag in row["registration"]["payload"]["spec"].get("tags", [])]
        return rows
