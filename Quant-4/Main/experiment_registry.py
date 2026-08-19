"""Append-only, hash-chained experiment and negative-result registry."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Iterable

RESULT_STATUSES = frozenset({"ACCEPT", "REJECT", "HOLD"})
FAILURE_CLASSES = frozenset({"NO_EFFECT", "UNSTABLE", "OVERFIT", "COST_FAILURE", "INSUFFICIENT_DATA", "OTHER"})


def _canonical(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class ExperimentRegistry:
    """Registry mutations append events; prior experiments are never overwritten."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def events(self) -> list[dict]:
        if not self.path.exists():
            return []
        rows = [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        previous = "GENESIS"
        for row in rows:
            event_hash = row.pop("event_hash", None)
            valid_hash = hashlib.sha256(_canonical(row).encode("utf-8")).hexdigest()
            if row.get("previous_hash") != previous or valid_hash != event_hash:
                raise ValueError("experiment registry hash chain is invalid")
            row["event_hash"] = event_hash
            previous = event_hash
        return rows

    def _append(self, event: dict) -> dict:
        prior = self.events()
        payload = {
            "schema_version": "experiment-registry/v1",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "previous_hash": prior[-1]["event_hash"] if prior else "GENESIS",
            **event,
        }
        payload["event_hash"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical(payload) + "\n")
        return payload

    def register(
        self, *, experiment_id: str, hypothesis: str, objective: str, variables: dict,
        data_version: str, code_version: str, parameter_version: str,
        parent_id: str | None = None, report_refs: Iterable[str] = (),
    ) -> dict:
        if not experiment_id.startswith("EXP-") or not all((hypothesis, objective, data_version, code_version, parameter_version)):
            raise ValueError("formal experiments require an EXP- id, hypothesis and version evidence")
        if not isinstance(variables, dict) or not variables:
            raise ValueError("changed or controlled variables are required")
        existing = {row.get("experiment_id") for row in self.events() if row.get("event_type") == "REGISTERED"}
        if experiment_id in existing:
            raise ValueError(f"experiment already registered: {experiment_id}")
        if parent_id and parent_id not in existing:
            raise ValueError(f"parent experiment is not registered: {parent_id}")
        return self._append({
            "event_type": "REGISTERED", "experiment_id": experiment_id, "parent_id": parent_id,
            "hypothesis": hypothesis, "objective": objective, "variables": variables,
            "data_version": data_version, "code_version": code_version,
            "parameter_version": parameter_version, "report_refs": list(report_refs),
        })

    def complete(
        self, *, experiment_id: str, status: str, evidence_gate: dict,
        result_summary: dict, failure_class: str | None = None,
        report_refs: Iterable[str] = (),
    ) -> dict:
        if status not in RESULT_STATUSES:
            raise ValueError(f"status must be one of {sorted(RESULT_STATUSES)}")
        events = self.events()
        registered = [row for row in events if row.get("experiment_id") == experiment_id and row.get("event_type") == "REGISTERED"]
        completed = [row for row in events if row.get("experiment_id") == experiment_id and row.get("event_type") == "COMPLETED"]
        if not registered or completed:
            raise ValueError("experiment must be registered exactly once before completion")
        if not isinstance(evidence_gate, dict) or "status" not in evidence_gate:
            raise ValueError("Research Evidence Gate result is required")
        if status == "ACCEPT" and evidence_gate.get("status") != "PASS":
            raise ValueError("an experiment cannot be ACCEPTed when its evidence gate did not PASS")
        if status == "REJECT" and failure_class not in FAILURE_CLASSES:
            raise ValueError(f"REJECT requires one of {sorted(FAILURE_CLASSES)}")
        return self._append({
            "event_type": "COMPLETED", "experiment_id": experiment_id, "status": status,
            "failure_class": failure_class, "evidence_gate": evidence_gate,
            "result_summary": result_summary, "report_refs": list(report_refs),
        })

    def snapshot(self) -> dict:
        events = self.events()
        registrations = {row["experiment_id"]: row for row in events if row.get("event_type") == "REGISTERED"}
        results = {row["experiment_id"]: row for row in events if row.get("event_type") == "COMPLETED"}
        return {
            "schema_version": "experiment-registry-snapshot/v1",
            "attempts": len(registrations), "completed": len(results),
            "open": sorted(set(registrations) - set(results)),
            "status_counts": {status: sum(row.get("status") == status for row in results.values()) for status in sorted(RESULT_STATUSES)},
            "failure_counts": {item: sum(row.get("failure_class") == item for row in results.values()) for item in sorted(FAILURE_CLASSES)},
            "negative_results": sorted(key for key, row in results.items() if row.get("status") == "REJECT"),
            "chain_head": events[-1]["event_hash"] if events else "GENESIS",
        }
