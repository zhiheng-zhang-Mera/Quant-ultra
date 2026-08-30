"""Conservative importer: unknown historical fields are never fabricated."""
from __future__ import annotations

from Research_OS.contracts.common import stable_id
from Research_OS.contracts.governance import MemoryRecord
from Research_OS.memory import ResearchMemory

UNKNOWN_VALUES = frozenset({"unknown", "not_recorded", "not_applicable"})


class HistoricalImporter:
    def __init__(self, memory: ResearchMemory):
        self.memory = memory

    def import_record(self, payload: dict, *, actor: str) -> dict:
        required = ("title", "mechanism", "result", "source_path")
        if any(key not in payload for key in required):
            raise ValueError("historical import lacks required provenance")
        outcome = str(payload["result"])
        memory_type = "FAILURE" if outcome.upper() in {"REJECT", "FAILED", "NEGATIVE"} else "EPISODIC"
        record = MemoryRecord(
            schema_version="memory-record/v1", memory_id=stable_id("MEM", "historical", payload["title"], payload["source_path"]),
            memory_type=memory_type, hypothesis_text=str(payload["title"]), mechanism=str(payload["mechanism"]),
            features=tuple(payload.get("features", ())), parameters=dict(payload.get("parameters", {})),
            universe=tuple(payload.get("universe", ())), market=str(payload.get("market", "unknown")),
            result=outcome, failure_reason=str(payload.get("failure_reason", "historical negative result" if memory_type == "FAILURE" else "")),
            source_refs=(str(payload["source_path"]), "historical_import"),
        )
        return self.memory.remember(record, actor=actor)
