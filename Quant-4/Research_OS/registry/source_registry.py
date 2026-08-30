from __future__ import annotations

from pathlib import Path

from Research_OS.contracts.evidence import SourceRecord

from .storage import HashChainStore


class SourceRegistry:
    def __init__(self, path: str | Path):
        self.store = HashChainStore(path, schema_version="source-registry/v1")

    def create(self, source: SourceRecord, *, actor: str) -> dict:
        return self.store.append("SOURCE_REGISTERED", source.to_dict(), actor=actor)

    def get(self, source_id: str) -> dict:
        rows = [row["payload"] for row in self.store.events({"SOURCE_REGISTERED"}) if row["payload"]["source_id"] == source_id]
        if not rows:
            raise KeyError(source_id)
        return rows[-1]
