from __future__ import annotations

from pathlib import Path

from Research_OS.contracts.research import Hypothesis

from .storage import HashChainStore


class HypothesisRegistry:
    def __init__(self, path: str | Path):
        self.store = HashChainStore(path, schema_version="hypothesis-registry/v1")

    def create(self, hypothesis: Hypothesis, *, actor: str) -> dict:
        if any(row["payload"]["hypothesis_id"] == hypothesis.hypothesis_id for row in self.store.events({"HYPOTHESIS_CREATED"})):
            raise ValueError("hypothesis already exists")
        return self.store.append("HYPOTHESIS_CREATED", hypothesis.to_dict(), actor=actor)

    def list(self, *, tags: tuple[str, ...] = ()) -> list[dict]:
        rows = [row["payload"] for row in self.store.events({"HYPOTHESIS_CREATED"})]
        return [row for row in rows if not tags or set(tags).issubset(row.get("tags", []))]
