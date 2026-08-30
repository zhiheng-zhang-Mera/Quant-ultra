"""Offline deterministic structured/lexical research memory."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from Research_OS.contracts.governance import MemoryRecord
from Research_OS.registry.storage import HashChainStore


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff]+", text.casefold()))


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a or b else 0.0


@dataclass(frozen=True)
class SimilarityWeights:
    semantic: float = 0.35
    feature: float = 0.20
    parameter: float = 0.15
    mechanism: float = 0.20
    universe: float = 0.10


class ResearchMemory:
    TYPES = frozenset({"SEMANTIC", "EPISODIC", "FAILURE", "PROCEDURAL"})

    def __init__(self, path: str | Path, weights: SimilarityWeights = SimilarityWeights()):
        self.store = HashChainStore(path, schema_version="research-memory/v1")
        self.weights = weights

    def remember(self, record: MemoryRecord, *, actor: str) -> dict:
        if record.memory_type not in self.TYPES:
            raise ValueError("invalid memory type")
        if record.memory_type == "FAILURE" and not record.failure_reason:
            raise ValueError("failure memory requires a failure reason")
        return self.store.append("MEMORY_RECORDED", record.to_dict(), actor=actor)

    def records(self, memory_type: str | None = None) -> list[dict]:
        rows = [row["payload"] for row in self.store.events({"MEMORY_RECORDED"})]
        return [row for row in rows if memory_type is None or row["memory_type"] == memory_type]

    def search(self, query: str, *, features: Iterable[str] = (), parameters: dict[str, str] | None = None,
               mechanism: str = "", universe: Iterable[str] = (), limit: int = 10) -> list[dict]:
        query_tokens = _tokens(query)
        results = []
        for row in self.records():
            semantic = _jaccard(query_tokens, _tokens(row["hypothesis_text"] + " " + row["result"]))
            feature = _jaccard(features, row.get("features", []))
            query_parameters = {f"{key}={value}" for key, value in (parameters or {}).items()}
            row_parameters = {f"{key}={value}" for key, value in row.get("parameters", {}).items()}
            parameter = _jaccard(query_parameters, row_parameters)
            mechanism_score = _jaccard(_tokens(mechanism or query), _tokens(row.get("mechanism", "")))
            universe_score = _jaccard(universe, row.get("universe", []))
            score = (self.weights.semantic * semantic + self.weights.feature * feature +
                     self.weights.parameter * parameter + self.weights.mechanism * mechanism_score +
                     self.weights.universe * universe_score)
            results.append({"score": round(score, 12), "novelty_warning": score >= 0.72,
                            "known_failure": row["memory_type"] == "FAILURE", "record": row})
        return sorted(results, key=lambda item: (-item["score"], item["record"]["memory_id"]))[:limit]
