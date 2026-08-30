"""Source-lineage, near-duplicate and vendor-reconciliation evidence."""
from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Iterable

from Research_OS.contracts.common import LifecycleStatus
from Research_OS.contracts.composite import SourceIndependenceEvidence
from Research_OS.contracts.evidence import EvidenceRecord, SourceRecord


def _normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[\w\u4e00-\u9fff]+", value.casefold()))


def _origin(source_id: str, sources: dict[str, SourceRecord]) -> str:
    seen: set[str] = set()
    current = source_id
    while current in sources and sources[current].upstream_source_id:
        if current in seen:
            raise ValueError("source lineage contains a cycle")
        seen.add(current)
        current = str(sources[current].upstream_source_id)
    return current


def near_duplicate_clusters(evidence: Iterable[EvidenceRecord], threshold: float = 0.90) -> tuple[tuple[str, ...], ...]:
    rows = list(evidence)
    parent = list(range(len(rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            exact = bool(rows[left].content_sha256 and rows[left].content_sha256 == rows[right].content_sha256)
            similar = SequenceMatcher(None, _normalized_text(rows[left].claim), _normalized_text(rows[right].claim)).ratio() >= threshold
            translated_or_syndicated = bool(rows[left].entity == rows[right].entity and rows[left].event_type == rows[right].event_type
                                            and rows[left].direction == rows[right].direction and similar)
            if exact or translated_or_syndicated:
                union(left, right)
    groups: dict[int, list[str]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row.evidence_id)
    return tuple(tuple(sorted(group)) for group in sorted(groups.values()) if len(group) > 1)


def reconcile_records(left: dict[str, Any], right: dict[str, Any], *, keys: Iterable[str],
                      source_a: str, source_b: str) -> tuple[dict[str, Any], ...]:
    findings = []
    for key in keys:
        a, b = left.get(key), right.get(key)
        if a != b:
            findings.append({"field": key, "source_a": source_a, "value_a": a,
                             "source_b": source_b, "value_b": b, "status": "CONFLICT"})
    return tuple(findings)


class SourceIndependenceAnalyzer:
    def analyze(self, experiment_id: str, sources: Iterable[SourceRecord], evidence: Iterable[EvidenceRecord], *,
                reconciliation_findings: tuple[dict[str, Any], ...] = (), minimum_origins: int = 2) -> SourceIndependenceEvidence:
        source_map = {source.source_id: source for source in sources}
        evidence_rows = list(evidence)
        referenced = sorted({row.source_id for row in evidence_rows if row.source_id in source_map})
        groups: dict[str, list[str]] = defaultdict(list)
        for source_id in referenced:
            groups[_origin(source_id, source_map)].append(source_id)
        families = {source_map[source_id].origin_family or source_map[source_id].publisher or _origin(source_id, source_map)
                    for source_id in referenced}
        raw_count, origin_count = len(referenced), len(groups)
        adjusted = origin_count / raw_count if raw_count else 0.0
        duplicates = near_duplicate_clusters(evidence_rows)
        reasons = []
        if origin_count < minimum_origins:
            reasons.append(f"independent origin count {origin_count} is below required {minimum_origins}")
        if reconciliation_findings:
            reasons.append("cross-source reconciliation contains unresolved conflicts")
        status = LifecycleStatus.PASS if not reasons else LifecycleStatus.HOLD
        return SourceIndependenceEvidence(
            schema_version="source-independence-evidence/v1", experiment_id=experiment_id,
            raw_source_count=raw_count, independent_origin_count=origin_count,
            source_family_count=len(families), syndication_adjusted_diversity=adjusted,
            origin_groups={key: tuple(sorted(value)) for key, value in sorted(groups.items())},
            near_duplicate_clusters=duplicates, reconciliation_findings=reconciliation_findings,
            status=status, reasons=tuple(reasons),
        )
