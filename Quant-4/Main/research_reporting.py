"""Validated, hash-addressed standard research reports."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

REPORT_TYPES = frozenset({
    "STRATEGY", "FACTOR", "EXPERIMENT", "VALIDATION", "RISK", "DATA_QUALITY",
    "ALTERNATIVE_DATA", "LLM_EVALUATION", "BENCHMARK", "NEGATIVE_RESULT", "REPRODUCIBILITY",
})
REPORT_STATUSES = frozenset({"PASS", "HOLD", "REJECT"})


@dataclass(frozen=True)
class ResearchReportMetadata:
    report_type: str
    experiment_id: str
    status: str
    code_version: str
    data_version: str
    parameter_version: str
    generated_at: str
    title: str

    def validate(self) -> None:
        if self.report_type not in REPORT_TYPES or self.status not in REPORT_STATUSES:
            raise ValueError("report type and PASS/HOLD/REJECT status must be governed")
        if not self.experiment_id.startswith(("EXP-", "LEGACY-")):
            raise ValueError("report experiment id must start with EXP- or LEGACY-")
        if not all((self.code_version, self.data_version, self.parameter_version, self.generated_at, self.title)):
            raise ValueError("report requires code, data, parameter, time and title metadata")
        datetime.fromisoformat(self.generated_at.replace("Z", "+00:00"))


def build_research_report(
    metadata: ResearchReportMetadata, *, summary: str, metrics: Mapping[str, object],
    evidence: Sequence[str], limitations: Sequence[str], decision: str,
) -> dict:
    metadata.validate()
    if not summary.strip() or not metrics or not evidence or not limitations or not decision.strip():
        raise ValueError("standard reports require summary, metrics, evidence, limitations and decision")
    payload = {"schema_version": "standard-research-report/v1", "metadata": asdict(metadata),
               "summary": summary, "metrics": dict(metrics), "evidence": list(evidence),
               "limitations": list(limitations), "decision": decision}
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    payload["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def render_markdown(report: dict) -> str:
    meta = report["metadata"]
    metric_rows = "\n".join(f"| {key} | {value} |" for key, value in sorted(report["metrics"].items()))
    evidence = "\n".join(f"- {item}" for item in report["evidence"])
    limitations = "\n".join(f"- {item}" for item in report["limitations"])
    return f"""# {meta['title']}

| Field | Value |
|---|---|
| Report type | {meta['report_type']} |
| Experiment ID | {meta['experiment_id']} |
| Status | {meta['status']} |
| Code version | {meta['code_version']} |
| Data version | {meta['data_version']} |
| Parameter version | {meta['parameter_version']} |
| Generated at | {meta['generated_at']} |
| Report SHA-256 | {report['report_sha256']} |

## Summary

{report['summary']}

## Metrics

| Metric | Value |
|---|---|
{metric_rows}

## Evidence

{evidence}

## Limitations

{limitations}

## Decision

{report['decision']}
"""


def write_research_report(path: Path, report: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(report), encoding="utf-8")
    return path


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()
