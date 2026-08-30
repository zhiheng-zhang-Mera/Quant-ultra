"""Hash-addressed, human-reviewable lifecycle workspace."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from Research_OS.contracts.common import canonical_json, sha256
from Research_OS.orchestration import StageContext
from Research_OS.security import safe_workspace_path

STAGE_FILES = (
    "00_request.json", "01_memory.md", "02_sources.json", "03_hypotheses.md", "04_triage.json",
    "05_mechanism.md", "06_feasibility.json", "07_experiment_design.json", "08_preregistration.json",
    "09_data_manifest.json", "10_pit_evidence.json", "11_data_gate.json", "12_implementation_manifest.json",
    "13_code_verification.json", "14_primary_kernel_run.json", "15_reproduction.json", "16_statistics.json",
    "17_robustness.json", "18_generalization.json", "19_governance.json", "20_memory_update.json",
)


class ReportBundleWriter:
    def __init__(self, reports_root: str | Path):
        self.reports_root = Path(reports_root).resolve()

    def write(self, context: StageContext) -> Path:
        run_root = safe_workspace_path(self.reports_root, context.run_id)
        run_root.mkdir(parents=True, exist_ok=True)
        hashes: dict[str, str] = {}
        for number, filename in enumerate(STAGE_FILES):
            stage_id = f"R{number}"
            result = context.results.get(stage_id)
            payload: dict[str, Any] = ({"stage_id": result.stage_id, "status": result.status.value,
                                       "output": result.output, "evidence_refs": result.evidence_refs,
                                       "reasons": result.reasons} if result else
                                      {"stage_id": stage_id, "status": "PENDING", "output": {},
                                       "evidence_refs": [], "reasons": ["stage not executed"]})
            payload["schema_version"] = "research-workspace-evidence/v1"
            payload["evidence_sha256"] = sha256(payload)
            path = safe_workspace_path(run_root, filename)
            if path.suffix == ".md":
                path.write_text(self._stage_markdown(payload), encoding="utf-8")
            else:
                path.write_text(canonical_json(payload) + "\n", encoding="utf-8")
            hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        summary = self._summary(context, hashes)
        safe_workspace_path(run_root, "summary.md").write_text(summary, encoding="utf-8")
        safe_workspace_path(run_root, "manifest.json").write_text(
            json.dumps({"schema_version": "research-workspace-manifest/v1", "run_id": context.run_id,
                        "files": hashes, "manifest_sha256": sha256(hashes)}, ensure_ascii=False, indent=2), encoding="utf-8")
        return run_root

    @staticmethod
    def _stage_markdown(payload: dict[str, Any]) -> str:
        return (f"# {payload['stage_id']} research evidence\n\n"
                f"Status: `{payload['status']}`\n\n"
                f"Evidence SHA-256: `{payload['evidence_sha256']}`\n\n"
                "```json\n" + json.dumps(payload["output"], ensure_ascii=False, indent=2) + "\n```\n")

    @staticmethod
    def _summary(context: StageContext, hashes: dict[str, str]) -> str:
        rows = []
        for stage_id in sorted(context.results, key=lambda value: int(value[1:])):
            result = context.results[stage_id]
            rows.append(f"| {stage_id} | {result.status.value} | {', '.join(result.reasons) or '—'} |")
        return f"""# Research OS run {context.run_id}

This workspace records research evidence. It is not investment advice and does not authorize live trading.

```mermaid
flowchart TD
    D[Discover R0-R3] --> T[Triage R4-R6]
    T --> L[Lock R7-R8]
    L --> V[Build and verify R9-R18]
    V --> G[Govern and learn R19-R20]
```

| Stage | Status | Reasons |
|---|---|---|
{chr(10).join(rows)}

Every immutable evidence file is listed in `manifest.json` with a SHA-256. Missing and conflicting evidence remain explicit; source agreement is not treated as truth.
"""
