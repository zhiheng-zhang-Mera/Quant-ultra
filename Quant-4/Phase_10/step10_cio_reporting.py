"""Phase 10 entrypoint: evidence-only CIO summary without fabricated metrics."""
from __future__ import annotations
import json
from pathlib import Path


def execute(context: dict) -> dict:
    report_root = Path(__file__).parents[1] / "reports" / "cio"
    report_root.mkdir(parents=True, exist_ok=True)
    run_id = context.get("run_metadata", {}).get("timestamp", "unknown")
    evidence = {
        "run_metadata": context.get("run_metadata", {}),
        "audit_summary": context.get("audit_summary"),
        "audit_passed": context.get("audit_passed"),
        "final_nav": context.get("final_nav"),
        "reconciliation_mae": context.get("reconciliation_mae"),
        "recon_passed": context.get("recon_passed"),
        "psi_consecutive_breaches": context.get("psi_consecutive_breaches"),
        "completed_phases": sorted(context.get("_completed_phases", [])),
    }
    missing = [k for k, value in evidence.items() if value is None]
    decision = "HOLD_FOR_REVIEW" if missing or evidence.get("audit_passed") is not True else "ELIGIBLE_FOR_PHASE_11"
    payload = {"decision": decision, "missing_evidence": missing, "evidence": evidence}
    path = report_root / f"cio_{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return {"cio_decision": decision, "cio_evidence": payload, "cio_report_path": str(path), "phase10_ready": True}

