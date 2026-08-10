"""Phase 10 entrypoint: evidence-only CIO summary without fabricated metrics."""
from __future__ import annotations
import json
from pathlib import Path
from Main.parameter_governance import validate_parameter_proposal, write_pending_proposal


def _json_safe(value):
    """Recursively convert NaN/Infinity into JSON-safe null values."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if not isinstance(value, (int, float)):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return value


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
    required_completed = {"Phase_8.step8_audit_stress_test", "Phase_9.step9_live_mlops"}
    completed = set(evidence["completed_phases"])
    prerequisite_gap = sorted(required_completed - completed)
    if prerequisite_gap:
        missing.append("completed_phases:" + ",".join(prerequisite_gap))
    governance_failed = evidence.get("audit_passed") is not True or evidence.get("recon_passed") is not True
    decision = "HOLD_FOR_REVIEW" if missing or governance_failed else "ELIGIBLE_FOR_PHASE_11"
    # 禁止非法的 NaN/Infinity 裸标记进入 JSON 报告（标准 JSON 解析器会拒绝），
    # 统一序列化为 null，同时保留“压力场景未覆盖”的标注供人工解读。
    payload = _json_safe({"decision": decision, "missing_evidence": missing, "evidence": evidence})
    path = report_root / f"cio_{run_id}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    proposal = context.get("parameter_proposal")
    proposal_status, proposal_path = "NOT_PROPOSED", None
    if proposal is not None:
        validation = validate_parameter_proposal(proposal, context.get("config", {}))
        proposal_path = write_pending_proposal(validation, report_root, run_id)
        proposal_status = "PENDING_HUMAN_APPROVAL" if validation["valid"] else "REJECTED"
    return {"cio_decision": decision, "cio_evidence": payload, "cio_report_path": str(path), "parameter_proposal_status": proposal_status, "parameter_proposal_path": str(proposal_path) if proposal_path else None, "phase10_ready": True}
