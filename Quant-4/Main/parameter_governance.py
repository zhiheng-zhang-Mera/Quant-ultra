"""Validate and persist parameter proposals; never mutates production config."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

RULES = {
    "max_single_stock_weight": {"min": 0.01, "max": 0.10, "max_relative_change": 0.25},
    "sector_limit": {"min": 0.10, "max": 0.40, "max_relative_change": 0.25},
    "transaction_cost_coeff": {"min": 0.00005, "max": 0.005, "max_relative_change": 0.50},
    "gamma_risk_initial": {"min": 0.5, "max": 10.0, "max_relative_change": 0.30},
    "psi_threshold": {"min": 0.10, "max": 0.35, "max_relative_change": 0.25},
}


def validate_parameter_proposal(proposal: dict, current: dict) -> dict:
    accepted, errors = {}, []
    if not isinstance(proposal, dict):
        return {"valid": False, "accepted": {}, "errors": ["proposal must be an object"]}
    for key, value in proposal.items():
        if key not in RULES:
            errors.append(f"parameter not allowlisted: {key}")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            errors.append(f"parameter must be numeric: {key}")
            continue
        rule = RULES[key]
        value = float(value)
        if not rule["min"] <= value <= rule["max"]:
            errors.append(f"parameter outside hard bounds: {key}")
            continue
        old = current.get(key)
        if isinstance(old, (int, float)) and old != 0 and abs(value - float(old)) / abs(float(old)) > rule["max_relative_change"]:
            errors.append(f"parameter change exceeds review limit: {key}")
            continue
        accepted[key] = value
    return {"valid": not errors and bool(accepted), "accepted": accepted, "errors": errors, "requires_human_approval": True, "applied": False}


def write_pending_proposal(validation: dict, report_dir: Path, run_id: str) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"pending_parameter_proposal_{run_id}.json"
    payload = {"created_at": datetime.now().astimezone().isoformat(), "status": "PENDING_HUMAN_APPROVAL" if validation["valid"] else "REJECTED", **validation}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
