"""DAG, contract and run-fingerprint safeguards for the one-click launcher."""
from __future__ import annotations
import hashlib
import json
import platform
from pathlib import Path


def validate_orchestration(modules, dependencies, input_schema, output_schema) -> dict:
    module_set = set(modules)
    if len(module_set) != len(modules):
        raise ValueError("PHASE_MODULES contains duplicates")
    missing_contracts = [m for m in modules if m not in input_schema or m not in output_schema]
    if missing_contracts:
        raise ValueError(f"Missing phase contracts: {missing_contracts}")
    positions = {m: i for i, m in enumerate(modules)}
    for phase in modules:
        unknown = set(dependencies.get(phase, set())) - module_set
        if unknown:
            raise ValueError(f"{phase} has unknown dependencies: {sorted(unknown)}")
        late = [d for d in dependencies.get(phase, set()) if positions[d] >= positions[phase]]
        if late:
            raise ValueError(f"{phase} dependency order/cycle violation: {late}")
    return {"valid": True, "phase_count": len(modules), "first_phase": modules[0], "last_phase": modules[-1]}


def build_run_fingerprint(git_hash: str, config: dict, modules: list[str]) -> tuple[str, dict]:
    material = {"git_hash": git_hash, "config": config, "modules": modules, "python": platform.python_version(), "schema_version": 1}
    canonical = json.dumps(material, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), material


def write_startup_manifest(report_root: Path, run_id: str, fingerprint: str, material: dict, dag_audit: dict) -> Path:
    path = report_root / "runs" / run_id / "startup_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"run_fingerprint": fingerprint, "fingerprint_material": material, "dag_audit": dag_audit}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return path

