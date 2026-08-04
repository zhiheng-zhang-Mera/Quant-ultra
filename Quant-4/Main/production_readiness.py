"""Deterministic preflight checks for deployable Quant-4 environments."""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import subprocess
from pathlib import Path

import yaml

from Main.orchestration_guard import validate_orchestration
from Main.schema_contracts import PHASE_DEPENDENCIES, PHASE_INPUT_SCHEMA, PHASE_MODULES, PHASE_OUTPUT_SCHEMA

PROJECT_ROOT = Path(__file__).parents[1]
REQUIRED_IMPORTS = ("numpy", "pandas", "yaml", "pyarrow", "pytest")


def run_preflight(*, require_clean_git: bool = True) -> dict:
    checks: dict[str, dict] = {}
    version = tuple(int(part) for part in platform.python_version_tuple()[:2])
    checks["python"] = {"passed": version >= (3, 11), "version": platform.python_version()}

    missing = [name for name in REQUIRED_IMPORTS if importlib.util.find_spec(name) is None]
    checks["dependencies"] = {"passed": not missing, "missing": missing}

    config_path = PROJECT_ROOT / "Main" / "default_param.yaml"
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        config_ok = isinstance(config, dict) and bool(config) and config.get("analysis_only") is True
    except Exception as exc:
        config, config_ok = {"error": repr(exc)}, False
    checks["default_config"] = {"passed": config_ok, "path": str(config_path), "keys": len(config) if isinstance(config, dict) else 0, "analysis_only": config.get("analysis_only") if isinstance(config, dict) else None}

    try:
        dag = validate_orchestration(PHASE_MODULES, PHASE_DEPENDENCIES, PHASE_INPUT_SCHEMA, PHASE_OUTPUT_SCHEMA)
        dag_ok = dag.get("phase_count") == 11
    except Exception as exc:
        dag, dag_ok = {"error": repr(exc)}, False
    checks["orchestration"] = {"passed": dag_ok, "evidence": dag}

    git = subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    dirty = bool(git.stdout.strip())
    checks["git"] = {"passed": git.returncode == 0 and (not dirty or not require_clean_git), "dirty": dirty, "required_clean": require_clean_git}

    return {
        "ready": all(item["passed"] for item in checks.values()),
        "checks": checks,
        "boundaries": [
            "Preflight proves local engineering prerequisites only.",
            "It does not prove market-data authenticity, broker connectivity, live execution safety, or investment performance.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Quant-4 production engineering preflight")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_preflight(require_clean_git=not args.allow_dirty)
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
