"""Environment, experiment and artifact reproducibility manifests."""
from __future__ import annotations

import hashlib
from importlib import metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import numpy as np

from Main.data_provenance import sha256_file


def _canonical(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _git(project_root: Path) -> tuple[str, bool]:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root, capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=project_root, capture_output=True, text=True, check=False)
    return head.stdout.strip() if head.returncode == 0 else "NO_GIT", bool(status.stdout.strip())


def build_environment_manifest(project_root: Path, *, random_seeds: Mapping[str, int], hardware: dict | None = None) -> dict:
    if not random_seeds or any(not isinstance(seed, int) for seed in random_seeds.values()):
        raise ValueError("all stochastic components require integer random seeds")
    project_root = Path(project_root)
    commit, dirty = _git(project_root)
    lock_files = sorted(project_root.glob("Quant-4/requirements-lock-*.txt"))
    packages = {item.metadata["Name"]: item.version for item in metadata.distributions() if item.metadata.get("Name")}
    payload = {
        "schema_version": "environment-manifest/v1", "python": sys.version,
        "python_executable": sys.executable, "implementation": platform.python_implementation(),
        "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
        "git_commit": commit, "git_dirty": dirty,
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "lock_files": {str(path.relative_to(project_root)).replace("\\", "/"): sha256_file(path) for path in lock_files},
        "random_seeds": dict(sorted(random_seeds.items())), "hardware": hardware or {},
        "runtime_controls": {name: os.environ.get(name) for name in
                             ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS", "TZ")},
    }
    payload["manifest_sha256"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return payload


def write_environment_manifest(path: Path, manifest: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def build_artifact_manifest(
    *, experiment_id: str, code_version: str, data_version: str, parameter_version: str,
    inputs: Mapping[str, Path], outputs: Mapping[str, Path], numeric_tolerances: Mapping[str, float],
) -> dict:
    if not all((experiment_id, code_version, data_version, parameter_version)):
        raise ValueError("artifact manifest requires experiment, code, data and parameter versions")
    if not outputs or not numeric_tolerances or any(value < 0 for value in numeric_tolerances.values()):
        raise ValueError("outputs and non-negative numeric tolerances are required")
    def hashes(files):
        result = {}
        for name, path in files.items():
            path = Path(path)
            if not path.is_file():
                raise FileNotFoundError(path)
            result[name] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        return result
    payload = {"schema_version": "research-artifact/v1", "experiment_id": experiment_id,
               "code_version": code_version, "data_version": data_version,
               "parameter_version": parameter_version, "inputs": hashes(inputs), "outputs": hashes(outputs),
               "numeric_tolerances": dict(numeric_tolerances)}
    payload["manifest_sha256"] = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return payload


def verify_artifact_manifest(manifest: dict) -> dict:
    failures = []
    for section in ("inputs", "outputs"):
        for name, item in manifest.get(section, {}).items():
            path = Path(item.get("path", ""))
            if not path.is_file():
                failures.append(f"missing:{section}:{name}")
            elif sha256_file(path) != item.get("sha256"):
                failures.append(f"hash_mismatch:{section}:{name}")
    material = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if hashlib.sha256(_canonical(material).encode("utf-8")).hexdigest() != manifest.get("manifest_sha256"):
        failures.append("manifest_hash_mismatch")
    return {"valid": not failures, "failures": failures}


def compare_numeric_results(expected: Mapping[str, float], actual: Mapping[str, float], tolerances: Mapping[str, float]) -> dict:
    missing = sorted((set(expected) | set(actual)) - set(tolerances))
    comparisons = {}
    for key in sorted(set(expected) & set(actual) & set(tolerances)):
        left, right, tolerance = float(expected[key]), float(actual[key]), float(tolerances[key])
        difference = abs(left - right)
        comparisons[key] = {"expected": left, "actual": right, "absolute_difference": difference,
                            "tolerance": tolerance, "passed": bool(np.isfinite(left) and np.isfinite(right) and difference <= tolerance)}
    passed = not missing and set(expected) == set(actual) and comparisons and all(item["passed"] for item in comparisons.values())
    return {"passed": bool(passed), "missing_tolerances": missing, "comparisons": comparisons}
