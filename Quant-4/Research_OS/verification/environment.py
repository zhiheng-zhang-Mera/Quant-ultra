"""Materialized workspace and execution-environment verification."""
from __future__ import annotations

import hashlib
import locale
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from Research_OS.contracts.common import sha256, stable_id
from Research_OS.contracts.composite import ExecutionEnvironmentManifest

DEFAULT_ENV_WHITELIST = ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TZ")


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(root: Path) -> tuple[str, str]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=False)
    if commit.returncode != 0 or dirty.returncode != 0:
        return "NO_GIT", "UNKNOWN"
    return commit.stdout.strip(), "DIRTY" if dirty.stdout.strip() else "CLEAN"


class WorkspaceMaterializer:
    IGNORE = (".git", "__pycache__", ".pytest_cache", ".mypy_cache", "reports", "Data_Cache", "Phase_Result")

    def materialize(self, source_root: str | Path, destination_parent: str | Path | None = None) -> Path:
        source = Path(source_root).resolve()
        if not source.is_dir():
            raise FileNotFoundError(source)
        if destination_parent is None:
            destination = Path(tempfile.mkdtemp(prefix="quant-ultra-independent-")) / "workspace"
        else:
            destination = Path(destination_parent).resolve() / "workspace"
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copytree(source, destination, ignore=shutil.ignore_patterns(*self.IGNORE), copy_function=shutil.copy2)
        return destination


def capture_environment(workspace: str | Path, *, experiment_spec_hash: str, data_manifest_hash: str,
                        random_seeds: dict[str, int], shared_writable_cache: bool = False,
                        environment_whitelist: Iterable[str] = DEFAULT_ENV_WHITELIST) -> ExecutionEnvironmentManifest:
    root = Path(workspace).resolve()
    git_sha, git_state = _git(root)
    locks = sorted(root.glob("**/requirements*.txt"))
    lock_hashes = {str(path.relative_to(root)).replace("\\", "/"): _hash_file(path) for path in locks if path.is_file()}
    selected_env = {key: os.environ.get(key) for key in sorted(set(environment_whitelist))}
    classification = "CLEAN_RESEARCH_RUN" if git_state == "CLEAN" and not shared_writable_cache else (
        "DIRTY_RESEARCH_RUN" if git_state == "DIRTY" else "HOLD"
    )
    return ExecutionEnvironmentManifest(
        schema_version="execution-environment-manifest/v1",
        workspace_id=stable_id("RUN", "workspace", str(root)), workspace_path_hash=sha256(str(root)),
        python_version=sys.version, dependency_lock_hashes=lock_hashes, os_name=platform.platform(),
        architecture=platform.machine(), timezone=os.environ.get("TZ", "SYSTEM"),
        locale=locale.getlocale()[0] or "UNKNOWN", random_seeds=dict(sorted(random_seeds.items())),
        cpu=platform.processor() or platform.machine(), gpu=os.environ.get("CUDA_VISIBLE_DEVICES", "NONE"),
        blas=str(getattr(__import__("numpy").__config__, "CONFIG", "NUMPY_DEFAULT")),
        git_sha=git_sha, git_state=git_state, environment_whitelist_hash=sha256(selected_env),
        data_manifest_hash=data_manifest_hash, experiment_spec_hash=experiment_spec_hash,
        shared_writable_cache=shared_writable_cache, run_classification=classification,
    )


def compare_environments(primary: ExecutionEnvironmentManifest, reproduction: ExecutionEnvironmentManifest) -> dict:
    fields = ("python_version", "dependency_lock_hashes", "architecture", "timezone", "locale", "random_seeds",
              "blas", "git_sha", "environment_whitelist_hash", "data_manifest_hash", "experiment_spec_hash")
    mismatches = {field: {"primary": getattr(primary, field), "reproduction": getattr(reproduction, field)}
                  for field in fields if getattr(primary, field) != getattr(reproduction, field)}
    isolated = primary.workspace_id != reproduction.workspace_id and not reproduction.shared_writable_cache
    return {"status": "PASS" if not mismatches and isolated else "HOLD", "isolated_workspace": isolated,
            "mismatches": mismatches, "production_candidate_eligible": not mismatches and isolated
            and primary.production_candidate_eligible and reproduction.production_candidate_eligible}
