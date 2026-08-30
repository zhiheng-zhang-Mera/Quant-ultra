"""Isolated adapter around canonical ``Main/main.py``.

The legacy ``main_engine.py`` is intentionally never imported.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from Research_OS.contracts.common import canonical_json, sha256


@dataclass(frozen=True)
class QuantKernelRequest:
    experiment_id: str
    spec_sha256: str
    config_overlay: dict[str, Any] = field(default_factory=dict)
    bounded_symbols: tuple[str, ...] = ()
    offline: bool = True
    timeout_seconds: int = 900
    allow_dirty_research: bool = False


@dataclass(frozen=True)
class QuantKernelResult:
    status: str
    return_code: int
    command: tuple[str, ...]
    stdout_sha256: str
    stderr_sha256: str
    production_config_unchanged: bool
    git_state: str = "UNKNOWN"
    run_classification: str = "HOLD"
    config_sha256: str = ""
    run_fingerprint: str = ""
    evidence_manifest: dict[str, Any] = field(default_factory=dict)
    evidence_paths: tuple[str, ...] = ()
    reason: str = ""


FORBIDDEN_OVERLAY_KEYS = frozenset({"analysis_only", "production", "live_trading", "broker", "accept_production"})


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "MISSING"


class QuantKernelAdapter:
    def __init__(self, project_root: str | Path | None = None,
                 runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
                 git_state_provider: Callable[[Path], str] | None = None):
        self.project_root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        self.main_path = self.project_root / "Main" / "main.py"
        self.production_config = self.project_root / "Main" / "default_param.yaml"
        self._runner = runner
        self._git_state_provider = git_state_provider or self._git_state
        if not self.main_path.is_file():
            raise FileNotFoundError(self.main_path)

    def validate_overlay(self, overlay: dict[str, Any]) -> dict[str, Any]:
        forbidden = FORBIDDEN_OVERLAY_KEYS & set(overlay)
        if forbidden:
            raise ValueError(f"research overlay cannot set privileged keys: {sorted(forbidden)}")
        safe = json.loads(canonical_json(overlay))
        safe["analysis_only"] = True
        return safe

    @staticmethod
    def _git_state(project_root: Path) -> str:
        repository_root = project_root.parent
        result = subprocess.run(
            ["git", "status", "--porcelain"], cwd=repository_root,
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return "UNKNOWN"
        return "DIRTY" if result.stdout.strip() else "CLEAN"

    def run(self, request: QuantKernelRequest) -> QuantKernelResult:
        overlay = self.validate_overlay(request.config_overlay)
        git_state = self._git_state_provider(self.project_root)
        if git_state != "CLEAN" and not request.allow_dirty_research:
            return QuantKernelResult(
                status="HOLD", return_code=2, command=(), stdout_sha256=sha256(""), stderr_sha256=sha256(""),
                production_config_unchanged=True, git_state=git_state, run_classification="HOLD",
                config_sha256=sha256(overlay), run_fingerprint=sha256([request.spec_sha256, overlay, git_state]),
                reason="research kernel requires a clean Git worktree unless an explicit dirty research override is recorded",
            )
        before = _file_hash(self.production_config)
        with tempfile.TemporaryDirectory(prefix="quant-ultra-research-") as temporary:
            config_path = Path(temporary) / "experiment.yaml"
            config_path.write_text(yaml.safe_dump(overlay, sort_keys=True), encoding="utf-8")
            command = [sys.executable, str(self.main_path), "--config", str(config_path), "--non-interactive"]
            if git_state != "CLEAN":
                command.append("--no-git-check")
            if request.offline:
                command.append("--offline")
            if request.bounded_symbols:
                command += ["--symbols", ",".join(request.bounded_symbols)]
            try:
                completed = self._runner(command, cwd=self.project_root, capture_output=True,
                                         text=True, timeout=request.timeout_seconds, check=False)
                reason = "" if completed.returncode == 0 else "kernel returned a non-zero status"
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
                stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
                completed = subprocess.CompletedProcess(command, 124, stdout, stderr)
                reason = "kernel execution timed out"
        unchanged = before == _file_hash(self.production_config)
        dirty_override = git_state != "CLEAN"
        status = "PRIMARY_EVIDENCE" if completed.returncode == 0 and unchanged and not dirty_override else "HOLD"
        classification = "DIRTY_RESEARCH_RUN" if dirty_override else ("CLEAN_RESEARCH_RUN" if status == "PRIMARY_EVIDENCE" else "HOLD")
        config_hash = sha256(overlay)
        fingerprint = sha256([request.spec_sha256, config_hash, git_state, completed.returncode])
        evidence_manifest = {
            "schema_version": "kernel-adapter-evidence/v2",
            "experiment_id": request.experiment_id,
            "spec_sha256": request.spec_sha256,
            "config_sha256": config_hash,
            "git_state": git_state,
            "run_classification": classification,
            "stdout_sha256": sha256(completed.stdout or ""),
            "stderr_sha256": sha256(completed.stderr or ""),
            "production_config_before": before,
            "production_config_after": _file_hash(self.production_config),
            "production_candidate_eligible": status == "PRIMARY_EVIDENCE",
        }
        return QuantKernelResult(status=status, return_code=completed.returncode, command=tuple(command),
                                 stdout_sha256=sha256(completed.stdout or ""), stderr_sha256=sha256(completed.stderr or ""),
                                 production_config_unchanged=unchanged, git_state=git_state,
                                 run_classification=classification, config_sha256=config_hash,
                                 run_fingerprint=fingerprint, evidence_manifest=evidence_manifest,
                                 reason=reason or ("dirty research runs are ineligible for production candidacy" if dirty_override else ""))
