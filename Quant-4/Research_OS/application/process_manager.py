"""Ownership, timeout, termination and reconciliation for research subprocesses."""
from __future__ import annotations

import hashlib
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ProcessRecord:
    process_id: str
    pid: int
    command: tuple[str, ...]
    cwd: str
    status: str
    started_at: float
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""


class ResearchProcessManager:
    def __init__(self, *, terminate_grace_seconds: float = 2.0):
        self.terminate_grace_seconds = terminate_grace_seconds
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._records: dict[str, ProcessRecord] = {}
        self._lock = RLock()

    def start(self, command: Sequence[str], *, cwd: str | Path, env: Mapping[str, str] | None = None) -> ProcessRecord:
        if not command or any(not isinstance(item, str) or not item for item in command):
            raise ValueError("command must contain non-empty arguments")
        root = Path(cwd).resolve()
        process = subprocess.Popen(tuple(command), cwd=root, env=dict(env) if env else None,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        identity = "PROC-" + hashlib.sha256(f"{process.pid}:{time.time_ns()}".encode()).hexdigest()[:16]
        record = ProcessRecord(identity, process.pid, tuple(command), str(root), "RUNNING", time.time())
        with self._lock:
            self._processes[identity] = process
            self._records[identity] = record
        return record

    def wait(self, process_id: str, *, timeout: float | None = None) -> ProcessRecord:
        process = self._process(process_id)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            status = "COMPLETED" if process.returncode == 0 else "CRASHED"
        except subprocess.TimeoutExpired:
            self.cancel(process_id)
            record = self._records[process_id]
            self._records[process_id] = replace(record, status="TIMED_OUT")
            return self._records[process_id]
        with self._lock:
            self._records[process_id] = replace(self._records[process_id], status=status,
                                                exit_code=process.returncode, stdout=stdout, stderr=stderr)
            return self._records[process_id]

    def cancel(self, process_id: str) -> ProcessRecord:
        process = self._process(process_id)
        if process.poll() is None:
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=self.terminate_grace_seconds)
                status = "CANCELLED"
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                status = "KILLED"
        else:
            stdout, stderr = process.communicate()
            status = "COMPLETED" if process.returncode == 0 else "CRASHED"
        with self._lock:
            self._records[process_id] = replace(self._records[process_id], status=status,
                                                exit_code=process.returncode, stdout=stdout, stderr=stderr)
            return self._records[process_id]

    def reconcile(self, process_id: str) -> ProcessRecord:
        process = self._process(process_id)
        if process.poll() is None:
            return self._records[process_id]
        return self.wait(process_id)

    def shutdown(self) -> None:
        for process_id, process in tuple(self._processes.items()):
            if process.poll() is None:
                self.cancel(process_id)

    def _process(self, process_id: str) -> subprocess.Popen[str]:
        with self._lock:
            if process_id not in self._processes:
                raise KeyError(process_id)
            return self._processes[process_id]
