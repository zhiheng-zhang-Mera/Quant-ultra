"""Append-only, hash-chained JSONL storage with corruption detection."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterable

from Research_OS.contracts.common import canonical_json, sha256


class RegistryCorruptionError(ValueError):
    pass


class HashChainStore:
    def __init__(self, path: str | Path, *, schema_version: str):
        self.path = Path(path)
        self.schema_version = schema_version
        self._lock = RLock()

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        previous = "GENESIS"
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
            for number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                event_hash = row.pop("event_hash", None)
                expected = sha256(row)
                if row.get("previous_hash") != previous or event_hash != expected:
                    raise RegistryCorruptionError(f"invalid hash chain at line {number}")
                row["event_hash"] = event_hash
                rows.append(row)
                previous = event_hash
        except (OSError, json.JSONDecodeError) as exc:
            raise RegistryCorruptionError(f"cannot read registry: {exc}") from exc
        return rows

    def append(self, event_type: str, payload: dict[str, Any], *, actor: str) -> dict[str, Any]:
        if not event_type or not actor:
            raise ValueError("event_type and actor are required")
        with self._lock:
            prior = self.read()
            row = {
                "schema_version": self.schema_version,
                "recorded_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "previous_hash": prior[-1]["event_hash"] if prior else "GENESIS",
                "event_type": event_type,
                "actor": actor,
                "payload": payload,
            }
            row["event_hash"] = sha256(row)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND
            descriptor = os.open(self.path, flags, 0o600)
            try:
                with os.fdopen(descriptor, "a", encoding="utf-8", newline="\n") as handle:
                    handle.write(canonical_json(row) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                os.close(descriptor)
                raise
            return row

    def events(self, event_types: Iterable[str] | None = None) -> list[dict[str, Any]]:
        rows = self.read()
        allowed = set(event_types or ())
        return [row for row in rows if not allowed or row["event_type"] in allowed]
