from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

SECRET_PATTERN = re.compile(r"(?i)(api[_-]?key|token|secret|password)(\s*[:=]\s*)([^\s,;]+)")


def redact_secrets(text: str) -> str:
    return SECRET_PATTERN.sub(lambda match: match.group(1) + match.group(2) + "[REDACTED]", text)


def safe_workspace_path(root: str | Path, candidate: str | Path) -> Path:
    base = Path(root).resolve()
    path = (base / candidate).resolve() if not Path(candidate).is_absolute() else Path(candidate).resolve()
    if path != base and base not in path.parents:
        raise ValueError("path escapes the research workspace")
    return path


@dataclass(frozen=True)
class ExecutionPolicy:
    workspace_root: Path
    allowed_commands: tuple[str, ...] = ("python", "pytest", "ruff", "mypy")

    def validate(self, command: Iterable[str]) -> tuple[str, ...]:
        parts = tuple(command)
        if not parts or Path(parts[0]).name not in self.allowed_commands:
            raise PermissionError("AI-provided command is not allowed")
        if any(part in {"-c", "--eval"} for part in parts):
            raise PermissionError("inline arbitrary code execution is not allowed")
        return parts
