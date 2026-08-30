"""Deterministic static checks plus independent-review findings."""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

from Research_OS.agents import validate_separation


class FindingSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Finding:
    code: str
    severity: FindingSeverity
    path: str
    line: int
    message: str
    resolved: bool = False


PATTERNS = {
    "FUTURE_SHIFT": re.compile(r"\.shift\(\s*-\d+"),
    "LABEL_LEAK": re.compile(r"(?:feature|X).*?(?:future|forward|target|label|y_test)|(?:target|label).*?(?:feature|X)", re.I),
    "BACKFILL": re.compile(r"\.(?:bfill|backfill)\("),
    "GLOBAL_RANDOM": re.compile(r"np\.random\.(?!default_rng)"),
    "SHELL_EXECUTION": re.compile(r"(?:os\.system|subprocess\.(?:run|Popen))\("),
    "HIDDEN_TARGET": re.compile(r"hidden_oos|desired_sharpe|target_result", re.I),
}


class StaticVerificationCoordinator:
    def verify(self, files: Mapping[str, str], *, developer_agent_id: str, reviewer_agent_id: str) -> dict:
        validate_separation(proposer_agent_id=developer_agent_id, verifier_agent_id=reviewer_agent_id)
        findings: list[Finding] = []
        for path in sorted(files):
            for line_number, line in enumerate(files[path].splitlines(), start=1):
                for code, pattern in PATTERNS.items():
                    if pattern.search(line):
                        severity = FindingSeverity.CRITICAL if code in {"FUTURE_SHIFT", "LABEL_LEAK", "HIDDEN_TARGET"} else FindingSeverity.WARNING
                        findings.append(Finding(code, severity, path, line_number, f"potential {code.lower().replace('_', ' ')}"))
        unresolved = [item for item in findings if item.severity == FindingSeverity.CRITICAL and not item.resolved]
        return {"status": "IMPLEMENTATION_REJECTED" if unresolved else "PASS", "findings": findings,
                "developer_agent_id": developer_agent_id, "reviewer_agent_id": reviewer_agent_id}
