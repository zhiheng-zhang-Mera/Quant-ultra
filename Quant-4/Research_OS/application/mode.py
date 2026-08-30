"""Capability-derived execution mode; clients may display but never choose trust."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ResearchExecutionMode(str, Enum):
    DEMO_OFFLINE = "DEMO_OFFLINE"
    RESEARCH_OFFLINE = "RESEARCH_OFFLINE"
    REAL_RESEARCH = "REAL_RESEARCH"
    UNKNOWN_LEGACY = "UNKNOWN_LEGACY"


@dataclass(frozen=True)
class ExecutionCapabilities:
    lifecycle_handler_kind: str
    providers_real: bool = False
    data_sources_real: bool = False
    kernel_real: bool = False
    verification_real: bool = False

    @property
    def mode(self) -> ResearchExecutionMode:
        if self.lifecycle_handler_kind in {"deterministic-stub", "offline-reference"}:
            return ResearchExecutionMode.DEMO_OFFLINE
        if all((self.providers_real, self.data_sources_real, self.kernel_real, self.verification_real)):
            return ResearchExecutionMode.REAL_RESEARCH
        if self.data_sources_real and self.kernel_real:
            return ResearchExecutionMode.RESEARCH_OFFLINE
        return ResearchExecutionMode.DEMO_OFFLINE


DEMO_CAPABILITIES = ExecutionCapabilities("offline-reference")
