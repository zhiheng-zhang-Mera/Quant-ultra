"""Versioned, redacted application events for UI and automation clients."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from Research_OS.contracts.common import Contract, utc_now


@dataclass(frozen=True)
class AppEvent(Contract):
    SCHEMA: ClassVar[str] = "research-application-event/v1"
    event_id: str = ""
    run_id: str = ""
    sequence: int = 0
    event_type: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=utc_now)
    correlation_id: str = ""
    redaction_level: str = "PUBLIC"

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.event_id or not self.run_id or not self.event_type or self.sequence < 1:
            raise ValueError("event id, run id, event type and positive sequence are required")
