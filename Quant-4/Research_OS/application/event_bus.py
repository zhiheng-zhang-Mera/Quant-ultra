"""Durable monotonic event bus with replay, deduplication and redaction."""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from pathlib import Path
from threading import RLock
from typing import Any

from Research_OS.contracts.common import canonical_json, sha256, stable_id
from Research_OS.registry.storage import HashChainStore

from .events import AppEvent

_SECRET_KEYS = frozenset({"token", "api_key", "apikey", "password", "secret", "credential", "authorization"})


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): "[REDACTED]" if str(key).lower() in _SECRET_KEYS else _redact(item)
                for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    return value


class PersistentEventBus:
    """Synchronous event bus; slow work belongs in a process manager, never a subscriber."""

    def __init__(self, path: str | Path, *, recent_limit: int = 500):
        if recent_limit < 1:
            raise ValueError("recent_limit must be positive")
        self.store = HashChainStore(path, schema_version="research-application-events/v1")
        self._recent: deque[AppEvent] = deque(maxlen=recent_limit)
        self._subscribers: list[Callable[[AppEvent], None]] = []
        self._sequence: dict[str, int] = defaultdict(int)
        self._ids: set[str] = set()
        self._event_index: dict[str, AppEvent] = {}
        self._lock = RLock()
        for event in self.replay():
            self._remember(event)

    @staticmethod
    def _from_payload(payload: dict[str, Any]) -> AppEvent:
        from datetime import datetime

        data = dict(payload)
        data["occurred_at"] = datetime.fromisoformat(data["occurred_at"].replace("Z", "+00:00"))
        return AppEvent(**data)

    def _remember(self, event: AppEvent) -> None:
        if event.event_id in self._ids:
            return
        expected = self._sequence[event.run_id] + 1
        if event.sequence != expected:
            raise ValueError(f"out-of-order event for {event.run_id}: expected {expected}, got {event.sequence}")
        self._sequence[event.run_id] = event.sequence
        self._ids.add(event.event_id)
        self._event_index[event.event_id] = event
        self._recent.append(event)

    def publish(self, run_id: str, event_type: str, payload: dict[str, Any], *,
                correlation_id: str = "", event_id: str | None = None) -> AppEvent:
        with self._lock:
            safe_payload = _redact(payload)
            sequence = self._sequence[run_id] + 1
            identity = event_id or stable_id("EVD", run_id, event_type, sequence, sha256(safe_payload))
            if identity in self._ids:
                return self._event_index[identity]
            event = AppEvent(AppEvent.SCHEMA, identity, run_id, sequence, event_type, safe_payload,
                             correlation_id=correlation_id)
            self.store.append("APP_EVENT", event.to_dict(), actor="ResearchApplicationService")
            self._remember(event)
            for subscriber in tuple(self._subscribers):
                subscriber(event)
            return event

    def replay(self, *, run_id: str | None = None, after_sequence: int = 0,
               event_types: Iterable[str] | None = None) -> list[AppEvent]:
        allowed = set(event_types or ())
        events = [self._from_payload(row["payload"]) for row in self.store.events(("APP_EVENT",))]
        return [event for event in events if (run_id is None or event.run_id == run_id)
                and event.sequence > after_sequence and (not allowed or event.event_type in allowed)]

    def subscribe(self, callback: Callable[[AppEvent], None], *, replay_recent: bool = False) -> Callable[[], None]:
        with self._lock:
            self._subscribers.append(callback)
            if replay_recent:
                for event in tuple(self._recent):
                    callback(event)
        return lambda: self._subscribers.remove(callback) if callback in self._subscribers else None

    def export_jsonl(self, *, run_id: str | None = None) -> str:
        return "\n".join(canonical_json(event) for event in self.replay(run_id=run_id))
