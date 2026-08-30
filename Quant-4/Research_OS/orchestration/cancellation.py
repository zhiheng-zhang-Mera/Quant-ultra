"""Thread-safe cooperative cancellation primitives."""
from __future__ import annotations

from threading import Event


class RunCancelled(RuntimeError):
    pass


class CancellationToken:
    def __init__(self) -> None:
        self._requested = Event()
        self._acknowledged = Event()

    @property
    def requested(self) -> bool:
        return self._requested.is_set()

    @property
    def acknowledged(self) -> bool:
        return self._acknowledged.is_set()

    def request(self) -> None:
        self._requested.set()

    def checkpoint(self) -> None:
        if self.requested:
            self._acknowledged.set()
            raise RunCancelled("run cancellation acknowledged")

    def acknowledge(self) -> None:
        self._acknowledged.set()
