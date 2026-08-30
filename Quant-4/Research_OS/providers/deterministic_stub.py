"""Offline provider for tests, demos and provider-outage fallback validation."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

from .base import AgentRequest, AgentResponse, Provider


class DeterministicStubProvider(Provider):
    def __init__(self, provider_id: str = "deterministic-stub", model_id: str = "stub-v1",
                 responder: Callable[[AgentRequest], dict[str, Any]] | None = None):
        self.provider_id = provider_id
        self.model_id = model_id
        self._responder = responder

    def complete(self, request: AgentRequest) -> AgentResponse:
        payload = self._responder(request) if self._responder else {
            "status": "HOLD", "proposal": f"offline assessment for {request.task}",
            "reasons": ["deterministic stub does not claim external evidence"],
        }
        if not isinstance(payload, dict):
            raise TypeError("provider output must be a mapping")
        return AgentResponse(request_id=request.request_id, provider_id=self.provider_id,
                             model_id=self.model_id, role=request.role, payload=deepcopy(payload))
