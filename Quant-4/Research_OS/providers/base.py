"""Provider-neutral, structured agent transport."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from Research_OS.contracts.common import sha256, utc_now


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentRequest:
    request_id: str
    role: str
    task: str
    context: dict[str, Any] = field(default_factory=dict)
    response_schema: dict[str, Any] = field(default_factory=dict)

    @property
    def context_sha256(self) -> str:
        return sha256(self.context)


@dataclass(frozen=True)
class AgentResponse:
    request_id: str
    provider_id: str
    model_id: str
    role: str
    payload: dict[str, Any]
    created_at: datetime = field(default_factory=utc_now)

    @property
    def output_sha256(self) -> str:
        return sha256(self.payload)


class Provider(ABC):
    provider_id: str
    model_id: str

    @abstractmethod
    def complete(self, request: AgentRequest) -> AgentResponse:
        raise NotImplementedError


class ProviderRegistry:
    def __init__(self):
        self._providers: dict[str, Provider] = {}

    def register(self, provider: Provider) -> None:
        if provider.provider_id in self._providers:
            raise ValueError("provider already registered")
        self._providers[provider.provider_id] = provider

    def get(self, provider_id: str) -> Provider:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise ProviderError(f"provider is unavailable: {provider_id}") from exc

    def independent_provider_count(self) -> int:
        return len(self._providers)
