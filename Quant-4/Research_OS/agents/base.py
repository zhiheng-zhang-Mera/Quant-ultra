"""Capability-based roles, permission checks and blinded context views."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from Research_OS.contracts.common import Permission, validate_stable_id
from Research_OS.providers import AgentRequest, AgentResponse, Provider


class Role(str, Enum):
    RESEARCH_DIRECTOR = "ResearchDirector"
    LITERATURE = "LiteratureAgent"
    DATA = "DataAgent"
    HYPOTHESIS = "HypothesisAgent"
    QUANT_DEVELOPER = "QuantDeveloperAgent"
    STATISTICIAN = "StatisticianAgent"
    ADVERSARIAL_CRITIC = "AdversarialCriticAgent"
    RISK = "RiskAgent"
    REPRODUCIBILITY = "ReproducibilityAgent"


ROLE_PERMISSIONS = {
    Role.RESEARCH_DIRECTOR: frozenset({Permission.RESEARCH}),
    Role.LITERATURE: frozenset({Permission.RESEARCH}),
    Role.DATA: frozenset({Permission.RESEARCH, Permission.IMPLEMENTATION, Permission.VALIDATION}),
    Role.HYPOTHESIS: frozenset({Permission.RESEARCH}),
    Role.QUANT_DEVELOPER: frozenset({Permission.RESEARCH, Permission.IMPLEMENTATION}),
    Role.STATISTICIAN: frozenset({Permission.RESEARCH, Permission.VALIDATION}),
    Role.ADVERSARIAL_CRITIC: frozenset({Permission.RESEARCH, Permission.VALIDATION}),
    Role.RISK: frozenset({Permission.RESEARCH, Permission.VALIDATION, Permission.GOVERNANCE}),
    Role.REPRODUCIBILITY: frozenset({Permission.VALIDATION}),
}


@dataclass(frozen=True)
class AgentIdentity:
    agent_id: str
    role: Role
    provider_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.agent_id, "AGT")
        if Permission.PRODUCTION in ROLE_PERMISSIONS[self.role]:
            raise ValueError("AI roles cannot receive PRODUCTION permission")


@dataclass(frozen=True)
class AgentAudit:
    agent_id: str
    role: str
    provider_id: str
    model_id: str
    context_sha256: str
    output_sha256: str


@dataclass(frozen=True)
class ModelDiversityPolicy:
    enabled: bool = True
    minimum_independent_providers: int = 2
    target_critical_providers: int = 3
    agreement_is_not_validation: bool = True

    def validate(self, provider_ids: list[str], *, critical: bool = False) -> None:
        needed = self.target_critical_providers if critical else self.minimum_independent_providers
        if self.enabled and len(set(provider_ids)) < needed:
            raise ValueError(f"insufficient independent providers: need {needed}")


def validate_separation(*, proposer_agent_id: str, verifier_agent_id: str,
                        implementation_agent_id: str | None = None,
                        reproduction_agent_id: str | None = None) -> None:
    if proposer_agent_id == verifier_agent_id:
        raise ValueError("proposer and verifier must be different agents")
    if implementation_agent_id and reproduction_agent_id and implementation_agent_id == reproduction_agent_id:
        raise ValueError("implementation and reproduction agents must be different")


HIDDEN_BY_VIEW = {
    "developer": frozenset({"desired_sharpe", "desired_return", "hidden_oos_metrics", "expected_conclusion", "primary_run_metrics"}),
    "reproduction": frozenset({"primary_run_metrics", "expected_conclusion", "proposer_confidence"}),
    "critic": frozenset({"desired_conclusion", "proposer_confidence", "want_to_pass"}),
}


def blinded_context(context: dict[str, Any], view: str) -> dict[str, Any]:
    hidden = HIDDEN_BY_VIEW.get(view, frozenset())
    return {key: value for key, value in context.items() if key not in hidden}


class Agent:
    def __init__(self, identity: AgentIdentity, provider: Provider):
        if identity.provider_id != provider.provider_id:
            raise ValueError("agent/provider identity mismatch")
        self.identity = identity
        self.provider = provider

    def require(self, permission: Permission) -> None:
        if permission not in ROLE_PERMISSIONS[self.identity.role]:
            raise PermissionError(f"{self.identity.role.value} lacks {permission.value}")

    def run(self, request: AgentRequest, *, view: str = "critic") -> tuple[AgentResponse, AgentAudit]:
        safe_request = AgentRequest(request_id=request.request_id, role=self.identity.role.value, task=request.task,
                                    context=blinded_context(request.context, view), response_schema=request.response_schema)
        response = self.provider.complete(safe_request)
        if not isinstance(response.payload, dict) or "status" not in response.payload:
            raise ValueError("malformed structured provider output")
        audit = AgentAudit(self.identity.agent_id, self.identity.role.value, response.provider_id,
                           response.model_id, safe_request.context_sha256, response.output_sha256)
        return response, audit
