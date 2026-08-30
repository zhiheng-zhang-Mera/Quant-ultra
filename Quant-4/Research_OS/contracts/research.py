"""Research intake, hypothesis and triage contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from .common import Contract, ValidationError, utc_now, validate_stable_id


@dataclass(frozen=True)
class ResearchRequest(Contract):
    SCHEMA: ClassVar[str] = "research-request/v1"
    request_id: str = ""
    title: str = ""
    research_type: str = "factor_research"
    question: str = ""
    constraints: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_stable_id(self.request_id, "REQ")
        if not self.title.strip() or not self.question.strip():
            raise ValidationError("research request title and question are required")


@dataclass(frozen=True)
class Hypothesis(Contract):
    SCHEMA: ClassVar[str] = "hypothesis/v1"
    hypothesis_id: str = ""
    request_id: str = ""
    statement: str = ""
    economic_rationale: str = ""
    expected_effect: str = ""
    expected_regime: str = ""
    counter_hypothesis: str = ""
    failure_conditions: tuple[str, ...] = ()
    falsification_tests: tuple[str, ...] = ()
    parent_hypothesis_id: str | None = None
    proposer_agent_id: str = ""
    created_at: datetime = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        super().__post_init__()
        validate_stable_id(self.hypothesis_id, "HYP")
        validate_stable_id(self.request_id, "REQ")
        validate_stable_id(self.proposer_agent_id, "AGT")
        if self.parent_hypothesis_id:
            validate_stable_id(self.parent_hypothesis_id, "HYP")
        required = (self.statement, self.economic_rationale, self.expected_effect, self.counter_hypothesis)
        if not all(str(value).strip() for value in required) or not self.falsification_tests:
            raise ValidationError("hypothesis is not mechanistic and falsifiable")


@dataclass(frozen=True)
class MechanismReview(Contract):
    SCHEMA: ClassVar[str] = "mechanism-review/v1"
    hypothesis_id: str = ""
    reviewer_agent_id: str = ""
    falsifiable: bool = False
    mechanism_supported: bool = False
    reasons: tuple[str, ...] = ()
    disagreements: tuple[str, ...] = ()


@dataclass(frozen=True)
class FeasibilityAssessment(Contract):
    SCHEMA: ClassVar[str] = "feasibility-assessment/v1"
    hypothesis_id: str = ""
    status: str = "INFEASIBLE"
    data_available: bool = False
    pit_feasible: bool = False
    sample_size: int = 0
    licensing_notes: str = ""
    compute_estimate: str = ""
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.status not in {"FEASIBLE", "RESEARCH_LIMITED", "INFEASIBLE"}:
            raise ValidationError("invalid feasibility status")
