"""Typed application commands; handlers remain in ResearchApplicationService."""
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CreateResearch:
    intake: dict[str, Any]
    run_id: str | None = None


@dataclass(frozen=True)
class RunCommand:
    run_id: str


@dataclass(frozen=True)
class StartResearch(RunCommand):
    intake: dict[str, Any] = field(default_factory=dict)


class ResumeResearch(RunCommand):
    pass


class CancelResearch(RunCommand):
    pass


class LockExperiment(RunCommand):
    pass


class RunKernel(RunCommand):
    pass


class RunReproduction(RunCommand):
    pass


class RunIndependentImplementation(RunCommand):
    pass


class RunRobustness(RunCommand):
    pass


class RunGeneralization(RunCommand):
    pass


class ExportReport(RunCommand):
    pass
