"""Typed query messages for stable application read APIs."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RunQuery:
    run_id: str


class ListRuns:
    pass


class GetRunSummary(RunQuery):
    pass


class GetLifecycleTopology(RunQuery):
    pass


class GetLifecycleState(RunQuery):
    pass


class GetVerificationMatrix(RunQuery):
    pass


class GetEvidenceGraph(RunQuery):
    pass


class GetExperiment(RunQuery):
    pass


class GetGovernance(RunQuery):
    pass


class GetMemoryResults(RunQuery):
    pass


class GetProviderHealth(RunQuery):
    pass


class GetKernelState(RunQuery):
    pass


class GetReportManifest(RunQuery):
    pass
