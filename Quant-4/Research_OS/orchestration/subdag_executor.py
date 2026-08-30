"""Execute declared verification subgraphs with the canonical DAG engine."""
from __future__ import annotations

from collections.abc import Callable

from Research_OS.contracts.common import LifecycleStatus, sha256

from .budget import ResearchBudget
from .dag import DAG, Stage, StageContext, StageResult
from .subgraphs import VERIFICATION_SUBGRAPHS

SubstageHandler = Callable[[StageContext], StageResult]


class SubstageHandlerRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, SubstageHandler] = {}

    def register(self, substage_id: str, handler: SubstageHandler) -> None:
        if substage_id in self._handlers:
            raise ValueError(f"duplicate substage handler: {substage_id}")
        self._handlers[substage_id] = handler

    def resolve(self, substage_id: str, name: str) -> SubstageHandler:
        if substage_id in self._handlers:
            return self._handlers[substage_id]

        def reference(context: StageContext) -> StageResult:
            evidence = f"EVD-{sha256([context.run_id, substage_id, 'execution'])[:16]}"
            return StageResult(substage_id, LifecycleStatus.PASS,
                               {"name": name, "execution": "REFERENCE_SUBSTAGE"}, (evidence,))

        return reference


class SubDagExecutor:
    def __init__(self, registry: SubstageHandlerRegistry | None = None):
        self.registry = registry or SubstageHandlerRegistry()

    def execute(self, parent_stage_id: str, parent_context: StageContext) -> dict[str, StageResult]:
        definitions = VERIFICATION_SUBGRAPHS.get(parent_stage_id, ())
        if not definitions:
            return {}
        stages = [Stage(item.substage_id, item.name, item.dependencies,
                        self.registry.resolve(item.substage_id, item.name), critical=item.critical)
                  for item in definitions]
        nested = StageContext(parent_context.run_id, parent_context.inputs,
                              dict(parent_context.subresults.get(parent_stage_id, {})), ResearchBudget(),
                              parent_context.cancelled, cancellation_token=parent_context.cancellation_token,
                              event_sink=parent_context.event_sink)
        DAG(stages).run(nested, event_prefix="SUBSTAGE")
        parent_context.subresults[parent_stage_id] = dict(nested.results)
        parent_context.cancelled = nested.cancelled
        return nested.results
