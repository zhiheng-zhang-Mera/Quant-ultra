"""Persistable deterministic DAG with bounded retries and fail-closed edges."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from Research_OS.contracts.common import LifecycleStatus, canonical_json

from .budget import ResearchBudget


@dataclass(frozen=True)
class StageResult:
    stage_id: str
    status: LifecycleStatus
    output: dict[str, Any] = field(default_factory=dict)
    evidence_refs: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass
class StageContext:
    run_id: str
    inputs: dict[str, Any]
    results: dict[str, StageResult] = field(default_factory=dict)
    budget: ResearchBudget = field(default_factory=ResearchBudget)
    cancelled: bool = False


@dataclass(frozen=True)
class Stage:
    stage_id: str
    name: str
    dependencies: tuple[str, ...]
    handler: Callable[[StageContext], StageResult]
    max_retries: int = 0
    critical: bool = True


class DAG:
    def __init__(self, stages: list[Stage]):
        self.stages = {stage.stage_id: stage for stage in stages}
        if len(self.stages) != len(stages):
            raise ValueError("duplicate stage id")
        for stage in stages:
            unknown = set(stage.dependencies) - set(self.stages)
            if unknown:
                raise ValueError(f"unknown dependencies for {stage.stage_id}: {sorted(unknown)}")
        self.order = self._topological_order()

    def _topological_order(self) -> tuple[str, ...]:
        remaining = {key: set(stage.dependencies) for key, stage in self.stages.items()}
        order: list[str] = []
        while remaining:
            ready = sorted(key for key, deps in remaining.items() if not deps)
            if not ready:
                raise ValueError("DAG contains a cycle")
            for key in ready:
                order.append(key)
                remaining.pop(key)
                for deps in remaining.values():
                    deps.discard(key)
        return tuple(order)

    def run(self, context: StageContext, *, state_path: str | Path | None = None) -> StageContext:
        for stage_id in self.order:
            if stage_id in context.results:
                continue
            stage = self.stages[stage_id]
            if context.cancelled:
                context.results[stage_id] = StageResult(stage_id, LifecycleStatus.SKIPPED, reasons=("run cancelled",))
                continue
            blockers = [
                dep for dep in stage.dependencies
                if context.results[dep].status in {LifecycleStatus.REJECT, LifecycleStatus.FAILED}
                or (context.results[dep].status == LifecycleStatus.HOLD and self.stages[dep].critical)
            ]
            if blockers:
                context.results[stage_id] = StageResult(stage_id, LifecycleStatus.SKIPPED,
                                                        reasons=(f"blocked by {','.join(blockers)}",))
                self._persist(context, state_path)
                continue
            result = None
            for attempt in range(stage.max_retries + 1):
                try:
                    context.budget.consume(requests=1)
                    result = stage.handler(context)
                    if result.stage_id != stage_id:
                        raise ValueError("stage returned mismatched id")
                    break
                except Exception as exc:
                    if attempt == stage.max_retries:
                        result = StageResult(stage_id, LifecycleStatus.HOLD if not stage.critical else LifecycleStatus.FAILED,
                                             reasons=(f"{type(exc).__name__}: {exc}",))
            if result is None:
                raise RuntimeError(f"stage {stage_id} produced no result")
            context.results[stage_id] = result
            self._persist(context, state_path)
        return context

    @staticmethod
    def _persist(context: StageContext, path: str | Path | None) -> None:
        if path is None:
            return
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": "research-run-state/v1", "run_id": context.run_id,
                   "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                   "inputs": context.inputs,
                   "cancelled": context.cancelled,
                   "results": {key: {"stage_id": value.stage_id, "status": value.status.value,
                                     "output": value.output, "evidence_refs": value.evidence_refs,
                                     "reasons": value.reasons} for key, value in context.results.items()}}
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(canonical_json(payload), encoding="utf-8")
        temporary.replace(target)

    @staticmethod
    def resume(path: str | Path, budget: ResearchBudget | None = None) -> StageContext:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        results = {key: StageResult(value["stage_id"], LifecycleStatus(value["status"]), value["output"],
                                    tuple(value["evidence_refs"]), tuple(value["reasons"]))
                   for key, value in payload["results"].items()}
        return StageContext(payload["run_id"], payload["inputs"], results,
                            budget or ResearchBudget(), payload.get("cancelled", False))
