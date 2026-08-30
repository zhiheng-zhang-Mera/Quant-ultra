"""Canonical R0-R20 lifecycle and deterministic offline reference handlers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from Research_OS.contracts.common import LifecycleStatus, sha256

from .budget import ResearchBudget
from .dag import DAG, Stage, StageContext, StageResult

STAGE_NAMES = (
    "Research Intake", "Research Memory Retrieval", "Evidence Reconnaissance", "Multi-AI Hypothesis Generation",
    "Novelty / Duplication Gate", "Mechanism / Falsifiability Review", "Feasibility Gate", "Experimental Design",
    "Preregistration / Experiment Budget", "Data Acquisition / Provenance", "PIT Evidence Normalization",
    "Data Quality / PIT Gate", "Blind Implementation", "Static + Independent Verification", "Quant Kernel Execution",
    "Independent Reproduction", "Statistical Validation", "Adversarial Robustness", "External Generalization",
    "Evidence Synthesis + Governance", "Research Memory Update",
)


def canonical_stages(handler_factory: Callable[[str, str], Callable[[StageContext], StageResult]]) -> list[Stage]:
    stages = []
    for number, name in enumerate(STAGE_NAMES):
        stage_id = f"R{number}"
        stages.append(Stage(stage_id, name, () if number == 0 else (f"R{number - 1}",),
                            handler_factory(stage_id, name), max_retries=1 if number in {2, 3, 9} else 0,
                            critical=number not in {1, 2, 20}))
    return stages


def _offline_handler(stage_id: str, name: str):
    def run(context: StageContext) -> StageResult:
        evidence_id = f"EVD-{sha256([context.run_id, stage_id])[:16]}"
        output: dict[str, Any] = {
            "stage": name, "mode": "offline-deterministic", "input_hash": sha256(context.inputs)
        }
        if stage_id == "R19":
            output.update({"action": "RESEARCH_ONLY", "production_activation": False,
                           "reason": "offline demo validates architecture, not market alpha"})
        return StageResult(stage_id, LifecycleStatus.PASS, output, (evidence_id,))
    return run


class ResearchLifecycle:
    def __init__(self, handler_factory: Callable[[str, str], Callable[[StageContext], StageResult]] = _offline_handler):
        self.dag = DAG(canonical_stages(handler_factory))

    def run(self, run_id: str, inputs: dict, *, state_path: str | Path | None = None,
            budget: ResearchBudget | None = None) -> StageContext:
        return self.dag.run(StageContext(run_id, inputs, budget=budget or ResearchBudget()), state_path=state_path)

    def resume(self, state_path: str | Path, *, budget: ResearchBudget | None = None) -> StageContext:
        return self.dag.run(self.dag.resume(state_path, budget), state_path=state_path)
