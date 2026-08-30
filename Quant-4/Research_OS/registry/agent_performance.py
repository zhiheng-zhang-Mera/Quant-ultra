"""Objective agent reliability metrics; profitability is intentionally absent."""
from __future__ import annotations

from pathlib import Path

from .storage import HashChainStore

ALLOWED_METRICS = frozenset({"coding_failure", "schema_failure", "leakage_detected", "false_positive",
                             "reproduction_success", "structured_output_success", "latency_seconds", "cost"})


class AgentPerformanceRegistry:
    def __init__(self, path: str | Path):
        self.store = HashChainStore(path, schema_version="agent-performance/v1")

    def record(self, agent_id: str, metric: str, value: float, *, actor: str, task_type: str) -> dict:
        if metric not in ALLOWED_METRICS:
            raise ValueError("only objectively checkable agent outcomes may be recorded")
        return self.store.append("AGENT_OUTCOME", {"agent_id": agent_id, "metric": metric,
                                                   "value": float(value), "task_type": task_type}, actor=actor)

    def profile(self, agent_id: str) -> dict:
        rows = [row["payload"] for row in self.store.events({"AGENT_OUTCOME"}) if row["payload"]["agent_id"] == agent_id]
        metrics = {}
        for metric in sorted(ALLOWED_METRICS):
            values = [row["value"] for row in rows if row["metric"] == metric]
            if values:
                metrics[metric] = sum(values) / len(values)
        return {"agent_id": agent_id, "metrics": metrics, "profitability_scored": False,
                "may_alter_statistical_thresholds": False}
