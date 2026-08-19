"""Structured, auditable financial-event representation for LLM research."""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np

EVENT_SCHEMA_VERSION = "financial-event/v1"
EVENT_TYPES = frozenset({"earnings", "guidance", "capital_action", "corporate_action", "credit", "legal", "regulatory", "management", "industry", "macro", "market", "other"})
EVENT_SCOPES = frozenset({"company", "industry", "macro", "regulatory", "market"})
REQUIRED_FIELDS = frozenset({"id", "event_type", "scope", "direction", "importance", "novelty", "uncertainty", "horizon_days", "reliability", "score", "confidence"})


def _probability(value, field: str) -> float:
    number = float(value)
    if not np.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{field} must be in [0, 1]")
    return number


def parse_financial_event_response(raw: str, expected_ids: Iterable[str]) -> dict[str, dict]:
    """Parse a complete event response; incomplete records are not research evidence."""
    payload = json.loads(raw)
    records = payload.get("results")
    if not isinstance(records, list):
        raise ValueError("financial event response requires a results array")
    expected, result = set(expected_ids), {}
    for record in records:
        if not isinstance(record, dict) or not REQUIRED_FIELDS.issubset(record):
            raise ValueError("financial event record is missing required fields")
        record_id = str(record["id"])
        if record_id not in expected or record_id in result:
            raise ValueError("financial event id is unexpected or duplicated")
        event_type, scope = str(record["event_type"]), str(record["scope"])
        if event_type not in EVENT_TYPES or scope not in EVENT_SCOPES:
            raise ValueError("financial event type or scope is invalid")
        direction, score = float(record["direction"]), float(record["score"])
        horizon = int(record["horizon_days"])
        if direction not in {-1.0, 0.0, 1.0} or not np.isfinite(score) or not -1 <= score <= 1:
            raise ValueError("direction must be -1/0/1 and score must be in [-1, 1]")
        if not 1 <= horizon <= 3650:
            raise ValueError("horizon_days must be in [1, 3650]")
        result[record_id] = {
            "event_schema_version": EVENT_SCHEMA_VERSION, "event_type": event_type, "event_scope": scope,
            "event_direction": direction, "event_importance": _probability(record["importance"], "importance"),
            "event_novelty": _probability(record["novelty"], "novelty"),
            "event_uncertainty": _probability(record["uncertainty"], "uncertainty"),
            "event_horizon_days": horizon, "event_reliability": _probability(record["reliability"], "reliability"),
            "llm_sentiment": score, "llm_confidence": _probability(record["confidence"], "confidence"),
        }
    return result


def evaluate_event_representation(events: dict[str, dict], lexical_scores: dict[str, float], requested: int) -> dict:
    scores = [events[key]["llm_sentiment"] for key in events]
    lexical = [float(lexical_scores[key]) for key in events]
    correlation = float(np.corrcoef(scores, lexical)[0, 1]) if len(scores) >= 2 and np.std(scores) > 0 and np.std(lexical) > 0 else None
    difference = float(np.mean(np.abs(np.asarray(scores) - np.asarray(lexical)))) if scores else None
    coverage = len(events) / max(int(requested), 1)
    reliability = float(np.mean([item["event_reliability"] for item in events.values()])) if events else None
    checks = {
        "complete_schema": coverage == 1.0,
        "structured_events": bool(events),
        "reliability": reliability is not None and reliability >= 0.50,
        "independent_signal_validation": False,
    }
    return {
        "schema_version": EVENT_SCHEMA_VERSION,
        "status": "RESEARCH_ONLY",
        "action": "OBSERVATION_ONLY",
        "checks": checks,
        "metrics": {"requested": int(requested), "structured_records": len(events), "coverage": coverage,
                    "mean_reliability": reliability, "lexical_score_correlation": correlation,
                    "mean_absolute_lexical_difference": difference,
                    "representation_sha256": hashlib.sha256(json.dumps(events, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()},
        "reason": "LLM event fields require independent PIT/OOS predictive validation before portfolio use",
    }
