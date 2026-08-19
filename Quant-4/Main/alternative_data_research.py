"""Unified quality, PIT, coverage and conflict research across alternative sources."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping

import numpy as np
import pandas as pd

REQUIRED_TEXT_SOURCES = ("news", "announcement", "forum")
RESEARCH_SPEC = {
    "version": "alternative-data-research/v1", "required_sources": list(REQUIRED_TEXT_SOURCES),
    "min_source_coverage": 0.50, "max_latency_hours": 72.0,
    "max_pit_violations": 0, "max_conflict_rate": 0.50,
}


def _cutoff(as_of) -> pd.Timestamp:
    value = pd.Timestamp(as_of)
    return value.tz_localize("UTC") if value.tzinfo is None else value.tz_convert("UTC")


def evaluate_alternative_sources(
    frames: Mapping[str, pd.DataFrame], evidence: Mapping[str, dict], assets,
    as_of, *, previous_manifest: dict | None = None,
) -> dict:
    """Evaluate source quality while distinguishing missing data from no event."""
    assets = {str(asset).upper() for asset in assets}
    cutoff = _cutoff(as_of)
    source_rows, signed = {}, []
    for source_name in REQUIRED_TEXT_SOURCES:
        frame = frames.get(source_name, pd.DataFrame()).copy()
        item = evidence.get(source_name, {})
        status = item.get("status", "MISSING_OPTIONAL_SOURCE")
        state = "DATA_MISSING" if status != "LOADED" else ("NO_ELIGIBLE_EVENT" if frame.empty else "OBSERVED_EVENTS")
        violations = 0
        if not frame.empty:
            published = pd.to_datetime(frame.get("published_at"), utc=True, errors="coerce")
            ingested = pd.to_datetime(frame.get("ingested_at"), utc=True, errors="coerce")
            violations = int(((published > cutoff) | (ingested > cutoff) | (ingested < published) | published.isna() | ingested.isna()).sum())
            score_column = "effective_sentiment" if "effective_sentiment" in frame else "sentiment"
            scores = pd.to_numeric(frame.get(score_column), errors="coerce")
            for index in frame.index:
                if pd.notna(scores.loc[index]):
                    signed.append({"source": source_name, "symbol": str(frame.loc[index, "symbol"]),
                                   "day": published.loc[index].floor("D"), "score": float(scores.loc[index])})
        covered = set(frame.get("symbol", pd.Series(dtype=str)).astype(str).str.upper()) & assets
        coverage = len(covered) / max(len(assets), 1)
        latency = item.get("max_source_latency_hours")
        latency = float(latency) if latency is not None else None
        reliability = coverage
        if latency is None or latency > RESEARCH_SPEC["max_latency_hours"]:
            reliability *= 0.5
        if violations:
            reliability = 0.0
        source_rows[source_name] = {
            "state": state, "status": status, "records": int(len(frame)), "covered_symbols": len(covered),
            "coverage": coverage, "pit_violations": violations, "max_latency_hours": latency,
            "reliability_score": float(reliability), "source_sha256": item.get("sha256"),
        }
    conflicts, comparable = 0, 0
    if signed:
        signed_frame = pd.DataFrame(signed)
        for _, group in signed_frame.groupby(["symbol", "day"]):
            if group["source"].nunique() >= 2:
                comparable += 1
                conflicts += int(group["score"].min() < -0.05 and group["score"].max() > 0.05)
    conflict_rate = conflicts / comparable if comparable else 0.0
    prior_hashes = (previous_manifest or {}).get("source_hashes", {})
    current_hashes = {name: row["source_sha256"] for name, row in source_rows.items()}
    source_changes = sorted(name for name, digest in current_hashes.items() if name in prior_hashes and prior_hashes[name] != digest)
    checks = {
        "required_sources": all(row["status"] == "LOADED" for row in source_rows.values()),
        "pit_integrity": sum(row["pit_violations"] for row in source_rows.values()) <= RESEARCH_SPEC["max_pit_violations"],
        "coverage": all(row["coverage"] >= RESEARCH_SPEC["min_source_coverage"] for row in source_rows.values()),
        "latency": all(row["max_latency_hours"] is not None and row["max_latency_hours"] <= RESEARCH_SPEC["max_latency_hours"] for row in source_rows.values()),
        "source_version_stable": not source_changes,
        "conflict_rate": conflict_rate <= RESEARCH_SPEC["max_conflict_rate"],
    }
    reasons = [name for name, passed in checks.items() if not passed]
    payload = {"spec": RESEARCH_SPEC, "as_of": cutoff.isoformat(), "sources": source_rows,
               "source_hashes": current_hashes, "source_changes": source_changes,
               "conflicts": {"comparable_events": comparable, "conflicting_events": conflicts, "conflict_rate": conflict_rate}}
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
    return {"status": "PASS" if not reasons else "HOLD", "action": "RESEARCH_ELIGIBLE" if not reasons else "OBSERVATION_ONLY",
            "checks": checks, "reasons": reasons, **payload}
