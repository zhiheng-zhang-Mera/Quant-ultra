"""Fail-closed governance gates for alternative-signal evidence."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def evaluate_signal_panel(
    panel: pd.DataFrame,
    symbols: Iterable[str],
    provenance: dict | None,
    *,
    min_symbol_coverage: float = 0.80,
    max_missing_rate: float = 0.20,
    max_source_latency_hours: float = 24.0,
    allow_fallback: bool = False,
) -> dict:
    """Evaluate a derived signal panel before an enabled engine may use it."""
    symbols = list(symbols)
    numeric = panel.reindex(columns=symbols).apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    finite_mask = np.isfinite(values)
    finite_values = values[finite_mask]
    nonempty_rows = numeric.notna().any(axis=1)
    active = numeric.loc[nonempty_rows]
    total_cells = int(active.shape[0] * len(symbols))
    observed_cells = int(active.notna().sum().sum()) if total_cells else 0
    missing_rate = 1.0 - observed_cells / total_cells if total_cells else 1.0
    per_row_coverage = active.notna().sum(axis=1) / max(len(symbols), 1)
    symbol_coverage = float(per_row_coverage.min()) if len(per_row_coverage) else 0.0
    provenance = provenance or {}
    latency = provenance.get("max_source_latency_hours")
    latency = float(latency) if latency is not None else float("inf")
    fallback_used = bool(provenance.get("fallback_used", True))
    checks = {
        "provenance_present": bool(provenance),
        "finite_values": bool(finite_mask.all()) if values.size else False,
        "bounded_values": bool(len(finite_values) and (finite_values >= -1.0).all() and (finite_values <= 1.0).all()),
        "symbol_coverage": symbol_coverage >= float(min_symbol_coverage),
        "missing_rate": missing_rate <= float(max_missing_rate),
        "source_latency": np.isfinite(latency) and latency <= float(max_source_latency_hours),
        "fallback_policy": allow_fallback or not fallback_used,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    passed = not reasons
    return {
        "passed": passed,
        "status": "PASS" if passed else "HOLD_FOR_REVIEW",
        "action": "RESEARCH_ELIGIBLE" if passed else "OBSERVATION_ONLY",
        "checks": checks,
        "reasons": reasons,
        "metrics": {
            "rows_with_signal": int(nonempty_rows.sum()),
            "symbol_coverage_min": symbol_coverage,
            "missing_rate": missing_rate,
            "max_source_latency_hours": latency if np.isfinite(latency) else None,
            "fallback_used": fallback_used,
        },
        "thresholds": {
            "min_symbol_coverage": float(min_symbol_coverage),
            "max_missing_rate": float(max_missing_rate),
            "max_source_latency_hours": float(max_source_latency_hours),
            "allow_fallback": bool(allow_fallback),
        },
    }


def evaluate_phase3_signal(
    result: pd.DataFrame,
    assets: Iterable[str],
    source_evidence: Iterable[dict],
    llm_evidence: dict,
    config: dict,
) -> dict:
    """Evaluate Phase 3 output even when optional sources are absent."""
    assets = list(assets)
    numeric = pd.to_numeric(result.get("alternative_signal", pd.Series(dtype=float)), errors="coerce")
    finite = bool(len(numeric) and np.isfinite(numeric.to_numpy(dtype=float)).all())
    bounded = bool(finite and numeric.between(-1.0, 1.0).all())
    loaded = [item for item in source_evidence if item.get("status") == "LOADED"]
    covered = set()
    for item in loaded:
        covered.update(item.get("covered_symbols", []))
    coverage = len(covered & set(assets)) / max(len(assets), 1)
    latencies = [item.get("max_source_latency_hours") for item in loaded]
    latencies = [float(value) for value in latencies if value is not None]
    max_latency = max(latencies) if latencies else None
    llm_enabled = bool(config.get("local_llm_sentiment_enabled", True))
    fallback_used = bool(llm_enabled and llm_evidence.get("fallback_used", True))
    checks = {
        "finite_values": finite,
        "bounded_values": bounded,
        "source_available": bool(loaded),
        "symbol_coverage": coverage >= float(config.get("alternative_signal_min_coverage", 0.80)),
        "source_latency": max_latency is not None and max_latency <= float(config.get("alternative_signal_max_latency_hours", 24.0)),
        "fallback_policy": bool(config.get("alternative_signal_allow_fallback", False)) or not fallback_used,
    }
    reasons = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not reasons,
        "status": "PASS" if not reasons else "HOLD_FOR_REVIEW",
        "action": "RESEARCH_ELIGIBLE" if not reasons else "OBSERVATION_ONLY",
        "checks": checks,
        "reasons": reasons,
        "metrics": {
            "symbol_coverage": coverage,
            "max_source_latency_hours": max_latency,
            "fallback_used": fallback_used,
        },
    }
