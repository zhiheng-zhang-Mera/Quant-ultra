"""Deterministic performance baselines with financial-result fingerprints."""
from __future__ import annotations

import hashlib
import json
import time
import tracemalloc
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

PERFORMANCE_BUDGETS: dict[str, dict[str, float | int | str]] = {
    "full_pool_factor": {"assets": 5478, "days": 756, "max_seconds": 5.0, "max_peak_mib": 512.0,
                         "result_sha256": "4f156bfd17577b7957f1e9d60e90faa5e2480adbf58730008c6e36b9824dcfe6"},
    "alternative_events": {"events": 12066, "max_seconds": 2.0, "max_peak_mib": 256.0,
                           "result_sha256": "d4b48e8b6ef23e43ca7c070ef23609f77c35aefe260cad4ad5ff0c014a5adc19"},
}


def _fingerprint(values: np.ndarray) -> str:
    rounded = np.round(np.asarray(values, dtype=np.float64), 12)
    return hashlib.sha256(rounded.tobytes(order="C")).hexdigest()


def _measure(operation: Callable[[], np.ndarray]) -> tuple[np.ndarray, float, float]:
    tracemalloc.start()
    started = time.perf_counter()
    values = operation()
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return values, elapsed, peak / (1024 * 1024)


def benchmark_full_pool(*, assets: int = 5478, days: int = 756, seed: int = 819) -> dict:
    def operation() -> np.ndarray:
        rng = np.random.default_rng(seed)
        returns = rng.normal(0.0002, 0.02, size=(days, assets)).astype(np.float64)
        momentum = np.prod(1.0 + returns[-126:], axis=0) - 1.0
        ranks = pd.Series(momentum).rank(method="average", pct=True).to_numpy()
        return ranks
    values, elapsed, peak = _measure(operation)
    budget = PERFORMANCE_BUDGETS["full_pool_factor"]
    result_hash = _fingerprint(values)
    expected_hash = budget["result_sha256"] if assets == budget["assets"] and days == budget["days"] else result_hash
    passed = elapsed <= float(budget["max_seconds"]) and peak <= float(budget["max_peak_mib"]) and result_hash == expected_hash
    return {"status": "PASS" if passed else "REGRESSION", "seconds": elapsed, "peak_mib": peak,
            "assets": assets, "days": days, "result_sha256": result_hash, "expected_result_sha256": expected_hash, "budget": budget}


def benchmark_alternative_events(*, events: int = 12066, seed: int = 819) -> dict:
    def operation() -> np.ndarray:
        rng = np.random.default_rng(seed)
        frame = pd.DataFrame({"symbol": rng.integers(0, 1000, events), "score": rng.normal(size=events),
                              "confidence": rng.uniform(0.0, 1.0, events)})
        weighted = frame.assign(weighted=frame["score"] * frame["confidence"])
        return weighted.groupby("symbol", sort=True)["weighted"].mean().to_numpy()
    values, elapsed, peak = _measure(operation)
    budget = PERFORMANCE_BUDGETS["alternative_events"]
    result_hash = _fingerprint(values)
    expected_hash = budget["result_sha256"] if events == budget["events"] else result_hash
    passed = elapsed <= float(budget["max_seconds"]) and peak <= float(budget["max_peak_mib"]) and result_hash == expected_hash
    return {"status": "PASS" if passed else "REGRESSION", "seconds": elapsed, "peak_mib": peak,
            "events": events, "result_sha256": result_hash, "expected_result_sha256": expected_hash, "budget": budget}


def run_performance_gate(output: Path | None = None) -> dict:
    results = {"full_pool_factor": benchmark_full_pool(), "alternative_events": benchmark_alternative_events()}
    payload = {"schema_version": "performance-gate/v1",
               "status": "PASS" if all(item["status"] == "PASS" for item in results.values()) else "HOLD",
               "results": results,
               "boundary": "Synthetic engineering baseline; not an investment-performance result."}
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
