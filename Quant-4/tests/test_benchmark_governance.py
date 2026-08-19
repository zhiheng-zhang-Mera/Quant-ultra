from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Main.benchmark_governance import BenchmarkCandidate, BenchmarkContext, run_comparable_benchmarks


def _context():
    return BenchmarkContext("2022-01-03", "2023-12-29", "2024-01-02", "prices-sha256:abc", "pit-universe/v2", 100000, .0003, 5, .0005, 100)


def _universe(index):
    rng = np.random.default_rng(19)
    values = {
        BenchmarkCandidate(name, category): pd.Series(rng.normal(.0002, .005, len(index)), index=index)
        for name, category in zip(("buy_hold", "sma", "value", "lightgbm"), ("passive", "classic", "factor", "ml"))
    }
    values[BenchmarkCandidate("quant_ultra", "quant_ultra", "quant_ultra_current", "8-19/v1")] = pd.Series(rng.normal(.0003, .004, len(index)), index=index)
    values[BenchmarkCandidate("quant_ultra_8_16", "quant_ultra", "quant_ultra_historical", "8-16/frozen")] = pd.Series(rng.normal(.00025, .004, len(index)), index=index)
    return values


def test_complete_same_context_benchmark_manifest_is_traceable():
    index = pd.bdate_range("2022-01-03", "2023-12-29")
    result = run_comparable_benchmarks(_universe(index), _context())
    assert result["claim_scope"] == "INTERNAL_COMPARABLE_EVIDENCE"
    assert result["external_rankings"] == "AUXILIARY_REFERENCE_ONLY"
    assert len(result["context_sha256"]) == len(result["manifest_sha256"]) == 64
    assert result["git_commit"]
    assert {row["role"] for row in result["benchmarks"]} >= {"quant_ultra_current", "quant_ultra_historical"}


def test_misaligned_or_incomplete_universe_fails_closed():
    index = pd.bdate_range("2022-01-03", "2023-12-29")
    values = _universe(index)
    first = next(iter(values))
    values[first] = values[first].iloc[1:]
    with pytest.raises(ValueError, match="identical observation index"):
        run_comparable_benchmarks(values, _context())
    incomplete = {key: value for key, value in _universe(index).items() if key.category != "ml"}
    with pytest.raises(ValueError, match="missing benchmark categories"):
        run_comparable_benchmarks(incomplete, _context())
