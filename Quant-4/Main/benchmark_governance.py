"""Comparable benchmark contracts and deterministic benchmark manifests.

External published rankings are intentionally excluded from this module. They
may be displayed as background, but production claims must use this same-data,
same-cost, same-constraint contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

REQUIRED_CATEGORIES = frozenset({"passive", "classic", "factor", "ml"})
REQUIRED_ROLES = frozenset({"quant_ultra_current", "quant_ultra_historical"})


@dataclass(frozen=True)
class BenchmarkContext:
    data_start: str
    data_end: str
    as_of: str
    data_version: str
    universe_version: str
    initial_capital: float
    commission_rate: float
    minimum_commission: float
    stamp_tax_rate: float
    board_lot: int
    leverage_limit: float = 1.0
    frequency: int = 252

    def validate(self) -> None:
        start, end, as_of = map(pd.Timestamp, (self.data_start, self.data_end, self.as_of))
        if not start < end <= as_of:
            raise ValueError("benchmark dates must satisfy data_start < data_end <= as_of")
        if not self.data_version or not self.universe_version:
            raise ValueError("data and universe versions are required")
        if self.initial_capital <= 0 or self.board_lot <= 0 or self.frequency <= 0:
            raise ValueError("capital, board lot and frequency must be positive")
        if not 0 <= self.commission_rate < 1 or not 0 <= self.stamp_tax_rate < 1:
            raise ValueError("cost rates must be in [0, 1)")
        if self.minimum_commission < 0 or not 0 < self.leverage_limit <= 1:
            raise ValueError("long-only benchmark contract forbids leverage")


@dataclass(frozen=True)
class BenchmarkCandidate:
    name: str
    category: str
    role: str = "benchmark"
    parameter_version: str = "frozen/v1"

    def validate(self) -> None:
        if not self.name or not self.parameter_version:
            raise ValueError("benchmark name and parameter version are required")
        if self.category not in REQUIRED_CATEGORIES | {"quant_ultra"}:
            raise ValueError(f"unsupported benchmark category: {self.category}")


def _sha(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _git_commit(project_root: Path | None) -> str | None:
    cwd = project_root or Path(__file__).parents[1]
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, check=False)
    value = result.stdout.strip()
    return value if result.returncode == 0 and len(value) == 40 else None


def _metrics(values: pd.Series, frequency: int) -> dict:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 2:
        raise ValueError("each benchmark needs at least two finite return observations")
    growth = float((1.0 + clean).prod())
    annual_return = growth ** (frequency / len(clean)) - 1.0 if growth > 0 else -1.0
    std = float(clean.std(ddof=1))
    sharpe = float(clean.mean() / std * math.sqrt(frequency)) if std > 0 else 0.0
    nav = (1.0 + clean).cumprod()
    return {
        "observations": int(len(clean)),
        "annual_return": float(annual_return),
        "sharpe": sharpe,
        "max_drawdown": float((nav / nav.cummax() - 1.0).min()),
        "return_sha256": _sha([round(float(item), 15) for item in clean]),
    }


def run_comparable_benchmarks(
    returns: Mapping[BenchmarkCandidate, pd.Series],
    context: BenchmarkContext,
    *,
    project_root: Path | None = None,
) -> dict:
    """Evaluate a complete internal benchmark universe under one contract."""
    context.validate()
    if not returns:
        raise ValueError("benchmark universe cannot be empty")
    categories, roles, indexes, rows = set(), set(), [], []
    for candidate, series in returns.items():
        candidate.validate()
        if not isinstance(series.index, pd.DatetimeIndex):
            raise TypeError(f"{candidate.name} requires a DatetimeIndex")
        if series.index.has_duplicates or not series.index.is_monotonic_increasing:
            raise ValueError(f"{candidate.name} index must be unique and sorted")
        bounded = series.loc[(series.index >= pd.Timestamp(context.data_start)) & (series.index <= pd.Timestamp(context.data_end))]
        indexes.append(bounded.index)
        categories.add(candidate.category)
        roles.add(candidate.role)
        rows.append({**asdict(candidate), "metrics": _metrics(bounded, context.frequency)})
    if not REQUIRED_CATEGORIES.issubset(categories):
        raise ValueError(f"missing benchmark categories: {sorted(REQUIRED_CATEGORIES - categories)}")
    if not REQUIRED_ROLES.issubset(roles):
        raise ValueError(f"missing Quant-Ultra roles: {sorted(REQUIRED_ROLES - roles)}")
    anchor = indexes[0]
    if any(not anchor.equals(index) for index in indexes[1:]):
        raise ValueError("all core benchmarks must use the identical observation index")
    contract = asdict(context)
    manifest = {
        "schema_version": "comparable-benchmark/v1",
        "claim_scope": "INTERNAL_COMPARABLE_EVIDENCE",
        "external_rankings": "AUXILIARY_REFERENCE_ONLY",
        "git_commit": _git_commit(project_root),
        "context": contract,
        "context_sha256": _sha(contract),
        "benchmarks": sorted(rows, key=lambda row: row["name"]),
    }
    manifest["manifest_sha256"] = _sha(manifest)
    return manifest
