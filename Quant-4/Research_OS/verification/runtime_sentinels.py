"""Metamorphic point-in-time tests independent of strategy implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class SentinelResult:
    name: str
    passed: bool
    detail: str


def future_mutation_sentinel(frame: pd.DataFrame, transform: Callable[[pd.DataFrame], Any], *, split: int) -> SentinelResult:
    if not 0 < split < len(frame):
        raise ValueError("split must leave past and future rows")
    baseline = transform(frame.copy())
    mutated = frame.copy()
    numeric = mutated.select_dtypes(include="number").columns
    mutated.loc[mutated.index[split:], numeric] = mutated.loc[mutated.index[split:], numeric] * -101.0 + 17.0
    candidate = transform(mutated)
    try:
        left = pd.DataFrame(baseline).iloc[:split].reset_index(drop=True)
        right = pd.DataFrame(candidate).iloc[:split].reset_index(drop=True)
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    except (AssertionError, ValueError) as exc:
        return SentinelResult("future_mutation", False, f"past output changed: {exc}")
    return SentinelResult("future_mutation", True, "past output invariant under future mutation")


def duplicate_calendar_sentinel(frame: pd.DataFrame, transform: Callable[[pd.DataFrame], Any]) -> SentinelResult:
    duplicated = pd.concat([frame, frame.iloc[[-1]]], axis=0)
    try:
        transform(duplicated)
    except (ValueError, AssertionError):
        return SentinelResult("duplicate_calendar", True, "duplicate timestamp rejected")
    return SentinelResult("duplicate_calendar", False, "transform accepted a duplicate calendar row")
