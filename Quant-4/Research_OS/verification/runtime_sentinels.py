"""Metamorphic point-in-time tests independent of strategy implementation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class SentinelResult:
    name: str
    passed: bool
    detail: str


def future_mutation_sentinel(frame: "pd.DataFrame", transform: Callable[["pd.DataFrame"], Any], *, split: int) -> SentinelResult:
    import pandas as pd

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


def duplicate_calendar_sentinel(frame: "pd.DataFrame", transform: Callable[["pd.DataFrame"], Any]) -> SentinelResult:
    import pandas as pd

    duplicated = pd.concat([frame, frame.iloc[[-1]]], axis=0)
    try:
        transform(duplicated)
    except (ValueError, AssertionError):
        return SentinelResult("duplicate_calendar", True, "duplicate timestamp rejected")
    return SentinelResult("duplicate_calendar", False, "transform accepted a duplicate calendar row")


def information_set_invariance(frame: "pd.DataFrame", transform: Callable[["pd.DataFrame"], Any], *,
                               split: int, name: str, mutate: Callable[["pd.DataFrame", int], "pd.DataFrame"]) -> SentinelResult:
    import pandas as pd

    baseline = pd.DataFrame(transform(frame.copy())).iloc[:split].reset_index(drop=True)
    candidate_frame = mutate(frame.copy(), split)
    try:
        candidate = pd.DataFrame(transform(candidate_frame)).iloc[:split].reset_index(drop=True)
        pd.testing.assert_frame_equal(baseline, candidate, check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
    except (AssertionError, ValueError) as exc:
        return SentinelResult(name, False, f"past output changed: {exc}")
    return SentinelResult(name, True, "past output invariant under unavailable-information mutation")


def run_pit_sentinel_suite(frame: "pd.DataFrame", transform: Callable[["pd.DataFrame"], Any], *, split: int,
                           label_columns: tuple[str, ...] = ("label", "target", "forward_return"),
                           publication_column: str = "publication_time") -> tuple[SentinelResult, ...]:
    import pandas as pd

    def permute_future(data: "pd.DataFrame", boundary: int) -> "pd.DataFrame":
        future = data.iloc[boundary:].sample(frac=1.0, random_state=17)
        return pd.concat([data.iloc[:boundary], future], axis=0)

    def mutate_labels(data: "pd.DataFrame", boundary: int) -> "pd.DataFrame":
        for column in label_columns:
            if column in data:
                data.loc[data.index[boundary:], column] = -999999
        return data

    def delay_publication(data: "pd.DataFrame", boundary: int) -> "pd.DataFrame":
        if publication_column in data:
            values = pd.to_datetime(data.loc[data.index[boundary:], publication_column], utc=True)
            data.loc[data.index[boundary:], publication_column] = values + pd.Timedelta(days=365)
        return data

    def revise_fundamentals(data: "pd.DataFrame", boundary: int) -> "pd.DataFrame":
        columns = [column for column in data.select_dtypes(include="number") if column not in label_columns]
        data.loc[data.index[boundary:], columns] = data.loc[data.index[boundary:], columns] * 37 + 11
        return data

    def insert_calendar_gap(data: "pd.DataFrame", boundary: int) -> "pd.DataFrame":
        return data.drop(data.index[boundary + 1:boundary + 2])

    def duplicate_future_event(data: "pd.DataFrame", boundary: int) -> "pd.DataFrame":
        return pd.concat([data, data.iloc[[boundary]]], axis=0)

    def permute_assets(data: "pd.DataFrame", boundary: int) -> "pd.DataFrame":
        if "asset" in data:
            future = data.iloc[boundary:].sort_values("asset", ascending=False)
            return pd.concat([data.iloc[:boundary], future], axis=0)
        return data

    mutations = (
        ("future_row_permutation", permute_future), ("future_label_mutation", mutate_labels),
        ("publication_time_delay", delay_publication), ("revised_fundamental_substitution", revise_fundamentals),
        ("calendar_gap_insertion", insert_calendar_gap), ("duplicate_future_event", duplicate_future_event),
        ("asset_order_permutation", permute_assets),
    )
    return tuple(information_set_invariance(frame, transform, split=split, name=name, mutate=mutation)
                 for name, mutation in mutations)
