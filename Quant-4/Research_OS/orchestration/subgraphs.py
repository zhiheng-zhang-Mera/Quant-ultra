"""Auditable internal sub-DAGs for composite verification stages.

The public lifecycle remains R0-R20. These definitions expose the independent
evidence units that must complete inside the verification-heavy stages.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubstageDefinition:
    substage_id: str
    name: str
    dependencies: tuple[str, ...] = ()
    critical: bool = True


VERIFICATION_SUBGRAPHS = {
    "R2": (
        SubstageDefinition("R2.S1", "source lineage"),
        SubstageDefinition("R2.S2", "near-duplicate clustering", ("R2.S1",)),
        SubstageDefinition("R2.S3", "cross-vendor reconciliation", ("R2.S1",)),
        SubstageDefinition("R2.S4", "source independence gate", ("R2.S2", "R2.S3")),
    ),
    "R12": (
        SubstageDefinition("R12.S1", "materialize workspace A"),
        SubstageDefinition("R12.S2", "materialize workspace B"),
        SubstageDefinition("R12.S3", "blind implementation A", ("R12.S1",)),
        SubstageDefinition("R12.S4", "blind implementation B", ("R12.S2",)),
    ),
    "R13": (
        SubstageDefinition("R13.S1", "AST and dataflow safety"),
        SubstageDefinition("R13.S2", "runtime PIT sentinels"),
        SubstageDefinition("R13.S3", "L1-L6 semantic comparison"),
        SubstageDefinition("R13.S4", "implementation independence gate", ("R13.S1", "R13.S2", "R13.S3")),
    ),
    "R16": (
        SubstageDefinition("R16.S1", "experiment-family registry"),
        SubstageDefinition("R16.S2", "multiplicity correction", ("R16.S1",)),
        SubstageDefinition("R16.S3", "dependence-aware confidence intervals", ("R16.S1",)),
        SubstageDefinition("R16.S4", "multi-benchmark validation", ("R16.S1",)),
        SubstageDefinition("R16.S5", "statistical decision", ("R16.S2", "R16.S3", "R16.S4")),
    ),
    "R18": tuple(SubstageDefinition(f"R18.S{index + 1}", f"transfer axis: {axis}") for index, axis in enumerate(
        ("time", "market", "sector", "liquidity", "capital", "regime", "execution", "vendor"))),
    "R19": (
        SubstageDefinition("R19.S1", "composite verification matrix"),
        SubstageDefinition("R19.S2", "policy hash and diff"),
        SubstageDefinition("R19.S3", "dissent preservation"),
        SubstageDefinition("R19.S4", "human authorization", ("R19.S1", "R19.S2", "R19.S3")),
    ),
}


def validate_subgraphs() -> None:
    for parent, substages in VERIFICATION_SUBGRAPHS.items():
        ids = {stage.substage_id for stage in substages}
        if len(ids) != len(substages):
            raise ValueError(f"duplicate substage in {parent}")
        for stage in substages:
            unknown = set(stage.dependencies) - ids
            if unknown:
                raise ValueError(f"unknown dependencies for {stage.substage_id}: {sorted(unknown)}")


validate_subgraphs()
