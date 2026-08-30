from pathlib import Path

import pandas as pd
import pytest

from Research_OS.application import PersistentEventBus, ResearchApplicationService
from Research_OS.contracts.common import LifecycleStatus
from Research_OS.contracts.composite import HoldoutVaultManifest, PolicyManifest
from Research_OS.governance.policy import GovernancePolicy
from Research_OS.orchestration import VERIFICATION_SUBGRAPHS
from Research_OS.verification import (
    HoldoutVault,
    assess_reasoning_independence,
    build_generalization_matrix,
    compare_policy,
    future_mutation_sentinel,
    inspect_source,
    verify_risk_plane,
)


def test_event_bus_is_monotonic_redacted_and_replayable(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    bus = PersistentEventBus(path, recent_limit=2)
    one = bus.publish("RUN-testing", "STARTED", {"token": "unsafe", "nested": {"password": "unsafe"}})
    two = bus.publish("RUN-testing", "DONE", {})
    assert (one.sequence, two.sequence) == (1, 2)
    assert one.payload == {"token": "[REDACTED]", "nested": {"password": "[REDACTED]"}}
    assert [event.event_id for event in PersistentEventBus(path).replay(run_id="RUN-testing")] == [one.event_id, two.event_id]


def test_application_service_validates_paths_and_runs_offline(tmp_path: Path) -> None:
    service = ResearchApplicationService(tmp_path)
    with pytest.raises(ValueError):
        service.create_research({"question": ""})
    with pytest.raises(ValueError):
        service.get_run("../../escape")
    intake = {"question": "Does a preregistered signal persist?"}
    run_id = service.create_research(intake)
    state = service.start_research(run_id, intake)
    assert len(state["results"]) == 21
    assert service.events.replay(run_id=run_id)[-1].event_type == "RUN_COMPLETED"


def test_ast_and_runtime_pit_guards() -> None:
    findings = inspect_source("x = frame['forward_return'].shift(-1)\ny = x.bfill()")
    assert {finding.code for finding in findings} >= {"TARGET_DERIVED_FEATURE", "PIT_NEGATIVE_SHIFT", "PIT_BACKFILL"}
    frame = pd.DataFrame({"value": range(10)})
    safe = future_mutation_sentinel(frame, lambda data: data["value"].rolling(2).mean(), split=5)
    unsafe = future_mutation_sentinel(frame, lambda data: data["value"][::-1].expanding().mean()[::-1], split=5)
    assert safe.passed and not unsafe.passed


def test_holdout_budget_and_roles_are_fail_closed(tmp_path: Path) -> None:
    manifest = HoldoutVaultManifest(HoldoutVaultManifest.SCHEMA, "family", "generation-1", holdout_sha256="a" * 64)
    vault = HoldoutVault(tmp_path / "vault.jsonl")
    with pytest.raises(PermissionError):
        vault.expose(manifest, role="hypothesis_agent", actor="AGT-dev", purpose="tuning")
    exposed = vault.expose(manifest, role="governance", actor="AGT-review", purpose="final review")
    assert exposed.exposure_count == 1 and not exposed.clean_holdout
    with pytest.raises(PermissionError):
        vault.expose(exposed, role="governance", actor="AGT-review", purpose="retry")


def test_reasoning_risk_transfer_and_policy_dimensions_remain_separate() -> None:
    reasoning = assess_reasoning_independence("EXP-testing", (
        {"provider_family": "a", "model_family": "x", "context_hash": "1", "evidence_set_hash": "e1", "peer_conclusions_visible": False},
        {"provider_family": "b", "model_family": "y", "context_hash": "2", "evidence_set_hash": "e2", "peer_conclusions_visible": False},
    ))
    assert reasoning.status == LifecycleStatus.PASS and reasoning.informational_only
    risk = verify_risk_plane("EXP-testing", {"max_drawdown": 1.0, "expected_shortfall": 2.0, "turnover": 3.0,
                                               "concentration": 4.0}, {}, capacity_profile={}, stress_passed=True)
    assert risk.status == LifecycleStatus.PASS
    axes = {"time": "SUPPORTED", "market": "SUPPORTED"}
    matrix = build_generalization_matrix("EXP-testing", "strict", axes, frozen_parameter_hash="frozen",
                                         target_parameter_hashes={"time": "frozen", "market": "changed"},
                                         target_search_trials={"time": 0, "market": 1},
                                         pit_metadata_complete={"time": True, "market": True})
    assert matrix.axes == {"time": "SUPPORTED", "market": "FAILED"}
    old = PolicyManifest(PolicyManifest.SCHEMA, "v1", ("PIT",), ("PIT",), {"alpha": 0.05})
    new = PolicyManifest(PolicyManifest.SCHEMA, "v2", ("PIT", "RISK"), ("PIT", "RISK"), {"alpha": 0.01})
    assert compare_policy(old, new)["changed"]
    assert GovernancePolicy().manifest.policy_hash
    assert {"R2", "R12", "R13", "R16", "R18", "R19"} <= set(VERIFICATION_SUBGRAPHS)
