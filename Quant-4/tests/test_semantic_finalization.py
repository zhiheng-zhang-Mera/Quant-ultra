from pathlib import Path

import pytest

from Research_OS.application import ExecutionCapabilities, ResearchApplicationService, ResearchExecutionMode
from Research_OS.contracts.common import AdmissionAction, LifecycleStatus
from Research_OS.contracts.composite import VERIFICATION_LEVELS
from Research_OS.governance import AdmissionProfile, GovernancePolicy
from Research_OS.verification import VerificationEvidenceStore, VerificationMatrixBuilder


def test_default_lifecycle_is_demo_and_created_run_survives_restart(tmp_path: Path) -> None:
    service = ResearchApplicationService(tmp_path)
    run_id = service.create_research({"question": "truthful mode?"})
    assert service.execution_mode == ResearchExecutionMode.DEMO_OFFLINE
    restarted = ResearchApplicationService(tmp_path)
    summary = restarted.get_run_summary(run_id)
    assert summary.execution_mode == "DEMO_OFFLINE"
    assert summary.lifecycle_status == "CREATED"
    assert [item.run_id for item in restarted.list_run_summaries()] == [run_id]


def test_real_mode_requires_complete_real_capabilities() -> None:
    partial = ExecutionCapabilities("configured", data_sources_real=True, kernel_real=True)
    complete = ExecutionCapabilities("configured", True, True, True, True)
    assert partial.mode == ResearchExecutionMode.RESEARCH_OFFLINE
    assert complete.mode == ResearchExecutionMode.REAL_RESEARCH


def test_lifecycle_completion_does_not_imply_verification_pass(tmp_path: Path) -> None:
    service = ResearchApplicationService(tmp_path)
    intake = {"question": "does lifecycle success equal evidence?"}
    run_id = service.create_research(intake)
    state = service.start_research(run_id, intake)
    assert state["lifecycle_status"] == "COMPLETED"
    assert state["admission_action"] == "RESEARCH_ONLY"
    assert {row.status for row in service.get_verification_matrix(run_id)} == {"HOLD"}


def test_matrix_is_built_only_from_persisted_evidence_and_survives_restart(tmp_path: Path) -> None:
    store_path = tmp_path / "verification.jsonl"
    store = VerificationEvidenceStore(store_path)
    run_id = "RUN-semantic-test"
    builder = VerificationMatrixBuilder(store)
    assert all(status == LifecycleStatus.HOLD for status in builder.build(run_id).levels.values())
    record = store.append(level=VERIFICATION_LEVELS[1], run_id=run_id, experiment_id="EXP-semantic-test",
                          record_type="source-independence-evidence/v1", record={"origins": 2},
                          status=LifecycleStatus.PASS, producer="test")
    matrix = VerificationMatrixBuilder(VerificationEvidenceStore(store_path)).build(run_id)
    assert matrix.levels[VERIFICATION_LEVELS[1]] == LifecycleStatus.PASS
    assert sum(value == LifecycleStatus.PASS for value in matrix.levels.values()) == 1
    assert record.verification_evidence_id in matrix.evidence_refs[VERIFICATION_LEVELS[1]]


@pytest.mark.parametrize("blocked_level", [VERIFICATION_LEVELS[1], VERIFICATION_LEVELS[4], VERIFICATION_LEVELS[11]])
def test_composite_gate_blocks_missing_or_divergent_critical_level(blocked_level: str, tmp_path: Path) -> None:
    store = VerificationEvidenceStore(tmp_path / f"{blocked_level}.jsonl")
    run_id = "RUN-composite-test"
    for level in VERIFICATION_LEVELS:
        store.append(level=level, run_id=run_id, experiment_id="EXP-composite-test", record_type="test/v1",
                     record={"level": level}, status=LifecycleStatus.REJECT if level == blocked_level else LifecycleStatus.PASS,
                     producer="test")
    matrix = VerificationMatrixBuilder(store).build(run_id)
    decision = GovernancePolicy().decide_composite("EXP-composite-test", matrix,
                                                   profile=AdmissionProfile.CRITICAL_CANDIDATE,
                                                   execution_mode="REAL_RESEARCH")
    assert decision.action == AdmissionAction.REJECTED
    assert blocked_level in " ".join(decision.reasons)


def test_demo_cannot_be_candidate_even_with_all_pass(tmp_path: Path) -> None:
    store = VerificationEvidenceStore(tmp_path / "all-pass.jsonl")
    for level in VERIFICATION_LEVELS:
        store.append(level=level, run_id="RUN-demo-pass", experiment_id="EXP-demo-pass", record_type="test/v1",
                     record={"level": level}, status=LifecycleStatus.PASS, producer="test")
    decision = GovernancePolicy().decide_composite("EXP-demo-pass", VerificationMatrixBuilder(store).build("RUN-demo-pass"),
                                                   execution_mode="DEMO_OFFLINE")
    assert decision.action == AdmissionAction.RESEARCH_ONLY
