import subprocess
import sys
from pathlib import Path
from threading import Event, Thread

import pandas as pd
import pytest

from Research_OS.application import (
    ExecutionCapabilities,
    PersistentEventBus,
    ResearchApplicationService,
    ResearchExecutionMode,
    ResearchProcessManager,
    RunStateCorruptionError,
)
from Research_OS.contracts.common import AdmissionAction, LifecycleStatus
from Research_OS.contracts.composite import VERIFICATION_LEVELS, IndependentImplementationManifest, ResearchGeneration
from Research_OS.contracts.evidence import EvidenceRecord, SourceRecord
from Research_OS.evidence.independence import SourceIndependenceAnalyzer
from Research_OS.governance import AdmissionProfile, GovernanceDecisionStore, GovernancePolicy
from Research_OS.orchestration import ResearchLifecycle, StageContext, StageResult, SubstageHandlerRegistry
from Research_OS.verification import (
    HoldoutVault,
    VerificationEvidenceStore,
    VerificationMatrixBuilder,
    run_pit_sentinel_suite,
    verify_risk_plane,
)
from Research_OS.verification.environment import WorkspaceMaterializer, capture_environment
from Research_OS.verification.independent_implementation import STRICT_INDEPENDENCE_PROFILE, compare_implementations


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


def test_gitless_materialization_keeps_source_provenance_and_detects_mutation(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "requirements.txt").write_text("PyYAML==6.0.3\n", encoding="utf-8")
    (source / "strategy.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=source, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=source, check=True)
    materializer = WorkspaceMaterializer()
    snapshot = materializer.capture_source_snapshot(source, spec_hash="spec", data_manifest_hash="data")
    workspace_a = materializer.materialize(source, tmp_path / "a")
    workspace_b = materializer.materialize(source, tmp_path / "b")
    manifest_a = materializer.describe_workspace(workspace_a, snapshot)
    manifest_b = materializer.describe_workspace(workspace_b, snapshot)
    assert manifest_a.workspace_content_tree_hash == manifest_b.workspace_content_tree_hash
    assert manifest_a.workspace_id != manifest_b.workspace_id
    environment = capture_environment(workspace_a, experiment_spec_hash="spec", data_manifest_hash="data",
                                      random_seeds={"python": 1}, source_snapshot=snapshot,
                                      workspace_manifest=manifest_a)
    assert environment.git_state == "CLEAN" and environment.production_candidate_eligible
    (workspace_b / "strategy.py").write_text("VALUE = 2\n", encoding="utf-8")
    mutated = materializer.describe_workspace(workspace_b, snapshot)
    assert mutated.workspace_content_tree_hash != manifest_b.workspace_content_tree_hash
    (source / "strategy.py").write_text("VALUE = 3\n", encoding="utf-8")
    assert materializer.capture_source_snapshot(source, spec_hash="spec", data_manifest_hash="data").source_git_state == "DIRTY"


def test_unknown_source_lineage_is_not_trusted(tmp_path: Path) -> None:
    source = SourceRecord(SourceRecord.SCHEMA, "SRC-child", "child", "news", "1", 0.9, "hash", "test",
                          upstream_source_id="SRC-unknown", publisher="publisher")
    evidence = EvidenceRecord(EvidenceRecord.SCHEMA, "EVD-record", source.source_id, "asset", "event", "claim",
                              publish_time=pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime(),
                              availability_time=pd.Timestamp("2025-01-01", tz="UTC").to_pydatetime())
    result = SourceIndependenceAnalyzer().analyze("EXP-source", [source], [evidence], minimum_origins=1)
    assert result.status == LifecycleStatus.HOLD
    assert result.independent_origin_count == 0
    assert result.lineage_resolution_states[source.source_id] == "UNRESOLVED_UPSTREAM"


def _implementation(identifier: str, *, agent: str, workspace: str, provider: str, model: str, code: str) -> IndependentImplementationManifest:
    return IndependentImplementationManifest(IndependentImplementationManifest.SCHEMA, identifier, "EXP-twin", "spec",
                                             agent, provider, model, workspace, "workspace-hash", code, False, False)


def test_typed_independent_comparators_and_strict_profile() -> None:
    left = _implementation("impl-a", agent="AGT-one", workspace="RUN-work-a", provider="p1", model="m1", code="a")
    right = _implementation("impl-b", agent="AGT-two", workspace="RUN-work-b", provider="p2", model="m2", code="b")
    outputs = {"schema": {"columns": ["x"]}, "factors": {"A": [1.0]}, "weights": {"A": 0.5},
               "trades": [{"timestamp": "t", "asset": "A", "side": "BUY", "quantity": 1.0}],
               "accounting": {"nav": 100.0, "cash": 50.0, "positions_value": 50.0, "fees": 0.0, "turnover": 1.0},
               "metrics": {"sharpe": 1.0}}
    confirmed = compare_implementations(left, right, outputs, outputs, profile=STRICT_INDEPENDENCE_PROFILE)
    assert confirmed.status == "INDEPENDENTLY_CONFIRMED"
    divergent = dict(outputs)
    divergent["accounting"] = {**outputs["accounting"], "nav": 101.0}
    result = compare_implementations(left, right, outputs, divergent, profile=STRICT_INDEPENDENCE_PROFILE)
    assert not result.level_results["L5_ACCOUNTING"] and result.status == "DIVERGENT_IMPLEMENTATION"


def test_risk_counterfactual_generalization_and_holdout_generation_are_fail_closed(tmp_path: Path) -> None:
    protected = {"max_drawdown": 1.0, "expected_shortfall": 1.0, "turnover": 1.0, "concentration": 1.0}
    risk = verify_risk_plane("EXP-risk", protected, {}, capacity_profile={}, stress_passed=True)
    assert risk.status == LifecycleStatus.HOLD
    assert any(item.startswith("counterfactual:") for item in risk.critical_failures)
    vault = HoldoutVault(tmp_path / "generation.jsonl")
    generation = ResearchGeneration(ResearchGeneration.SCHEMA, "generation-1", None, "family", True, 0, "INITIAL")
    exposed = vault.expose_generation(generation, actor="reviewer", purpose="final")
    assert not exposed.clean_holdout
    child = vault.child_generation(exposed, generation_id="generation-2", reason="post-holdout tuning")
    assert child.clean_holdout and child.parent_generation_id == exposed.generation_id


def test_expanded_pit_suite_preserves_past_for_causal_transform() -> None:
    frame = pd.DataFrame({"value": range(10), "label": range(10), "asset": list("AABBCCDDEE"),
                          "publication_time": pd.date_range("2025-01-01", periods=10, tz="UTC")})
    results = run_pit_sentinel_suite(frame, lambda data: data[["value"]].expanding().mean(), split=5)
    assert len(results) >= 7 and all(result.passed for result in results)


def test_stage_started_is_persisted_before_handler_finishes(tmp_path: Path) -> None:
    entered, release = Event(), Event()

    def factory(stage_id: str, _name: str):
        def handler(_context: StageContext) -> StageResult:
            if stage_id == "R0":
                entered.set()
                assert release.wait(5)
            return StageResult(stage_id, LifecycleStatus.PASS)
        return handler

    service = ResearchApplicationService(tmp_path, lifecycle=ResearchLifecycle(factory))
    intake = {"question": "are events live?"}
    run_id = service.create_research(intake)
    worker = Thread(target=service.start_research, args=(run_id, intake), daemon=True)
    worker.start()
    assert entered.wait(5)
    types = [event.event_type for event in service.events.replay(run_id=run_id)]
    assert "STAGE_STARTED" in types and "STAGE_COMPLETED" not in types
    release.set()
    worker.join(10)
    assert not worker.is_alive()


def test_cancel_active_run_stops_downstream_and_never_completes_governance(tmp_path: Path) -> None:
    entered, release = Event(), Event()

    def factory(stage_id: str, _name: str):
        def handler(_context: StageContext) -> StageResult:
            if stage_id == "R0":
                entered.set()
                assert release.wait(5)
            return StageResult(stage_id, LifecycleStatus.PASS)
        return handler

    service = ResearchApplicationService(tmp_path, lifecycle=ResearchLifecycle(factory))
    intake = {"question": "does cancel stop execution?"}
    run_id = service.create_research(intake)
    worker = Thread(target=service.start_research, args=(run_id, intake), daemon=True)
    worker.start()
    assert entered.wait(5)
    service.cancel_research(run_id)
    release.set()
    worker.join(10)
    state = service.get_run(run_id)
    assert state["lifecycle_status"] == "CANCELLED" and state["admission_action"] is None
    assert state["results"]["R1"]["status"] == "SKIPPED"
    types = [event.event_type for event in service.events.replay(run_id=run_id)]
    assert "CANCELLED" in types and "RUN_COMPLETED" not in types


def test_critical_substage_failure_blocks_parent_and_persists_evidence() -> None:
    registry = SubstageHandlerRegistry()
    registry.register("R13.S2", lambda _context: StageResult("R13.S2", LifecycleStatus.FAILED,
                                                             evidence_refs=("EVD-runtime-failure",),
                                                             reasons=("runtime sentinel failed",)))
    context = ResearchLifecycle(substage_registry=registry).run("RUN-subdag", {"question": "subdag"})
    assert context.subresults["R13"]["R13.S2"].status == LifecycleStatus.FAILED
    assert context.results["R13"].status == LifecycleStatus.FAILED
    assert context.results["R14"].status == LifecycleStatus.SKIPPED


def test_subscriber_failure_does_not_rollback_persisted_event(tmp_path: Path) -> None:
    bus = PersistentEventBus(tmp_path / "events.jsonl")
    bus.subscribe(lambda _event: (_ for _ in ()).throw(RuntimeError("subscriber crashed")))
    event = bus.publish("RUN-subscriber", "TEST_EVENT", {"value": 1})
    assert bus.subscriber_error_count == 1
    assert PersistentEventBus(tmp_path / "events.jsonl").replay()[0].event_id == event.event_id


def test_process_manager_timeout_and_cancel_leave_no_running_child(tmp_path: Path) -> None:
    manager = ResearchProcessManager(terminate_grace_seconds=0.1)
    sleeping = manager.start([sys.executable, "-c", "import time; time.sleep(10)"], cwd=tmp_path)
    timed_out = manager.wait(sleeping.process_id, timeout=0.05)
    assert timed_out.status == "TIMED_OUT" and timed_out.exit_code is not None
    running = manager.start([sys.executable, "-c", "import time; time.sleep(10)"], cwd=tmp_path)
    cancelled = manager.cancel(running.process_id)
    assert cancelled.status in {"CANCELLED", "KILLED"} and cancelled.exit_code is not None
    manager.shutdown()


def test_state_corruption_is_surfaced_not_silently_omitted(tmp_path: Path) -> None:
    service = ResearchApplicationService(tmp_path)
    run_id = service.create_research({"question": "corruption?"})
    service.run_store.path_for(run_id).write_text("{broken", encoding="utf-8")
    with pytest.raises(RunStateCorruptionError):
        ResearchApplicationService(tmp_path)


def test_policy_change_appends_decision_without_rewriting_history(tmp_path: Path) -> None:
    evidence = VerificationEvidenceStore(tmp_path / "evidence.jsonl")
    for level in VERIFICATION_LEVELS:
        evidence.append(level=level, run_id="RUN-policy", experiment_id="EXP-policy", record_type="test/v1",
                        record={"level": level}, status=LifecycleStatus.PASS, producer="test")
    matrix = VerificationMatrixBuilder(evidence).build("RUN-policy", experiment_id="EXP-policy")
    policy = GovernancePolicy()
    first = policy.decide_composite("EXP-policy", matrix, profile=AdmissionProfile.STANDARD_RESEARCH,
                                    execution_mode="REAL_RESEARCH")
    second = policy.decide_composite("EXP-policy", matrix, profile=AdmissionProfile.CRITICAL_CANDIDATE,
                                     execution_mode="REAL_RESEARCH")
    store = GovernanceDecisionStore(tmp_path / "decisions.jsonl")
    old = store.append("RUN-policy", matrix, first)
    new = store.append("RUN-policy", matrix, second, supersedes_decision_id=old.decision_id)
    history = store.list_by_run("RUN-policy")
    assert [item.decision_id for item in history] == [old.decision_id, new.decision_id]
    assert history[0].policy_hash != history[1].policy_hash
