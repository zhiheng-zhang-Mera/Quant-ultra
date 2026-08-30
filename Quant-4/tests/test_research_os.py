from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from Research_OS.adapters.quant_kernel import QuantKernelAdapter, QuantKernelRequest
from Research_OS.agents import Agent, AgentIdentity, ModelDiversityPolicy, Role, validate_separation
from Research_OS.agents.base import blinded_context
from Research_OS.contracts.common import (
    AdmissionAction,
    LifecycleStatus,
    LockState,
    Permission,
    ValidationError,
    canonical_json,
    sha256,
    stable_id,
)
from Research_OS.contracts.evidence import EvidenceRecord, SourceRecord
from Research_OS.contracts.experiment import ExperimentDesign, ExperimentSpec, KernelRunManifest
from Research_OS.contracts.governance import AgentAssessment, EvidenceMatrix, MemoryRecord
from Research_OS.contracts.research import Hypothesis, ResearchRequest
from Research_OS.evidence.data_gate import CRITICAL_CHECKS, evaluate_data_gate
from Research_OS.evidence.graph import EvidenceGraph, validate_pit
from Research_OS.governance import GovernancePolicy, ProductionInvariantError
from Research_OS.memory import ResearchMemory
from Research_OS.orchestration import ResearchLifecycle
from Research_OS.orchestration.budget import ResearchBudget
from Research_OS.providers import AgentRequest, DeterministicStubProvider
from Research_OS.registry import ExperimentRegistry
from Research_OS.registry.storage import HashChainStore, RegistryCorruptionError
from Research_OS.reporting import ReportBundleWriter
from Research_OS.security import ExecutionPolicy, redact_secrets, safe_workspace_path
from Research_OS.verification import AttackRegistry, StaticVerificationCoordinator, compare_reproduction, placebo_attack


def _design() -> ExperimentDesign:
    return ExperimentDesign(
        schema_version="experiment-design/v1", universe=("AAA", "BBB"), market="SYNTHETIC", data_cutoff="2024-12-31",
        train_window=("2020-01-01", "2021-12-31"), validation_window=("2022-01-10", "2022-12-31"),
        oos_window=("2023-01-10", "2024-12-31"), walk_forward=True, embargo_days=5, benchmark="EQUAL_WEIGHT",
        transaction_cost_model="explicit-v1", primary_metrics=("sharpe",), acceptance_rules={"sharpe": 0.5},
        rejection_rules={"max_drawdown": -0.3}, robustness_tests=("signal_randomization_placebo",),
    )


def _spec(number: int = 1, *, max_trials: int = 2) -> ExperimentSpec:
    return ExperimentSpec(
        schema_version="experiment-spec/v1", experiment_id=f"EXP-test-{number:03d}", hypothesis_id="HYP-test-001",
        experiment_family_id="FAMILY-test", max_trials=max_trials, design=_design(),
        frozen_code_scope=("Main/factor_research.py",), frozen_data_scope=("synthetic",),
        frozen_metric_definitions={"sharpe": "annualized excess-return Sharpe"},
        parent_experiment_id="EXP-test-001" if number > 1 else None,
    )


def _run(run_id: str, metrics: dict[str, float], *, data_hash: str = "data") -> KernelRunManifest:
    return KernelRunManifest(schema_version="kernel-run-manifest/v1", run_id=run_id, experiment_id="EXP-test-001",
                             spec_sha256="spec", code_sha256="code", data_sha256=data_hash, git_sha="git",
                             environment_sha256="env", metrics=metrics)


def test_canonical_serialization_and_hash_are_deterministic():
    first = {"b": {3, 1, 2}, "a": datetime(2026, 8, 30, tzinfo=timezone.utc)}
    second = {"a": datetime(2026, 8, 30, tzinfo=timezone.utc), "b": {2, 3, 1}}
    assert canonical_json(first) == canonical_json(second)
    assert sha256(first) == sha256(second)
    with pytest.raises(ValidationError):
        canonical_json(float("nan"))


def test_contracts_reject_wrong_versions_and_bad_ids():
    with pytest.raises(ValidationError):
        ResearchRequest(schema_version="wrong", request_id="REQ-test-001", title="x", question="q")
    with pytest.raises(ValidationError):
        stable_id("BAD", "x")


def test_hypothesis_requires_counter_hypothesis_and_falsification():
    with pytest.raises(ValidationError):
        Hypothesis(schema_version="hypothesis/v1", hypothesis_id="HYP-test-001", request_id="REQ-test-001",
                   statement="x", economic_rationale="mechanism", expected_effect="positive",
                   counter_hypothesis="", proposer_agent_id="AGT-test-001")


def test_experiment_spec_is_frozen_and_hash_changes_with_field():
    spec = _spec()
    with pytest.raises(FrozenInstanceError):
        spec.max_trials = 4
    changed = replace(spec, max_trials=3)
    assert changed.spec_sha256 != spec.spec_sha256


def test_overlapping_experiment_windows_are_rejected():
    with pytest.raises(ValidationError):
        replace(_design(), validation_window=("2021-12-01", "2022-12-31"))


def test_registry_is_append_only_and_enforces_budget(tmp_path):
    registry = ExperimentRegistry(tmp_path / "experiments.jsonl")
    registry.register(_spec(), actor="human:test")
    registry.transition("EXP-test-001", LockState.PREREGISTERED, actor="human:test", reason="design frozen")
    registry.transition("EXP-test-001", LockState.LOCKED, actor="human:test", reason="hash verified")
    registry.register(_spec(2), actor="human:test")
    assert registry.family_budget("FAMILY-test")["remaining_trials"] == 0
    with pytest.raises(ValueError, match="exhausted"):
        registry.register(_spec(3), actor="human:test")
    with pytest.raises(ValueError):
        registry.transition("EXP-test-001", LockState.DRAFT, actor="human:test", reason="erase history")


def test_hash_chain_detects_registry_tampering(tmp_path):
    path = tmp_path / "events.jsonl"
    store = HashChainStore(path, schema_version="test/v1")
    store.append("CREATED", {"value": 1}, actor="test")
    payload = json.loads(path.read_text())
    payload["payload"]["value"] = 2
    path.write_text(json.dumps(payload) + "\n")
    with pytest.raises(RegistryCorruptionError):
        store.read()


def test_negative_memory_is_first_class_and_retrievable(tmp_path):
    memory = ResearchMemory(tmp_path / "memory.jsonl")
    record = MemoryRecord(schema_version="memory-record/v1", memory_id="MEM-rsi-reject", memory_type="FAILURE",
                          hypothesis_text="standalone RSI reversal factor", mechanism="short-term reversal",
                          features=("rsi14",), parameters={"window": "14"}, universe=("A-share",), market="CN",
                          result="REJECT", failure_reason="full-pool negative after small-sample positive")
    memory.remember(record, actor="migration:test")
    result = memory.search("RSI reversal", features=("rsi14",), mechanism="short term reversal", universe=("A-share",))[0]
    assert result["known_failure"] and result["score"] > 0.5


def test_evidence_graph_preserves_conflict_and_source_diversity():
    graph = EvidenceGraph()
    now = datetime.now(timezone.utc)
    for number, source_type in ((1, "news"), (2, "filing")):
        graph.add_source(SourceRecord(schema_version="source-record/v1", source_id=f"SRC-test-00{number}",
                                      name=source_type, source_type=source_type, version="v1", reliability=0.8,
                                      content_sha256=f"source-{number}", license_or_usage_note="research"))
        graph.add_evidence(EvidenceRecord(schema_version="evidence-record/v1", evidence_id=f"EVD-test-00{number}",
                                          source_id=f"SRC-test-00{number}", entity="AAA", event_type="earnings",
                                          claim="direction", direction="POSITIVE" if number == 1 else "NEGATIVE",
                                          horizon="1m", confidence=0.7, event_time=now - timedelta(days=2),
                                          publish_time=now - timedelta(days=1), availability_time=now - timedelta(hours=12),
                                          ingest_time=now, content_sha256=f"evidence-{number}"))
    assert len(graph.detect_conflicts()) == 1
    assert graph.source_diversity() == 1.0
    assert graph.agreement_metric()["truth_probability"] is False


def test_invalid_pit_chronology_cannot_pass():
    now = datetime.now(timezone.utc)
    record = EvidenceRecord(schema_version="evidence-record/v1", evidence_id="EVD-test-pit", source_id="SRC-test-pit",
                            entity="AAA", event_type="filing", claim="x", confidence=0.5,
                            event_time=now, publish_time=now - timedelta(days=1), availability_time=now,
                            ingest_time=now, content_sha256="x")
    result = validate_pit(record, use_time=now - timedelta(hours=1))
    assert not result.valid
    assert "event_after_publication" in result.reasons and "not_available_at_use_time" in result.reasons


def test_missing_data_and_no_eligible_event_remain_distinct():
    now = datetime.now(timezone.utc)
    missing = EvidenceRecord(schema_version="evidence-record/v1", evidence_id="EVD-test-missing", source_id="SRC-test-pit",
                             state="DATA_MISSING", ingest_time=now)
    none = replace(missing, evidence_id="EVD-test-none", state="NO_ELIGIBLE_EVENT")
    assert validate_pit(missing).reasons != validate_pit(none).reasons


def test_data_gate_fails_closed_on_missing_or_failed_checks():
    passed = {key: True for key in CRITICAL_CHECKS}
    assert evaluate_data_gate(passed).status == "DATA_PASS"
    missing = dict(passed)
    missing.pop("lookahead")
    assert evaluate_data_gate(missing).status == "DATA_HOLD"
    failed = dict(passed)
    failed["lookahead"] = False
    assert evaluate_data_gate(failed).status == "DATA_REJECT"


def test_agent_has_no_production_permission_and_developer_cannot_self_verify():
    identity = AgentIdentity("AGT-test-dev", Role.QUANT_DEVELOPER, "stub")
    agent = Agent(identity, DeterministicStubProvider(provider_id="stub"))
    with pytest.raises(PermissionError):
        agent.require(Permission.PRODUCTION)
    with pytest.raises(ValueError):
        validate_separation(proposer_agent_id=identity.agent_id, verifier_agent_id=identity.agent_id)


def test_blinded_context_removes_result_leakage():
    context = {"hypothesis": "x", "desired_sharpe": 2.0, "hidden_oos_metrics": {"sharpe": 3.0}}
    assert blinded_context(context, "developer") == {"hypothesis": "x"}


def test_malformed_provider_output_fails_closed():
    provider = DeterministicStubProvider(provider_id="stub", responder=lambda request: {"proposal": "missing status"})
    agent = Agent(AgentIdentity("AGT-test-critic", Role.ADVERSARIAL_CRITIC, "stub"), provider)
    with pytest.raises(ValueError, match="malformed"):
        agent.run(AgentRequest("REQ-test-001", "critic", "review"))


def test_model_diversity_is_not_validation():
    policy = ModelDiversityPolicy()
    assert policy.agreement_is_not_validation
    with pytest.raises(ValueError):
        policy.validate(["one", "one"])


def test_static_verifier_detects_future_and_label_leakage():
    result = StaticVerificationCoordinator().verify(
        {"factor.py": "feature = prices.shift(-1)\nX['label_feature'] = y_test"},
        developer_agent_id="AGT-test-dev", reviewer_agent_id="AGT-test-review",
    )
    assert result["status"] == "IMPLEMENTATION_REJECTED"
    assert {item.code for item in result["findings"]} >= {"FUTURE_SHIFT", "LABEL_LEAK"}


def test_reproduction_requires_clean_independent_matching_run():
    primary = _run("RUN-primary", {"sharpe": 1.0})
    reproduced = compare_reproduction(primary, _run("RUN-shadow", {"sharpe": 1.0}),
                                      reproduction_agent_id="AGT-test-repro", implementation_agent_id="AGT-test-dev")
    assert reproduced.status == "REPRODUCED"
    mismatch = compare_reproduction(primary, _run("RUN-dirty", {"sharpe": 1.0}, data_hash="contaminated"),
                                    reproduction_agent_id="AGT-test-repro", implementation_agent_id="AGT-test-dev")
    assert mismatch.status == "NON_REPRODUCIBLE"


def test_randomized_placebo_cannot_inherit_alpha():
    attacks = AttackRegistry()
    attacks.register(placebo_attack("signal", "alpha_present"))
    evidence = attacks.run("EXP-test-001", ("signal_randomization_placebo",),
                           {"signal_randomized": True, "alpha_present": True})
    assert evidence.critical_failures == ("signal_randomization_placebo",)


def test_governance_is_deterministic_not_majority_vote():
    dimensions = {key: LifecycleStatus.PASS for key in ("mechanism", "source", "data", "implementation",
                                                        "reproducibility", "statistics", "risk", "robustness", "generalization")}
    dimensions["data"] = LifecycleStatus.REJECT
    votes = tuple(AgentAssessment(schema_version="agent-assessment/v1", agent_id=f"AGT-test-{index:03d}", role="critic",
                                  status=LifecycleStatus.PASS, confidence=1.0) for index in range(9))
    matrix = EvidenceMatrix(schema_version="evidence-matrix/v1", dimensions=dimensions)
    decision = GovernancePolicy().decide("EXP-test-001", matrix, votes)
    assert decision.action == AdmissionAction.REJECTED


def test_missing_mandatory_evidence_holds_and_all_pass_is_only_candidate():
    policy = GovernancePolicy()
    hold = policy.decide("EXP-test-001", EvidenceMatrix(schema_version="evidence-matrix/v1", dimensions={"data": LifecycleStatus.PASS}))
    assert hold.action == AdmissionAction.HOLD_FOR_REVIEW
    dimensions = {key: LifecycleStatus.PASS for key in ("mechanism", "source", "data", "implementation",
                                                        "reproducibility", "statistics", "risk", "robustness", "generalization")}
    candidate = policy.decide("EXP-test-001", EvidenceMatrix(schema_version="evidence-matrix/v1", dimensions=dimensions))
    assert candidate.action == AdmissionAction.PRODUCTION_CANDIDATE
    assert candidate.requires_human_authorization


def test_production_invariants_reject_target_tuning_and_ai_authority():
    with pytest.raises(ProductionInvariantError):
        GovernancePolicy.assert_production_invariants(
            locked_spec=True, pit_passed=True, reproducible=True, critical_findings=0, trial_count_known=True,
            target_search_trials=1, parameter_hash_match=True, production_config_mutated=False,
            actor_has_production_permission=True,
        )


def test_full_offline_r0_r20_is_deterministic_and_resumable(tmp_path):
    state = tmp_path / "state.json"
    first = ResearchLifecycle().run("RUN-test-offline", {"question": "synthetic"}, state_path=state)
    assert tuple(first.results) == tuple(f"R{number}" for number in range(21))
    assert all(result.status == LifecycleStatus.PASS for result in first.results.values())
    assert first.results["R19"].output["action"] == "RESEARCH_ONLY"
    resumed = ResearchLifecycle().resume(state)
    assert {key: value.output for key, value in resumed.results.items()} == {key: value.output for key, value in first.results.items()}


def test_critical_hold_blocks_downstream_stage(tmp_path):
    from Research_OS.orchestration.dag import DAG, Stage, StageContext, StageResult

    def hold(context):
        return StageResult("R0", LifecycleStatus.HOLD, reasons=("critical evidence missing",))

    def should_not_run(context):
        pytest.fail("blocked downstream handler ran")

    dag = DAG([
        Stage("R0", "critical gate", (), hold, critical=True),
        Stage("R1", "downstream", ("R0",), should_not_run),
    ])
    context = dag.run(StageContext("RUN-test-hold", {}), state_path=tmp_path / "state.json")
    assert context.results["R1"].status == LifecycleStatus.SKIPPED


def test_budget_exhaustion_holds_or_fails_closed(tmp_path):
    context = ResearchLifecycle().run("RUN-test-budget", {"question": "x"},
                                      budget=ResearchBudget(max_requests=1), state_path=tmp_path / "state.json")
    assert context.results["R1"].status in {LifecycleStatus.HOLD, LifecycleStatus.FAILED}
    assert all(context.results[f"R{number}"].status != LifecycleStatus.PASS for number in range(1, 21))


def test_report_bundle_contains_all_hashed_stage_files(tmp_path):
    context = ResearchLifecycle().run("RUN-test-report", {"question": "x"})
    root = ReportBundleWriter(tmp_path).write(context)
    manifest = json.loads((root / "manifest.json").read_text())
    assert len(manifest["files"]) == 21
    first_name, first_hash = next(iter(manifest["files"].items()))
    assert hashlib.sha256((root / first_name).read_bytes()).hexdigest() == first_hash
    assert (root / "summary.md").is_file()
    assert "production_activation" in (root / "19_governance.json").read_text()


def test_security_redacts_secrets_and_blocks_traversal_and_shell():
    assert "super-secret" not in redact_secrets("api_key=super-secret")
    with pytest.raises(ValueError):
        safe_workspace_path("/tmp/workspace", "../escape")
    policy = ExecutionPolicy(Path("/tmp/workspace"))
    with pytest.raises(PermissionError):
        policy.validate(("bash", "-c", "echo unsafe"))


def test_kernel_adapter_uses_main_py_isolated_overlay_and_preserves_defaults(tmp_path):
    project = tmp_path / "Quant-4"
    (project / "Main").mkdir(parents=True)
    (project / "Main" / "main.py").write_text("print('kernel')")
    (project / "Main" / "default_param.yaml").write_text("analysis_only: true\n")
    calls = []
    def runner(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "ok", "")
    adapter = QuantKernelAdapter(project, runner=runner)
    result = adapter.run(QuantKernelRequest("EXP-test-001", "spec", {"embargo_min": 5}, ("AAA",)))
    assert result.status == "PRIMARY_EVIDENCE" and result.production_config_unchanged
    assert "main_engine.py" not in " ".join(calls[0][0])
    with pytest.raises(ValueError):
        adapter.validate_overlay({"live_trading": True})
