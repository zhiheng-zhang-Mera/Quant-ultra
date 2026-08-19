from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Main.experiment_registry import ExperimentRegistry


def _register(registry, experiment_id="EXP-20260819-001", parent_id=None):
    return registry.register(
        experiment_id=experiment_id, hypothesis="orthogonal signal improves OOS evidence",
        objective="test incremental information", variables={"signal_weight": [0.0, 0.1]},
        data_version="prices/v1", code_version="git:abc", parameter_version="params/v3",
        parent_id=parent_id, report_refs=["report:pending"],
    )


def test_registry_keeps_accept_reject_and_attempt_counts(tmp_path):
    registry = ExperimentRegistry(tmp_path / "registry.jsonl")
    _register(registry)
    registry.complete(
        experiment_id="EXP-20260819-001", status="REJECT", failure_class="COST_FAILURE",
        evidence_gate={"status": "HOLD"}, result_summary={"net_alpha": -0.01}, report_refs=["report:1"],
    )
    _register(registry, "EXP-20260819-002", "EXP-20260819-001")
    registry.complete(
        experiment_id="EXP-20260819-002", status="ACCEPT",
        evidence_gate={"status": "PASS"}, result_summary={"net_alpha": 0.02},
    )
    snapshot = registry.snapshot()
    assert snapshot["attempts"] == snapshot["completed"] == 2
    assert snapshot["status_counts"]["ACCEPT"] == snapshot["status_counts"]["REJECT"] == 1
    assert snapshot["failure_counts"]["COST_FAILURE"] == 1
    assert snapshot["negative_results"] == ["EXP-20260819-001"]


def test_registry_rejects_selective_or_invalid_history(tmp_path):
    path = tmp_path / "registry.jsonl"
    registry = ExperimentRegistry(path)
    _register(registry)
    with pytest.raises(ValueError, match="cannot be ACCEPTed"):
        registry.complete(experiment_id="EXP-20260819-001", status="ACCEPT", evidence_gate={"status": "HOLD"}, result_summary={})
    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    payload["hypothesis"] = "tampered"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash chain"):
        registry.events()
