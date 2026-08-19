import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Main.data_provenance import (
    DatasetLineage,
    append_provenance_event,
    build_dataset_manifest,
    verify_dataset_manifest,
    verify_provenance_history,
)
from Main.reproducibility import build_artifact_manifest, build_environment_manifest, compare_numeric_results, verify_artifact_manifest


def _lineage(tmp_path: Path) -> DatasetLineage:
    raw, cleaned = tmp_path / "raw.csv", tmp_path / "clean.parquet"
    raw.write_text("date,close\n2024-01-02,10\n", encoding="utf-8")
    cleaned.write_bytes(b"clean-v1")
    return DatasetLineage("prices:X", "vendor", "https://vendor.test/X", "research license", "2024-01-03T00:00:00Z",
                          "2024-01-02", "2024-01-02", str(raw), str(cleaned), ("normalize",))


def test_dataset_manifest_is_content_addressed_and_detects_mutation(tmp_path):
    manifest = build_dataset_manifest([_lineage(tmp_path)])
    assert verify_dataset_manifest(manifest)["valid"]
    Path(manifest["datasets"][0]["raw_path"]).write_text("mutated", encoding="utf-8")
    assert verify_dataset_manifest(manifest)["failures"] == ["hash_mismatch:prices:X:raw_path"]


def test_provenance_history_is_hash_chained(tmp_path):
    manifest = build_dataset_manifest([_lineage(tmp_path)])
    history = tmp_path / "history.jsonl"
    first = append_provenance_event(history, manifest)
    second = append_provenance_event(history, manifest, "REUSED")
    assert second["previous_hash"] == first["event_hash"]
    assert verify_provenance_history(history)["valid"]
    rows = history.read_text(encoding="utf-8").splitlines()
    event = json.loads(rows[0])
    event["event_type"] = "TAMPERED"
    rows[0] = json.dumps(event)
    history.write_text("\n".join(rows), encoding="utf-8")
    assert not verify_provenance_history(history)["valid"]


def test_artifact_manifest_and_numeric_tolerance_fail_closed(tmp_path):
    source, output = tmp_path / "input", tmp_path / "output"
    source.write_text("a", encoding="utf-8")
    output.write_text("b", encoding="utf-8")
    manifest = build_artifact_manifest(experiment_id="exp", code_version="git", data_version="data", parameter_version="params",
                                       inputs={"source": source}, outputs={"result": output}, numeric_tolerances={"sharpe": 1e-9})
    assert verify_artifact_manifest(manifest)["valid"]
    assert compare_numeric_results({"sharpe": 1.0}, {"sharpe": 1.0 + 1e-10}, {"sharpe": 1e-9})["passed"]
    assert not compare_numeric_results({"sharpe": 1.0}, {"sharpe": 1.1}, {"sharpe": 1e-9})["passed"]
    output.write_text("changed", encoding="utf-8")
    assert not verify_artifact_manifest(manifest)["valid"]


def test_environment_manifest_records_lock_seeds_and_git():
    root = Path(__file__).resolve().parents[2]
    manifest = build_environment_manifest(root, random_seeds={"numpy": 42})
    assert manifest["git_commit"]
    assert manifest["random_seeds"] == {"numpy": 42}
    assert "Quant-4/requirements-lock-py312.txt" in manifest["lock_files"]
    assert len(manifest["manifest_sha256"]) == 64
    with pytest.raises(ValueError, match="random seeds"):
        build_environment_manifest(root, random_seeds={})
