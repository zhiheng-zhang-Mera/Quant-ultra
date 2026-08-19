import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Main.research_reporting import ResearchReportMetadata, build_research_report, render_markdown, write_research_report


def _metadata(**changes):
    values = {"report_type": "BENCHMARK", "experiment_id": "EXP-20260819-001", "status": "HOLD",
              "code_version": "git:abc", "data_version": "sha256:data", "parameter_version": "sha256:params",
              "generated_at": "2026-08-19T00:00:00+00:00", "title": "Comparable Benchmark Report"}
    values.update(changes)
    return ResearchReportMetadata(**values)


def test_standard_report_requires_versions_evidence_limits_and_decision(tmp_path):
    report = build_research_report(_metadata(), summary="Same-contract benchmark capability is available.",
                                   metrics={"candidates": 6}, evidence=["manifest hash verified"],
                                   limitations=["No investment-validity claim."], decision="HOLD until real data run.")
    assert len(report["report_sha256"]) == 64
    rendered = render_markdown(report)
    for field in ("Experiment ID", "Code version", "Data version", "Limitations", "Decision"):
        assert field in rendered
    path = write_research_report(tmp_path / "report.md", report)
    assert path.read_text(encoding="utf-8") == rendered


def test_failed_experiment_is_a_first_class_report():
    report = build_research_report(_metadata(report_type="NEGATIVE_RESULT", status="REJECT"), summary="Hypothesis failed.",
                                   metrics={"oos_delta": -0.01}, evidence=["registered experiment"],
                                   limitations=["single market"], decision="REJECT and retain history.")
    assert report["metadata"]["status"] == "REJECT"
    with pytest.raises(ValueError):
        build_research_report(_metadata(status="SUCCESS"), summary="x", metrics={"x": 1}, evidence=["x"], limitations=["x"], decision="x")


def test_report_rejects_missing_traceability():
    with pytest.raises(ValueError, match="requires code"):
        _metadata(data_version="").validate()
    with pytest.raises(ValueError, match="require summary"):
        build_research_report(_metadata(), summary="", metrics={}, evidence=[], limitations=[], decision="")


def test_research_documentation_index_and_governed_reports_are_complete():
    repository = Path(__file__).resolve().parents[2]
    docs = repository / "docs" / "research"
    required_methods = {
        "METHODOLOGY.md", "STATISTICAL_VALIDATION.md", "DATA_AND_PIT.md",
        "EXPERIMENT_AND_BENCHMARK_GOVERNANCE.md", "FACTOR_AND_ALTERNATIVE_DATA.md",
        "GENERALIZATION.md", "REPRODUCIBILITY.md", "LIMITATIONS.md",
    }
    index = (docs / "README.md").read_text(encoding="utf-8")
    assert all((docs / name).is_file() and name in index for name in required_methods)
    for name in ("BENCHMARK_REPORT.md", "NEGATIVE_RESULTS_REPORT.md", "REPRODUCIBILITY_REPORT.md"):
        report = (docs / "reports" / name).read_text(encoding="utf-8")
        for field in ("Experiment ID", "Code version", "Data version", "Parameter version", "Report SHA-256", "## Limitations", "## Decision"):
            assert field in report
