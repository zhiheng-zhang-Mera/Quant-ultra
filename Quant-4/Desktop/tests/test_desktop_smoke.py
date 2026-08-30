from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication

from Desktop.research_workbench.models import EventStreamModel, ResearchGraphModel
from Desktop.research_workbench.workbench import WorkbenchViewModel
from Research_OS.application import ResearchApplicationService


def test_models_expose_stable_roles() -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    model = ResearchGraphModel()
    model.replace([{"stageId": "R0", "name": "Intake", "status": "PASS", "x": 0, "y": 0, "critical": True}])
    assert app is not None and model.rowCount() == 1
    assert set(model.roleNames().values()) == {b"stageId", b"name", b"status", b"x", b"y", b"critical"}
    assert EventStreamModel().rowCount() == 0


def test_qml_engine_loads_offscreen(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PySide6.QtGui", exc_type=ImportError)
    from Desktop.research_workbench.app import create_engine

    _app, engine = create_engine(tmp_path)
    assert engine.rootObjects()


def test_desktop_hydrates_backend_mode_matrix_and_graph(tmp_path: Path) -> None:
    app = QCoreApplication.instance() or QCoreApplication([])
    service = ResearchApplicationService(tmp_path)
    run_id = service.create_research({"question": "hydrate truth"})
    view_model = WorkbenchViewModel(tmp_path)
    assert app is not None and view_model.activeRun == run_id
    assert view_model.executionMode == "DEMO_OFFLINE" and view_model.demoMode
    assert len(view_model.runs.rows()) == 1
    assert len(view_model.verification.rows()) == 12
    assert {row["status"] for row in view_model.verification.rows()} == {"HOLD"}
    view_model.service.events.publish(run_id, "STAGE_STARTED", {"stage_id": "R0"}, stage_id="R0")
    assert view_model.graph.rows()[0]["status"] == "RUNNING"
    view_model.shutdown()


def test_desktop_has_no_direct_orchestration_truth_import() -> None:
    source = Path(__file__).parents[1] / "research_workbench" / "workbench.py"
    assert "Research_OS.orchestration" not in source.read_text(encoding="utf-8")
