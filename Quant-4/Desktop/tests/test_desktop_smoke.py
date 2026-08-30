from pathlib import Path

import pytest
from PySide6.QtCore import QCoreApplication

from Desktop.research_workbench.models import EventStreamModel, ResearchGraphModel


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
