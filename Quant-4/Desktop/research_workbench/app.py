from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine

from .workbench import WorkbenchViewModel


def create_engine(state_dir: str | Path | None = None) -> tuple[QGuiApplication, QQmlApplicationEngine]:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()
    view_model = WorkbenchViewModel(state_dir or Path.cwd() / "reports" / "workbench_state")
    engine.rootContext().setContextProperty("workbench", view_model)
    engine.setProperty("workbenchViewModel", view_model)
    app.aboutToQuit.connect(view_model.shutdown)
    qml = Path(__file__).resolve().parent / "qml" / "Main.qml"
    engine.load(QUrl.fromLocalFile(str(qml)))
    return app, engine


def main() -> int:
    os.environ.setdefault("QT_QUICK_CONTROLS_STYLE", "Basic")
    app, engine = create_engine()
    if not engine.rootObjects():
        return 2
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
