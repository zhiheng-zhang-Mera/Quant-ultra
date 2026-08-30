"""Clean-install smoke entry used by the desktop CI job."""
from __future__ import annotations

import tempfile
from pathlib import Path

import research_workbench
from research_workbench.app import create_engine


def main() -> int:
    qml = Path(research_workbench.__file__).resolve().parent / "qml" / "Main.qml"
    if not qml.is_file():
        raise FileNotFoundError("packaged Main.qml is missing")
    with tempfile.TemporaryDirectory(prefix="quant-workbench-smoke-") as directory:
        _app, engine = create_engine(directory)
        if not engine.rootObjects():
            raise RuntimeError("QML engine created no root object")
        view_model = engine.property("workbenchViewModel")
        view_model.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
