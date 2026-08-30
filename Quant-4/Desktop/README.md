# Quant Ultra Research Workbench

PySide6/QML desktop client for `ResearchApplicationService`. QML receives only stable Qt models and commands; it never imports the research DAG, registries, policy configuration, or provider credentials.

Run from `Quant-4`:

```bash
PYTHONPATH=. python -m Desktop.research_workbench.app
```

For headless smoke testing set `QT_QPA_PLATFORM=offscreen`. The desktop dependency is intentionally isolated from the headless Research OS environment. Windows packaging uses `pyside6-deploy Desktop/research_workbench/app.py`; release artifacts still require a signed Windows runner and are not produced by source tests.
