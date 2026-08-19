#!/usr/bin/env python
"""One-command, isolated dependency installer for Quant-Ultra.

Ollama and the configured local model are inspected only. This script never
installs Ollama, starts its service, or pulls/removes a model.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from urllib.request import urlopen

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
LOCK_REQUIREMENTS = PROJECT_ROOT / f"requirements-lock-py{sys.version_info.major}{sys.version_info.minor}.txt"
REPORT_PATH = PROJECT_ROOT / "reports" / "setup" / "setup_report.json"

IMPORT_CHECKS = {
    "numpy": "numpy", "pandas": "pandas", "scipy": "scipy",
    "scikit-learn": "sklearn", "lightgbm": "lightgbm",
    "statsmodels": "statsmodels", "cvxpy": "cvxpy", "pyarrow": "pyarrow",
    "PyYAML": "yaml", "pytz": "pytz", "requests": "requests",
    "tenacity": "tenacity", "akshare": "akshare", "baostock": "baostock",
    "efinance": "efinance", "yfinance": "yfinance", "tushare": "tushare",
    "Cython": "Cython", "pytest": "pytest", "tqdm": "tqdm",
    "matplotlib": "matplotlib", "tabulate": "tabulate",
    "pandas-market-calendars": "pandas_market_calendars", "psutil": "psutil",
    "optuna": "optuna", "hmmlearn": "hmmlearn", "transformers": "transformers",
    "torch": "torch", "plotly": "plotly",
    "fancyimpute": "fancyimpute", "chinese-calendar": "chinese_calendar",
}


def run(command: list[str], cwd: Path = PROJECT_ROOT, check: bool = True) -> subprocess.CompletedProcess:
    print("[setup]", subprocess.list2cmdline(command), flush=True)
    return subprocess.run(command, cwd=cwd, check=check, text=True)


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def configured_model() -> str:
    path = PROJECT_ROOT / "Main" / "default_param.yaml"
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("local_llm_model:"):
            return line.split(":", 1)[1].strip().strip("\"'")
    return "qwen3-coder:30b"


def inspect_ollama(model: str) -> dict:
    result = {"check_only": True, "executable": None, "version": None, "service_reachable": False, "configured_model": model, "model_present": False, "models": []}
    executable = shutil.which("ollama")
    result["executable"] = executable
    if executable:
        try:
            version = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=5, check=False)
            result["version"] = (version.stdout or version.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["version_error"] = str(exc)
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = sorted({str(item.get("name") or item.get("model")) for item in payload.get("models", []) if item.get("name") or item.get("model")})
        result.update({"service_reachable": True, "models": names, "model_present": any(name == model or name.startswith(model + ":") for name in names)})
    except Exception as exc:
        result["service_error"] = str(exc)
    return result


def verify_imports() -> tuple[list[str], dict[str, str]]:
    passed, failed = [], {}
    for distribution, module in IMPORT_CHECKS.items():
        try:
            importlib.import_module(module)
            passed.append(distribution)
        except Exception as exc:
            failed[distribution] = f"{type(exc).__name__}: {exc}"
    return passed, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install and verify all Quant-Ultra Python dependencies")
    parser.add_argument("--venv", type=Path, default=REPOSITORY_ROOT / ".venv-full")
    parser.add_argument("--mirror")
    parser.add_argument("--inside-venv", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-pip-upgrade", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = args.venv.resolve()
    python = venv_python(target)
    if not args.inside_venv:
        if not python.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            run([sys.executable, "-m", "venv", str(target)])
        command = [str(python), str(Path(__file__).resolve()), "--venv", str(target), "--inside-venv"]
        if args.mirror: command += ["--mirror", args.mirror]
        if args.skip_tests: command.append("--skip-tests")
        if args.no_pip_upgrade: command.append("--no-pip-upgrade")
        return run(command).returncode

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pip_base = [sys.executable, "-m", "pip", "install"]
    if args.mirror: pip_base += ["--index-url", args.mirror]
    if not args.no_pip_upgrade:
        run(pip_base + ["--upgrade", "pip", "setuptools", "wheel"])
    install_requirements = LOCK_REQUIREMENTS if LOCK_REQUIREMENTS.is_file() else REQUIREMENTS
    run(pip_base + ["--requirement", str(install_requirements)])
    pip_check = run([sys.executable, "-m", "pip", "check"], check=False)
    passed, failed = verify_imports()
    compile_check = run([sys.executable, "-m", "compileall", "-q", "Main", "Phase_1", "Phase_2", "Phase_3", "Phase_4", "Phase_5", "Phase_6", "Phase_7", "Phase_8", "Phase_9", "Phase_10", "Phase_11"], check=False)
    test_code = None
    if not args.skip_tests:
        test_code = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_improvements.py"], check=False).returncode
    ollama = inspect_ollama(configured_model())
    success = pip_check.returncode == 0 and not failed and compile_check.returncode == 0 and test_code in (None, 0)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(), "success": success,
        "project_root": str(PROJECT_ROOT), "venv": str(target), "python": sys.executable,
        "requirements": str(install_requirements), "lock_used": install_requirements == LOCK_REQUIREMENTS,
        "imports_passed": passed, "imports_failed": failed,
        "pip_check_exit_code": pip_check.returncode, "compile_exit_code": compile_check.returncode,
        "test_exit_code": test_code, "ollama": ollama,
        "notes": ["Ollama/model checks are read-only", "Native Cython build is optional; verified NumPy fallback remains available"],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
