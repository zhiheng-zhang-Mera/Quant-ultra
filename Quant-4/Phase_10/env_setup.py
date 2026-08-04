"""Read-only Ollama environment inspection for the optional Phase 10 helper."""
from __future__ import annotations

import json
import shutil
import subprocess
from urllib.request import urlopen

from .config import TARGET_MODEL


def ensure_python_package() -> dict:
    """Report the optional Python client; never install it."""
    try:
        import ollama  # noqa: F401
        return {"python_client_present": True}
    except ImportError:
        return {"python_client_present": False, "note": "optional; setup does not install it"}


def ensure_ollama_framework() -> dict:
    """Report the executable and version; never install or start Ollama."""
    executable = shutil.which("ollama")
    result = {"executable": executable, "installed": bool(executable), "check_only": True}
    if executable:
        try:
            proc = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=5, check=False)
            result["version"] = (proc.stdout or proc.stderr).strip()
        except (OSError, subprocess.TimeoutExpired) as exc:
            result["error"] = str(exc)
    return result


def ensure_model_pulled() -> dict:
    """Report service/model availability; never pull, remove, or modify models."""
    result = {"target_model": TARGET_MODEL, "service_reachable": False, "model_present": False, "check_only": True}
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        names = sorted({str(item.get("name") or item.get("model")) for item in payload.get("models", []) if item.get("name") or item.get("model")})
        result.update({"service_reachable": True, "models": names, "model_present": any(name == TARGET_MODEL or name.startswith(TARGET_MODEL + ":") for name in names)})
    except Exception as exc:
        result["error"] = str(exc)
    return result


def setup_all() -> dict:
    result = {"python_client": ensure_python_package(), "framework": ensure_ollama_framework(), "model": ensure_model_pulled()}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
