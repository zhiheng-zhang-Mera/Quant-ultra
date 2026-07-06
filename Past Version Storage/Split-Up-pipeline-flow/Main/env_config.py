"""
Quant-Ultra Flow - Environment & Path Configuration
"""
import subprocess
from pathlib import Path

# 路径向上推一级，直达项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def get_git_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "NO_GIT"

def get_git_status() -> str:
    try:
        status = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        return "CLEAN" if not status else "DIRTY"
    except Exception:
        return "UNKNOWN"