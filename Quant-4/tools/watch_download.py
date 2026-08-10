"""Self-healing supervisor for the universe downloader.

The data sources are flaky (Eastmoney rate-limits, py_mini_racer can crash on
concurrent V8 init, Sina throttles bursts), but the downloader is resumable:
it skips already-complete parquet caches and persists failures. This watchdog
simply relaunches it whenever it exits before the coverage target is reached,
so the job converges even if individual processes crash or stall.

Usage:
    python tools/watch_download.py [coverage_target] [expected_files]
"""
from __future__ import annotations

import glob
import subprocess
import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[2]
QUANT4 = WORKSPACE / "Quant-4"
CACHE_DIR = QUANT4 / "Data_Cache"
VENV_PY = r"D:\quant-4-test-cache\.venv-quant4\Scripts\python.exe"
DOWNLOADER = str(QUANT4 / "tools" / "download_universe.py")

LOG = Path(r"D:\quant-4-test-cache\universe_watchdog.log")


def _covered() -> int:
    return len(glob.glob(str(CACHE_DIR / "*_history.parquet")))


def main() -> int:
    target_ratio = float(sys.argv[1]) if len(sys.argv) > 1 else 0.95
    expected = int(sys.argv[2]) if len(sys.argv) > 2 else 5475
    target_files = expected * target_ratio
    restart_delay = 15.0
    with LOG.open("a", encoding="utf-8") as log:
        def logm(msg: str) -> None:
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")
            log.flush()

        logm(f"watchdog start: covered={_covered()} target={target_files:.0f} "
             f"downloader={DOWNLOADER} cache={CACHE_DIR}")
        while _covered() < target_files:
            logm(f"launch downloader (covered={_covered()})")
            try:
                proc = subprocess.run(
                    [VENV_PY, DOWNLOADER, "--workers", "2", "--limit", "0"],
                    cwd=str(WORKSPACE), timeout=7200,
                )
                logm(f"downloader exited code={proc.returncode} covered={_covered()}")
            except subprocess.TimeoutExpired:
                logm("downloader run timed out (7200s); relaunching")
            except Exception as exc:  # noqa: BLE001
                logm(f"watchdog error: {type(exc).__name__}: {exc}")
            if _covered() < target_files:
                time.sleep(restart_delay)
        logm(f"watchdog done: covered={_covered()} >= target {target_files:.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
