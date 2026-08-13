"""Process-level watchdog for resumable baostock downloads.

baostock can hang in a C-level socket call that holds the GIL, which starves
thread-based timeouts. This wrapper runs the fetcher as a child process,
monitors the cache file's symbol count, and kills + restarts the child
(resuming from the cache) whenever no progress is made for ``--stall``
minutes. It exits when the child finishes cleanly.

Usage:
    python tools/run_resumable_download.py --out OUT.json --target 600 --stall 5 -- \\
        python tools/fetch_fundamentals.py --top-n 600 --quarterly --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def count_symbols(out: Path) -> int:
    try:
        return len(json.loads(out.read_text(encoding="utf-8")))
    except Exception:
        return -1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--stall", type=int, default=5, help="stall minutes before restart")
    parser.add_argument("cmd", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not args.cmd or args.cmd[0] != "--":
        raise SystemExit("expected '--' before the download command")
    cmd = args.cmd[1:]
    restarts = 0
    while True:
        before = max(count_symbols(args.out), 0)
        print(f"[watchdog] run start, cache={before}/{args.target} (restarts={restarts})", flush=True)
        proc = subprocess.Popen(cmd)
        last = before
        last_t = time.time()
        while proc.poll() is None:
            time.sleep(20)
            cur = count_symbols(args.out)
            if cur > last:
                last = cur
                last_t = time.time()
                print(f"[watchdog] progress: {last}/{args.target}", flush=True)
            elif time.time() - last_t > args.stall * 60:
                print(f"[watchdog] STALL at {last} for {args.stall} min, killing and restarting", flush=True)
                proc.kill()
                proc.wait()
                restarts += 1
                break
        else:
            if proc.returncode == 0:
                print(f"[watchdog] clean exit, {last}/{args.target}", flush=True)
                return 0
            print(f"[watchdog] child exited rc={proc.returncode}, restarting", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
