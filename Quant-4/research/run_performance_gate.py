"""Run deterministic full-pool and alternative-data performance budgets."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Main.performance_engineering import run_performance_gate


def main() -> int:
    parser = argparse.ArgumentParser(description="Quant-4 performance regression gate")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "performance_gate.json")
    args = parser.parse_args()
    result = run_performance_gate(args.output)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
