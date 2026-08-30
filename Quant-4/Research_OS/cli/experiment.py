from __future__ import annotations

import argparse
import json
from pathlib import Path

from Research_OS.registry import ExperimentRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect a preregistered experiment")
    parser.add_argument("experiment_id")
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(ExperimentRegistry(args.registry).get(args.experiment_id), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
