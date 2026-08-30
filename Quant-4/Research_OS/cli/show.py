from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show persisted Research OS run state")
    parser.add_argument("state", type=Path)
    args = parser.parse_args(argv)
    payload = json.loads(args.state.read_text(encoding="utf-8"))
    print(f"run_id={payload['run_id']}")
    for stage_id in sorted(payload["results"], key=lambda value: int(value[1:])):
        print(f"{stage_id}\t{payload['results'][stage_id]['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
