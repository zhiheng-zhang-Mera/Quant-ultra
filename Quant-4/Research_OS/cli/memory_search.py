from __future__ import annotations

import argparse
import json
from pathlib import Path

from Research_OS.memory import ResearchMemory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search offline Research OS memory")
    parser.add_argument("query")
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args(argv)
    print(json.dumps(ResearchMemory(args.registry).search(args.query, limit=args.limit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
