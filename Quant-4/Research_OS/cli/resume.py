from __future__ import annotations

import argparse
from pathlib import Path

from Research_OS.orchestration import ResearchLifecycle
from Research_OS.reporting import ReportBundleWriter


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resume a persisted Research OS lifecycle")
    parser.add_argument("state", type=Path)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/research_os"))
    args = parser.parse_args(argv)
    context = ResearchLifecycle().resume(args.state)
    print(ReportBundleWriter(args.reports_dir).write(context))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
