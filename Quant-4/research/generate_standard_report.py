"""Generate a governed Markdown report from a JSON specification."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Main.research_reporting import ResearchReportMetadata, build_research_report, write_research_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a standard Quant-4 research report")
    parser.add_argument("spec", type=Path, help="JSON object containing metadata, summary, metrics, evidence, limitations and decision")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = json.loads(args.spec.read_text(encoding="utf-8"))
    report = build_research_report(ResearchReportMetadata(**source["metadata"]),
                                   summary=source["summary"], metrics=source["metrics"], evidence=source["evidence"],
                                   limitations=source["limitations"], decision=source["decision"])
    write_research_report(args.output, report)
    print(json.dumps({"status": report["metadata"]["status"], "report_sha256": report["report_sha256"],
                      "output": str(args.output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
