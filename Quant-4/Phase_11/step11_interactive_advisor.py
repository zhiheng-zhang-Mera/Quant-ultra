"""Phase 11 entrypoint: candidate generation followed by optional portfolio query."""
from __future__ import annotations
import sys
from pathlib import Path
from Main.investment_advisor import build_pipeline_recommendations, write_candidate_report


def execute(context: dict) -> dict:
    if context.get("phase10_ready") is not True:
        raise RuntimeError("Phase 11 requires a completed Phase 10 evidence report")
    run_id = context.get("run_metadata", {}).get("timestamp", "unknown")
    report_dir = Path(__file__).parents[1] / "reports" / "runs" / run_id / "phase11"
    candidates = build_pipeline_recommendations(context)
    md_path, csv_path = write_candidate_report(candidates, report_dir)
    interactive = bool(context.get("config", {}).get("phase11_interactive", False)) and sys.stdin.isatty()
    if interactive:
        from analyze_cn_asset import run_portfolio_query
        run_portfolio_query(context["data_manager"], report_dir=report_dir / "portfolio_queries")
    return {
        "investment_candidates": candidates,
        "phase11_report_path": str(md_path),
        "phase11_csv_path": str(csv_path),
        "phase11_interactive_started": interactive,
        "phase11_ready": True,
    }
