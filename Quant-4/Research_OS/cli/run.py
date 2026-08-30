"""Start a deterministic Research OS lifecycle from YAML."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from Research_OS.contracts.common import stable_id
from Research_OS.orchestration import ResearchLifecycle
from Research_OS.orchestration.budget import ResearchBudget
from Research_OS.reporting import ReportBundleWriter


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run an evidence-governed quantitative research lifecycle")
    result.add_argument("intake", type=Path)
    result.add_argument("--offline-agents", action="store_true", default=False)
    result.add_argument("--provider", default="deterministic-stub")
    result.add_argument("--max-requests", type=int, default=100)
    result.add_argument("--max-tokens", type=int, default=1_000_000)
    result.add_argument("--max-compute-seconds", type=float, default=3600.0)
    result.add_argument("--run-id")
    result.add_argument("--state-dir", type=Path, default=Path("reports/research_os_state"))
    result.add_argument("--reports-dir", type=Path, default=Path("reports/research_os"))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if not args.intake.is_file():
        raise FileNotFoundError(args.intake)
    intake = yaml.safe_load(args.intake.read_text(encoding="utf-8"))
    if not isinstance(intake, dict):
        raise ValueError("intake YAML must contain a mapping")
    if args.provider != "deterministic-stub":
        raise ValueError("remote providers are not configured; use deterministic-stub or install an explicit provider adapter")
    run_id = args.run_id or stable_id("RUN", intake)
    budget = ResearchBudget(args.max_requests, args.max_tokens, args.max_compute_seconds)
    state_path = args.state_dir / f"{run_id}.json"
    context = ResearchLifecycle().run(run_id, intake, state_path=state_path, budget=budget)
    report_path = ReportBundleWriter(args.reports_dir).write(context)
    print(f"run_id={run_id}")
    print(f"status={context.results['R19'].output.get('action', context.results['R19'].status.value)}")
    print(f"workspace={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
