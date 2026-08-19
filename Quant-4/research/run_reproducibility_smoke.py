"""One-command reproducibility proof for the frozen US ETF experiment."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

QUANT4_ROOT = Path(__file__).resolve().parents[1]
if str(QUANT4_ROOT) not in sys.path:
    sys.path.insert(0, str(QUANT4_ROOT))

from Main.data_provenance import DatasetLineage, build_dataset_manifest, verify_dataset_manifest
from Main.reproducibility import (
    build_artifact_manifest,
    build_environment_manifest,
    compare_numeric_results,
    verify_artifact_manifest,
    write_environment_manifest,
)
from research.run_us_etf_generalization import (
    FROZEN_PARAMETERS,
    PARAMETER_HASH,
    US_ETF_UNIVERSE,
    _git_commit,
    build_validation,
    fetch_or_load,
)


def _metrics(validation: dict) -> dict[str, float]:
    return {name: float(validation["strategy"][name]) for name in ("annual_return", "sharpe", "max_drawdown")}


def run_smoke(cache_dir: Path, output_dir: Path, *, start: str, end: str, download: bool) -> dict:
    project_root = QUANT4_ROOT.parent
    frames, data_version = fetch_or_load(cache_dir, start=start, end=end, download=download)
    first = build_validation(frames, data_version, project_root)
    second = build_validation(frames, data_version, project_root)
    tolerances = {name: 1e-12 for name in _metrics(first)}
    repeatability = compare_numeric_results(_metrics(first), _metrics(second), tolerances)

    lineages = []
    for ticker in US_ETF_UNIVERSE:
        raw_path = cache_dir / f"us_etf_{ticker}_raw.csv"
        cleaned_path = cache_dir / f"us_etf_{ticker}_history.parquet"
        frame = frames[ticker]
        lineages.append(DatasetLineage(
            dataset_id=f"US_ETF:{ticker}", source_name="Yahoo Finance", source_uri=f"https://finance.yahoo.com/quote/{ticker}/history",
            license="Yahoo Finance terms; research access", downloaded_at=datetime.fromtimestamp(raw_path.stat().st_mtime, timezone.utc).isoformat(),
            coverage_start=str(frame.index.min().date()), coverage_end=str(frame.index.max().date()),
            raw_path=str(raw_path.resolve()), cleaned_path=str(cleaned_path.resolve()),
            transformations=("auto_adjust=true", "normalize OHLCV columns", "drop invalid date/open/close", "deduplicate by date", "derive amount=close*volume"),
        ))
    data_manifest = build_dataset_manifest(lineages)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "data_manifest.json"
    data_path.write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    environment_path = write_environment_manifest(
        output_dir / "environment_manifest.json",
        build_environment_manifest(project_root, random_seeds={"numpy": 42, "lightgbm": 42}),
    )
    result_path = output_dir / "us_etf_generalization.json"
    result_payload = {"validation": first, "parameters": FROZEN_PARAMETERS, "universe": list(US_ETF_UNIVERSE)}
    result_path.write_text(json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    artifact_manifest = build_artifact_manifest(
        experiment_id="us-etf-generalization/v1", code_version=_git_commit(project_root), data_version=data_version,
        parameter_version=PARAMETER_HASH,
        inputs={f"{ticker}:raw": cache_dir / f"us_etf_{ticker}_raw.csv" for ticker in US_ETF_UNIVERSE},
        outputs={"result": result_path, "data_manifest": data_path, "environment_manifest": environment_path},
        numeric_tolerances=tolerances,
    )
    artifact_path = output_dir / "artifact_manifest.json"
    artifact_path.write_text(json.dumps(artifact_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    checks = {"repeatability": repeatability, "data": verify_dataset_manifest(data_manifest),
              "artifact": verify_artifact_manifest(artifact_manifest)}
    passed = repeatability["passed"] and checks["data"]["valid"] and checks["artifact"]["valid"]
    return {"schema_version": "reproducibility-smoke/v1", "status": "PASS" if passed else "HOLD",
            "data_version": data_version, "parameter_version": PARAMETER_HASH,
            "checks": checks, "artifact_manifest_sha256": artifact_manifest["manifest_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce and verify the frozen US ETF experiment")
    parser.add_argument("--cache-dir", type=Path, default=QUANT4_ROOT / "Data_Cache" / "external_validation")
    parser.add_argument("--output-dir", type=Path, default=QUANT4_ROOT / "reports" / "reproducibility_smoke")
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=datetime.now(timezone.utc).date().isoformat())
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    result = run_smoke(args.cache_dir, args.output_dir, start=args.start, end=args.end, download=args.download)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
