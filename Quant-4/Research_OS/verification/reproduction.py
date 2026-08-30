from __future__ import annotations

from Research_OS.contracts.experiment import KernelRunManifest
from Research_OS.contracts.verification import ReproductionManifest


def compare_reproduction(primary: KernelRunManifest, reproduction: KernelRunManifest, *,
                         reproduction_agent_id: str, implementation_agent_id: str,
                         tolerances: dict[str, float] | None = None,
                         fresh_process: bool = True, clean_cache: bool = True) -> ReproductionManifest:
    if reproduction_agent_id == implementation_agent_id:
        raise ValueError("implementation agent cannot reproduce its own result")
    limits = tolerances or {key: 1e-12 for key in primary.metrics}
    differences = {key: abs(primary.metrics[key] - reproduction.metrics.get(key, float("inf"))) for key in primary.metrics}
    hashes_match = all((primary.spec_sha256 == reproduction.spec_sha256,
                        primary.code_sha256 == reproduction.code_sha256,
                        primary.data_sha256 == reproduction.data_sha256))
    metrics_match = all(differences[key] <= limits.get(key, 0.0) for key in differences)
    if not fresh_process or not clean_cache:
        status = "REPRODUCTION_HOLD"
    elif hashes_match and metrics_match:
        status = "REPRODUCED"
    else:
        status = "NON_REPRODUCIBLE"
    return ReproductionManifest(schema_version="reproduction-manifest/v1", experiment_id=primary.experiment_id,
                                primary_run_id=primary.run_id, reproduction_run_id=reproduction.run_id,
                                reproduction_agent_id=reproduction_agent_id, fresh_process=fresh_process,
                                clean_cache=clean_cache, matching_hashes=hashes_match,
                                metric_differences=differences, tolerances=limits, status=status)
