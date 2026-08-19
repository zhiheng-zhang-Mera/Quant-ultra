# Representative Reproducibility Report

| Field | Value |
|---|---|
| Report type | REPRODUCIBILITY |
| Experiment ID | EXP-20260819-US-ETF-REPRO-001 |
| Status | PASS |
| Code version | git:9ce8d49542b18e48c07fef195067e1aba82f4daf |
| Data version | sha256:0b9d6384f81b3685bc3c1b356056116d696b127eaa5755b047c733988dfd70f2 |
| Parameter version | sha256:6a87d5e16797662a7fe23d6fa5ed55f2cfd03487bfce665cdca231069cf03647 |
| Generated at | 2026-08-19T05:40:00+00:00 |
| Report SHA-256 | b3f4c0ffde1dfb41c10c126e74a49d77c983d5b9042a0a569a125b432271fbc7 |

## Summary

Ten raw/cleaned US ETF datasets were content-addressed and the frozen external-market experiment was executed twice with identical governed metrics.

## Metrics

| Metric | Value |
|---|---|
| annual_return | 0.2834394999713368 |
| artifact_manifest_sha256 | fecbe1fbffcd301ef322cf01dc427d1c3fdd2fabcdd7ea728eb850e55e6a826c |
| datasets | 10 |
| max_drawdown | -0.6176202507968612 |
| maximum_repeat_difference | 0 |
| numeric_tolerance | 1e-12 |
| sharpe | 0.8493714932892283 |

## Evidence

- Data-manifest verification passed for all ten datasets.
- Input and output artifact hashes verified.
- GitHub Actions run 32220421848 passed tests, dependency integrity, compilation and production readiness.

## Limitations

- Runtime data is local and intentionally not committed.
- Yahoo-adjusted history can be revised after download.
- Repeatability is engineering evidence, not investment validity.

## Decision

PASS the representative reproducibility gate; retain the external strategy conclusion independently as TRANSFER_MIXED.
