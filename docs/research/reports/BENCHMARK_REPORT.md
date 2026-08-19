# Comparable Benchmark Report

| Field | Value |
|---|---|
| Report type | BENCHMARK |
| Experiment ID | EXP-20260819-BENCHMARK-001 |
| Status | HOLD |
| Code version | git:4c57019744b592fd228cda5b9f226b1f39ec4b60 |
| Data version | NOT_RUN |
| Parameter version | comparable-benchmark/v1 |
| Generated at | 2026-08-19T06:05:00+00:00 |
| Report SHA-256 | 23b87b3202017e38bb659b0b5e1c656b29e2451fccee7d7288009de5d5a7b022 |

## Summary

The comparable benchmark contract and automatic manifest are implemented and tested, but no new real-data same-contract benchmark run was performed in this strengthening wave.

## Metrics

| Metric | Value |
|---|---|
| performance_claim_status | HOLD |
| required_categories | 4 |
| required_quant_ultra_versions | 2 |
| unit_tests_at_wave_1 | 202 |

## Evidence

- Quant-4/Main/benchmark_governance.py enforces one index, data version, costs, capital and constraints.
- Quant-4/tests/test_benchmark_governance.py covers complete and fail-closed contracts.
- GitHub Actions run 32217606288 passed Wave 1.

## Limitations

- Capability tests are not a real-market comparative result.
- External open-source rankings remain auxiliary only.
- A future formal run must supply the immutable data manifest.

## Decision

HOLD all new benchmark performance claims until a registered real-data run emits the governed manifest.
