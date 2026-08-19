# Quant-Ultra Research Documentation

This directory is the authoritative entry point for research methods. The root README is intentionally limited to orientation and a high-level status summary.

| Topic | Document | Primary implementation |
|---|---|---|
| End-to-end method, risk, execution | [Methodology](METHODOLOGY.md) | `Quant-4/Main/main.py`, Phases 1-11 |
| Statistical inference | [Statistical Validation](STATISTICAL_VALIDATION.md) | `Main/statistical_validation_v2.py` |
| Data and point-in-time rules | [Data and PIT](DATA_AND_PIT.md) | `Main/data_provenance.py`, `Main/data_bus.py` |
| Experiments and benchmarks | [Governance](EXPERIMENT_AND_BENCHMARK_GOVERNANCE.md) | registries and benchmark contract |
| Factors and text/event data | [Signals](FACTOR_AND_ALTERNATIVE_DATA.md) | factor/event research modules |
| External and regime validation | [Generalization](GENERALIZATION.md) | `Main/generalization_validation.py` |
| Rebuilding results | [Reproducibility](REPRODUCIBILITY.md) | manifests and smoke runner |
| Known evidence boundaries | [Limitations](LIMITATIONS.md) | evidence gates |

Independent governed reports:

- [Comparable Benchmark Report](reports/BENCHMARK_REPORT.md)
- [Negative Results Report](reports/NEGATIVE_RESULTS_REPORT.md)
- [Reproducibility Report](reports/REPRODUCIBILITY_REPORT.md)

Runtime-generated reports and data remain untracked. A report is formal only when it has an experiment ID, code/data/parameter versions, limitations, a decision, and a report hash.
