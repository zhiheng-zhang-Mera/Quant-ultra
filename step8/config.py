"""
Phase 8 Local Configuration
Fully compliant with Final-Flow.md [2026 Production Release]
"""

DEFAULT_CONFIG = {
    "min_coverage": 0.935,
    "christoffersen_pval_threshold": 0.01,
    "sharpe_threshold": 0.50,
    "dsr_pval_threshold": 0.05,
    "stress_scenarios": {
        "2015_liq": ("2015-06-01", "2015-09-30"),
        "2016_meltdown": ("2016-01-01", "2016-02-29"),
        "2024_microcap": ("2024-01-01", "2024-02-29"),
    },
    "min_samples_for_dsr": 20,
    "stock_cap_pct": 0.045,
}