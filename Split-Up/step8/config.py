"""
Phase 8 Local Configuration
Fully compliant with Flow-Pro.md [2026 Pure Long Spot Release]
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
    
    # 刚性对齐 Flow-Pro 8.5 个人版静态本金底座与流动性安全垫配置
    "total_equity": 10000000.0,            # 个体账户既定纯现金本金底座（默认1000万元）
    "max_participation_threshold": 0.05,   # 静态大单参与率（Participation Rate）安全警戒硬红线（5%）
}