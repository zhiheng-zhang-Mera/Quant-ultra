"""
Phase 6: Position Sizing - Local Parameters Configuration
"""

DEFAULT_CONFIG = {
    'tau_BL': 0.02,
    'omega_min': 1e-8,
    'omega_max': 0.01,
    'width_halflife': 21,
    'gamma_risk_initial': 2.5,
    'max_leverage': 1.0,
    'sector_limit': 0.3,
    'epsilon': 0.001,
    'transaction_cost_coeff': 0.0003,
    'short_upper_limit': -0.25, # 空头仓位刚性上限
    'max_shares_pct': 0.045,     # 4.5% 总股本硬性权重约束
    'lookback_cov': 252          # 协方差估计回溯期
}