"""
Phase 6: Position Sizing - Local Parameters Configuration (Pure Long Individual Version)
"""

DEFAULT_CONFIG = {
    'tau_BL': 0.02,
    'omega_min': 1e-8,
    'omega_max': 0.01,
    'width_halflife': 21,
    'gamma_risk_initial': 2.5,
    'sector_limit': 0.3,
    'epsilon': 0.001,
    'transaction_cost_coeff': 0.0003,
    'lookback_cov': 252,           # 协方差估计回溯期
    'lookback_adv': 20,            # 流动性防御ADV估计回溯期
    'individual_account_equity': 10000000.0  # 个人账户可用总权益本金基准底座
}