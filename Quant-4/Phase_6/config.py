# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_6 Position Sizing Config & Threshold boundaries
"""

DEFAULT_CONFIG = {
    'tau_BL': 0.02,                         # 先验协方差矩阵缩放因子
    'omega_min': 1e-8,                      # 置信度协方差最低防除零溢出边界
    'omega_max': 0.01,                      # 置信度协方差最高降级兜底边界
    'width_halflife': 21,                   # 预测区间不确定性指数平滑半衰期
    'gamma_risk_initial': 2.5,              # 凸优化器初始风险厌恶系数
    'sector_limit': 0.3,                    # 单行业最大横截面敞口暴露上限 (30%)
    'epsilon': 0.001,                       # 凸优化收敛收敛容忍度
    'transaction_cost_coeff': 0.0003,       # 单边调仓换手交易摩擦冲击惩罚系数
    'lookback_cov': 252,                    # 稳健协方差滚动历史回溯期 (252天)
    'lookback_adv': 20,                     # 个人流动性硬约束平均成交额回溯期 (20天)
    'individual_account_equity': 10000000.0  # 个人账户分配可用总权益名义本金基准底座 (1000万)
}