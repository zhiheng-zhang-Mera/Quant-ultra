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
    'individual_account_equity': 10000000.0, # 个人账户分配可用总权益名义本金基准底座 (1000万)
    'allow_native_solver': True,
    'native_solver_min_memory_gb': 2.0,
    'parallel_time_slices': True,
    'parallel_slice_min_dates': 4,
    'parallel_time_slice_cap': 8,
    'python_solver_iterations': 250,
    'python_solver_step': 0.05,
    # ---- 8-9 更新计划任务二/三：整手约束 + 最小调仓阈值 + 多日复合调仓 ----
    'board_lot_rounding': True,          # 目标权重按整手(100/200股)对齐，消除 Phase6/7 对账漂移
    'min_trade_weight': 0.005,           # 单资产权重变化低于 0.5% 不交易，抑制高频微调摩擦
    'rebalance_every': 1,                # 每隔 N 个交易日才重新求解（主流程通过 rotation_rebalance_days 覆盖为 5）
}
