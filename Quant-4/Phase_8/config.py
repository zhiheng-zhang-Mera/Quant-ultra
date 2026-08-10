# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_8 Configurations & Auditing Standards
"""

DEFAULT_CONFIG = {
    # 阶段 8.3 Christoffersen 似然比覆盖率检验合规阈值
    "min_coverage": 0.935,                  # 95% 置信度 VaR 的无条件覆盖率保底容忍线 (93.5%)
    "christoffersen_pval_threshold": 0.01,  # VaR 独立性 LR 检验的显著性 p 值门槛 (1%)
    # 8-9 更新计划容错保护：违规样本过少（<10 次）时卡方近似不可靠，
    # 独立性 LR 检验退化为无聚集证据（p=1.0），仅以无条件覆盖率判定。
    "min_violations_for_lr_test": 10,
    
    # 阶段 8.1 统计学夏普多重试验 DSR 审计门槛
    "sharpe_threshold": 0.50,              # 策略期望的目标夏普比率底线 (0.50)
    "dsr_pval_threshold": 0.05,            # DSR 统计检验拒绝零假设的 p 值显著性门槛 (5%)
    "min_samples_for_dsr": 20,             # 启动 DSR 计算所需的最小有效净值观测点数 (20天)
    
    # 阶段 8.4 历史极端系统性体制崩溃测试窗口
    "stress_scenarios": {
        "2015_liq": ("2015-06-01", "2015-09-30"),       # 2015 A 股高杠杆异常波动及流动性挤踏
        "2016_meltdown": ("2016-01-01", "2016-02-29"),  # 2016 A 股熔断机制极限压力测试
        "2024_microcap": ("2024-01-01", "2024-02-29"),  # 2024 量化微盘股流动性危机
    },
    
    # 阶段 8.5 个人版静态本金底座与流动性安全垫硬指标
    "total_equity": 10000000.0,            # 个体账户既定本金可用总权益名义底座 (1000万元)
    "max_participation_threshold": 0.05,   # 极端历史截面中个股最大调仓换手参与率警戒红线 (5%)
}
