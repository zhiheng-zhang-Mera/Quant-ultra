# -*- coding: utf-8 -*-
"""
step4/config.py
Phase 4 局部超参数、危机时间视窗与全局防熔断常数固化层
"""

# 指数时间衰减因子 λ
DEFAULT_LAMBDA_DECAY = 0.01

# 波动率计算滚动窗口与最小有效观测数
DEFAULT_VOL_WINDOW = 20
DEFAULT_MIN_VOL_OBS = 5

# 自适应三屏障波动率乘数
DEFAULT_THRESHOLD_MULTIPLIER = 0.5

# 全局残值波动率兜底（防止除零或停牌造成的体制真空）
GLOBAL_VOL_MEDIAN_FALLBACK = 0.02

# 海内外重大系统性危机历史窗口注册表（Flow-Pro 4.2 约束，强制降权以过滤黑天鹅噪音）
CRISIS_WINDOWS = [
    ("2007-10-01", "2009-04-30"),  # 2008全球次贷金融危机
    ("2015-06-12", "2016-02-29"),  # 2015A股杠杆异常波动及熔断测试
    ("2020-01-20", "2020-04-30")   # 2020全球疫情流动性冲击
]

# 危机窗口样本刚性降权乘数
CRISIS_NOISE_WEIGHT = 0.10