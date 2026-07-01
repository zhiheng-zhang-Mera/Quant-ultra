# -*- coding: utf-8 -*-
"""
step4/config.py
Phase 4 局部超参数与全局防熔断常数定义
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