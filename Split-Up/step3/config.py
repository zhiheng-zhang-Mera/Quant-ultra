# -*- coding: utf-8 -*-
"""
step3/config.py
收拢 Phase 3 的所有计算窗口、线程分配及局部阈值
"""

# 数据源并发加载配置
DEFAULT_START_YEAR = 2010
DEFAULT_MAX_WORKERS = 10
DEFAULT_PROGRESS_INTERVAL = 20

# 波动率体制标签参数
DEFAULT_VOL_WINDOW = 20

# 特征提取维度名称定义
FEATURE_COLUMNS = ["Mom_1D", "Mom_5D", "Mom_20D", "GK_Vol", "Turnover_Shock"]