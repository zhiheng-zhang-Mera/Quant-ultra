# -*- coding: utf-8 -*-
"""
step3/config.py
收拢 Phase 3 的所有计算窗口、线程分配及分层分域特征注册表
"""

# 数据源并发加载配置
DEFAULT_START_YEAR = 2010
DEFAULT_MAX_WORKERS = 10
DEFAULT_PROGRESS_INTERVAL = 20

# 波动率体制标签参数
DEFAULT_VOL_WINDOW = 20

# 特征提取维度名称定义（模块一：共享特征层 - 跨市场通用因子，明文直通）
FEATURE_COLUMNS = ["Mom_1D", "Mom_5D", "Mom_20D", "GK_Vol", "Turnover_Shock"]

# 模块二：垂直私有特征层独立注册表（各节点独立维护特有制度数据，严禁跨出域外）
A_PRIVATE_COLUMNS = ["Limit_Price_Matrix", "ST_Status", "Free_Float_Cap", "Northbound_Flow", "Dragon_Tiger_Seats"]
US_PRIVATE_COLUMNS = ["Short_Interest", "VIX_Implied", "Earnings_Window", "Insider_Trading"]