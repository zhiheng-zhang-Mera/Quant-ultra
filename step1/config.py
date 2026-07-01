"""
Quant-Ultra Flow - Step 1 Local Configuration Thresholds
"""
CONFIG = {
    "ADV_WINDOW": 20,                     # ADV 计算滚动窗口
    "MIN_ADV_THRESHOLD": 1e7,             # 最小流动性门槛 (1000万)
    "IPO_SAFETY_DAYS": 20,                # 次新股过滤天数
    "MAX_PARTICIPATION_RATE": 0.05,       # 最大市场参与率
    "EXPECTED_TURNOVER": 0.05,            # 预期换手率
    "MAX_SINGLE_STOCK_WEIGHT": 0.05,      # 单股最大权重
    "CACHE_EXPIRE_DAYS": 7,               # 缓存失效周期
    "DOWNLOAD_WORKERS": 8,                 # 固定降级线程数
    "ADAPTIVE_MIN_WORKERS": 2,            # 自适应最小线程数
    "ADAPTIVE_MAX_WORKERS": 20,           # 自适应最大线程数
    "TARGET_CPU_UTIL": 0.70,              # 目标 CPU 利用率
    "CHECK_INTERVAL": 2.0,                # 自适应控制检测步长
    "ERROR_RATE_THRESHOLD": 0.10,         # 容忍的最大错误率门槛
}