"""
Quant-Ultra Flow - Step 1 Local Configuration Thresholds (Long-Horizon Edition)
"""
CONFIG = {
    # 时空纵深核心配置（由原先的 400 天短周期升级为自 2015 年开始的饱满大本营）
    "START_DATE": "2015-01-01",           # 全量数据抓取与对齐的物理起点
    
    "ADV_WINDOW": 20,                     # ADV 计算滚动窗口
    "MIN_ADV_THRESHOLD": 1e7,             # 最小流动性门槛 (1000万)
    "IPO_SAFETY_DAYS": 20,                # 次新股过滤天数
    "MAX_PARTICIPATION_RATE": 0.05,       # 最大市场参与率
    "EXPECTED_TURNOVER": 0.05,            # 预期换手率
    "MAX_SINGLE_STOCK_WEIGHT": 0.05,      # 单股最大权重
    "CACHE_EXPIRE_DAYS": 7,               # 缓存失效周期
    
    # 防封锁安全降级风控（由于拉取 8 年历史数据包极大，调低线程以严防触发防火墙 RST 硬拦截）
    "DOWNLOAD_WORKERS": 4,                 # 固定降级线程数（从 8 调降至 4，安全脱水）
    "ADAPTIVE_MIN_WORKERS": 1,            # 自适应最小线程数
    "ADAPTIVE_MAX_WORKERS": 6,            # 自适应最大线程数（上限收敛，避免锁步踩踏）
    "TARGET_CPU_UTIL": 0.65,              # 目标 CPU 利用率
    "CHECK_INTERVAL": 2.0,                # 自适应控制检测步长
    "ERROR_RATE_THRESHOLD": 0.08,         # 容忍的最大错误率门槛
}