# step2/config.py

# 默认五段式顺次切分比例（基于交易日序号令牌对齐拓扑）
DEFAULT_SLICING_RATIOS = [0.50, 0.60, 0.70, 0.85]

# 保底与禁运最低周期（单位：交易日序号差）
DEFAULT_HOLDING_PERIOD = 5
DEFAULT_EMBARGO_MIN = 5

# ACF 分布式边缘自相关算子约束
MIN_DAYS_FOR_ACF = 100        # 单票有效实际全收益率时序行数下限
ACF_NLAGS = 10                # 观测最大自相关滞后阶数
ACF_ALPHA = 0.05              # 95% 显著性置信水平
MAX_SAMPLE_ASSETS = 50        # 每类节点边缘资产抽样上限，防跨域超大高并发I/O阻塞

# 联邦熔断与隔离安全哨兵阈值
MIN_SLICE_WARN_LEN = 10       # 触发审计总线预警的最低窗口长度
MIN_SLICE_CRIT_LEN = 5        # 触发硬物理熔断的底线长度（RuntimeError）