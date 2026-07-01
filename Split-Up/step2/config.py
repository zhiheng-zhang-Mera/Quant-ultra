# step2/config.py

# 默认五段式物理切分比例
DEFAULT_SLICING_RATIOS = [0.50, 0.60, 0.70, 0.85]

# 保底与禁运默认周期（单位：交易日）
DEFAULT_HOLDING_PERIOD = 5
DEFAULT_EMBARGO_MIN = 5

# ACF 统计学计算约束
MIN_DAYS_FOR_ACF = 100       # 至少需要100天有效收益率样本
ACF_NLAGS = 10               # 自相关最大观测滞后阶数
ACF_ALPHA = 0.05             # 95% 显著性水平置信区间
MAX_SAMPLE_ASSETS = 100      # 最大流式抽样资产数，防止全池扫描拖慢速度

# 熔断与预警硬性阈值
MIN_SLICE_WARN_LEN = 10      # 切片样本少于 10 天触发审计日志预警
MIN_SLICE_CRIT_LEN = 5       # 切片样本少于 5 天触发物理熔断（RuntimeError）