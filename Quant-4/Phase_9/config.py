"""
Quant-Ultra Flow - Step 9 MLOps & Production Configuration
Isolates all operational thresholds, safety redlines, and alert parameters.
"""
import os
from pathlib import Path
from Main.env_config import PROJECT_ROOT

# 影子对账硬阈值
MAE_THRESHOLD = 1e-5
WATCHDOG_TIMEOUT = 30  # 秒

# PSI 稳定性与全量重训红线
PSI_THRESHOLD = 0.25
PSI_CONSECUTIVE_DAYS = 5
SMOOTHING_PERIOD = 25  # 新老模型线性步进切换周期（天）

# 嵌套风控预警分位数
CROWDED_CORR_THRESHOLD = 0.95     # 条件 A：风格相关性 95% 分位数
VOL_COMPRESS_QUANTILE = 0.10      # 条件 B：滚动残差波动率最低 10% 分位数
CROWDED_RISK_CAP = 0.20           # 双重触发时纯现货多头的最大持仓上限（20%）

LOOKBACK_WINDOW_PSI = 60          # PSI 基准分布历史回溯窗口
VOLATILITY_WINDOW = 20            # 净值滚动残差波动率计算窗口

# 严格响应 README 规范的会话缓存双格式本地存储路径
CACHE_PARQUET_DIR = PROJECT_ROOT / "Phase_Result" / "parquet" / "Phase_9"
CACHE_FEATHER_DIR = PROJECT_ROOT / "Phase_Result" / "feather" / "Phase_9"

os.makedirs(CACHE_PARQUET_DIR, exist_ok=True)
os.makedirs(CACHE_FEATHER_DIR, exist_ok=True)

# 统一封装 MLOps 顶层核心配置契约，阻断下游组件 ImportError 触雷
DEFAULT_MLOPS_CONFIG = {
    'reconciliation_mae_ceiling': MAE_THRESHOLD,
    'volatility_window': VOLATILITY_WINDOW,
    'vol_compress_quantile': VOL_COMPRESS_QUANTILE,
    'crowded_corr_threshold': CROWDED_CORR_THRESHOLD,
    'enforce_crowded_allocation_cap': CROWDED_RISK_CAP,
    'lookback_psi_window': LOOKBACK_WINDOW_PSI,
    'psi_drift_crit_threshold': PSI_THRESHOLD,
    'psi_consecutive_days_trigger': PSI_CONSECUTIVE_DAYS,
    'model_smoothing_period': SMOOTHING_PERIOD
}
