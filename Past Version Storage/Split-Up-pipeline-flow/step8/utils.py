import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger("AuditStressTest.Utils")

def safe_get_shadow(context: Dict, key: str, default: Any = None) -> Any:
    """提取全局上下文的影子副本，防止游离操作污染内存引用"""
    val = context.get(key, default)
    if val is None and default is None:
        logger.error(f"❌ [数据断层] 内存总线中缺失核心依赖键: '{key}'")
        raise KeyError(f"Missing required key '{key}' in pipeline context.")
    
    if isinstance(val, (pd.Series, pd.DataFrame)):
        return val.copy()
    return val

def compute_max_drawdown(nav: pd.Series) -> float:
    """纯前向最大回撤计算器"""
    if len(nav) < 2:
        return 0.0
    try:
        peak = nav.cummax()
        dd = (nav - peak) / peak
        return float(dd.min())
    except Exception as e:
        logger.error(f"❌ 最大回撤矩阵运算溢出: {str(e)}")
        return 0.0