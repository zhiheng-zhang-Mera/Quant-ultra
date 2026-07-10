# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_8 Auditing & Stress-Test Core Utilities
"""
import logging
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger("AuditStressTest.Utils")

def safe_get_shadow(context: Dict, key: str, default: Any = None) -> Any:
    """
    提取全局上下文的影子副本。
    返回 DataFrame 或 Series 的硬拷贝，物理阻断游离操作对主内存引用链的毒化或篡改。
    """
    val = context.get(key, default)
    if val is None and default is None:
        logger.critical("[OP] Intercept Memory Retrieval | [SOURCE] Pipeline Context Registry | [RESULT] Key Absent Exception: %s | [SIGNIFICANCE] Hard halt prevents down-stream null pointer propagation", key)
        logger.critical("[操作] 拦截内存变量检索 | [来源] 流水线主注册上下文 | [结果] 核心依赖键缺失异常: %s | [意义] 强行挂起中断，严防下游因空指针引用造成计算链脱轨崩溃")
        raise KeyError(f"Missing required key '{key}' in pipeline context.")
    
    if isinstance(val, (pd.Series, pd.DataFrame)):
        return val.copy()
    return val

def compute_max_drawdown(nav: pd.Series) -> float:
    """标准前向最大回撤计算器，抵御时间序列异常波动造成的数值溢出"""
    if len(nav) < 2:
        return 0.0
    try:
        peak = nav.cummax()
        dd = (nav - peak) / peak
        return float(dd.min())
    except Exception as e:
        logger.error("[OP] Calculate Maximum Drawdown | [SOURCE] Floating NAV Series | [RESULT] Arithmetic Overflow: %s | [SIGNIFICANCE] Numerical instability fallback to zero default", str(e))
        logger.error("[操作] 求解历历最大回撤 | [来源] 漂移净值时间序列 | [结果] 算术矩阵运算溢出: %s | [意义] 捕获非稳态数学不稳定性，自愈降级默认为零")
        return 0.0