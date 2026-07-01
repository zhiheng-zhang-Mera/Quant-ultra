# -*- coding: utf-8 -*-
"""
step4/sample_weighting.py
基于指数时间衰减的样本加权器
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Iterable

logger = logging.getLogger("LabelingWeighting.Weighting")

def compute_exponential_decay_weights(
    sample_keys: Iterable[Tuple[pd.Timestamp, str]],
    t_max: pd.Timestamp,
    lambda_decay: float
) -> Dict[Tuple[pd.Timestamp, str], float]:
    """
    根据样本日期与最大训练截面的距离，计算指数级时间衰减权重 w_t = e^(-λ * Δt)
    """
    logger.info("开始计算指数时间衰减权重...")
    sample_weights = {}
    
    # 刚性时间类型对齐
    t_max_dt = t_max.to_pydatetime() if isinstance(t_max, pd.Timestamp) else t_max

    for (date, sym) in sample_keys:
        date_dt = date.to_pydatetime() if isinstance(date, pd.Timestamp) else date
        delta_days = (t_max_dt - date_dt).days
        
        # 刚性防止负时间溢出（防止前瞻泄露污染）
        if delta_days < 0:
            delta_days = 0
            
        w = np.exp(-lambda_decay * delta_days)
        sample_weights[(date, sym)] = float(w)

    if sample_weights:
        logger.info(f"样本权重分配完成，权重范围: [{min(sample_weights.values()):.6f}, {max(sample_weights.values()):.6f}]")
    else:
        logger.warning("⚠️ 样本权重池为空，请检查前置标签生成状态！")
        
    return sample_weights