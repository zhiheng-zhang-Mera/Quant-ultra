# -*- coding: utf-8 -*-
"""
step4/sample_weighting.py
基于复合指数时间衰减与危机窗口物理降权的样本加权引擎
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Iterable, List

logger = logging.getLogger("LabelingWeighting.Weighting")

def compute_exponential_decay_weights(
    sample_keys: Iterable[Tuple[pd.Timestamp, str]],
    t_max: pd.Timestamp,
    lambda_decay: float,
    crisis_windows: List[Tuple[str, str]],
    crisis_noise_weight: float
) -> Dict[Tuple[pd.Timestamp, str], float]:
    """
    计算满足 Flow-Pro 4.2 约定的复合样本权重 w_t = w_time * w_noise
    w_time = e^(-λ * Δt) ，Delta_t 为与当前训练折最大截面的天数跨度
    若样本日期陷入危机窗口（如2015/2020），w_noise 刚性下调至 0.10，确保常态规律提炼。
    """
    logger.info("开始执行复合指数时间衰减与危机窗口交叉审计加权...")
    sample_weights = {}
    
    # 时区擦除与刚性时间对齐
    t_max_ts = pd.Timestamp(t_max).replace(tzinfo=None)
    
    # 预解析危机视窗边界
    parsed_crisis_ranges = []
    for start_str, end_str in crisis_windows:
        parsed_crisis_ranges.append((pd.Timestamp(start_str), pd.Timestamp(end_str)))

    for (date, sym) in sample_keys:
        date_ts = pd.Timestamp(date).replace(tzinfo=None)
        delta_days = (t_max_ts - date_ts).days
        
        # 刚性边界哨兵：防止负值导致前瞻性信息污染
        if delta_days < 0:
            delta_days = 0
            
        w_time = np.exp(-lambda_decay * delta_days)
        
        # 判断时间点是否命中海内外系统性危机视窗
        in_crisis_zone = False
        for start_dt, end_dt in parsed_crisis_ranges:
            if start_dt <= date_ts <= end_dt:
                in_crisis_zone = True
                break
                
        w_noise = crisis_noise_weight if in_crisis_zone else 1.0
        
        # 复合层融合
        sample_weights[(date, sym)] = float(w_time * w_noise)

    if sample_weights:
        logger.info(f"复合权重加权完成。极值视窗: [{min(sample_weights.values()):.6f}, {max(sample_weights.values()):.6f}]")
    else:
        logger.warning("⚠️ 警告：加权样本总池为空，请检查前置标签生成器！")
        
    return sample_weights