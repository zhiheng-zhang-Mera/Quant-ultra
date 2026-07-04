# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 4.2: Exponential Decay & Crisis Isolation Sample Weighting Engine
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
    
    sample_weights = {}
    t_max_ts = pd.Timestamp(t_max).replace(tzinfo=None)
    parsed_crisis_ranges = []
    
    for start_str, end_str in crisis_windows:
        parsed_crisis_ranges.append((pd.Timestamp(start_str), pd.Timestamp(end_str)))

    for (date, sym) in sample_keys:
        date_ts = pd.Timestamp(date).replace(tzinfo=None)
        delta_days = (t_max_ts - date_ts).days
        if delta_days < 0: delta_days = 0
            
        w_time = np.exp(-lambda_decay * delta_days)
        in_crisis_zone = False
        for start_dt, end_dt in parsed_crisis_ranges:
            if start_dt <= date_ts <= end_dt:
                in_crisis_zone = True; break
                
        w_noise = crisis_noise_weight if in_crisis_zone else 1.0
        sample_weights[(date, sym)] = float(w_time * w_noise)

    if sample_weights:
        logger.info("[OP] Intercept Crisis Black Swan Nodes | [SOURCE] Conformal Temporal Windows Map | [RESULT] Enforced weights scope: [%.6f, %.6f] | [SIGNIFICANCE] Downweights macro systematic anomalies to ensure stable market rules distillation", min(sample_weights.values()), max(sample_weights.values()))
        logger.info("[操作] 交叉拦截黑天鹅灾难样本 | [来源] 合规系统性危机注册历史视窗 | [结果] 固化复合样本权重区间: [%.6f, %.6f] | [意义] 强行调降宏观系统性突变极值的置信度权重，确保机器学习模型提炼常态规律")
    else:
        logger.warning("[OP] Evaluate Sample Weights Pool | [SOURCE] Label Keys Tracker | [RESULT] Empty weighting registry map | [SIGNIFICANCE] CRITICAL WARNING: Preceding step label generation failed to yield rows")
        logger.warning("[操作] 评估加权样本总体规模 | [来源] 标签要素特征主轴 | [结果] 加权结果注册映射表为空 | [意义] 核心核心警报：前置步骤标签生成器未能剥离出任何有效行，总线面临真空")
        
    return sample_weights