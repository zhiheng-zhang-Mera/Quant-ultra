# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 6.1: Conformal Machine Learning Directional Probability Gating Filter
"""
import logging
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from Phase_6.utils import _get_features_for_date

logger = logging.getLogger("PositionSizing.DirectionalMask")

def step_m_1_directional_mask(context: dict, date: datetime) -> dict:
    assets = context['assets']
    clf = context.get('direction_classifier')
    gamma = context.get('gamma_star', 0.5)
    
    if clf is None: raise RuntimeError("Context classifier missing.")
        
    masks = {}
    long_count, neutral_count = 0, 0
    
    for sym in assets:
        feat = _get_features_for_date(sym, date, context)
        if feat is None: masks[sym] = 0; neutral_count += 1; continue
            
        try:
            # 轻量级自适应包装特征列名，防止 LightGBM 抛出名义缺失非致命性警告
            if hasattr(clf, "feature_name_") and clf.feature_name_ is not None:
                feat_df = pd.DataFrame(feat.reshape(1, -1), columns=clf.feature_name_)
                prob = clf.predict_proba(feat_df)[0]
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    prob = clf.predict_proba(feat.reshape(1, -1))[0]
            
            prob_neg = prob[0]
            prob_pos = prob[2] if len(prob) == 3 else prob[1]
            
            # Flow-Pro 6.1: 联合条件不等式审查。激发条件：多头胜率超越激活阈值γ* 且 绝对占优于空头风险
            if prob_pos >= gamma and prob_pos > prob_neg:
                masks[sym] = 1; long_count += 1
            else:
                masks[sym] = 0; neutral_count += 1
        except Exception as e:
            logger.debug(f"Direction asset check fail on {sym}: {e}")
            masks[sym] = 0; neutral_count += 1
            
    # 性能优化：引入交易日级别的动态看门狗，每 100 天释放一次控制台 stdout 锁
    if not hasattr(step_m_1_directional_mask, "_call_count"):
        step_m_1_directional_mask._call_count = 0
    step_m_1_directional_mask._call_count += 1
    
    if step_m_1_directional_mask._call_count == 1 or step_m_1_directional_mask._call_count % 100 == 0:
        logger.info("[OP] Filter Conditional odds Gates | [SOURCE] Phase_5 Real-time Classifier Model | [RESULT] Total Long Checklist: %s, Neutralized Shunts: %s | [SIGNIFICANCE] Forms categorical state gates for multi-stage allocation (Total days: %d)", long_count, neutral_count, step_m_1_directional_mask._call_count)
        logger.info("[操作] 执行条件胜率门控筛选 | [来源] 阶段5活体机器学习分类 network | [结果] 多头激发计入数: %s, 判定中性拦截数: %s | [意义] 完成横截面预测胜率置信度合规初筛，为级联矩阵提供条件状态掩码 (当前累计执行: %d 天)", long_count, neutral_count, step_m_1_directional_mask._call_count)
        
    return masks