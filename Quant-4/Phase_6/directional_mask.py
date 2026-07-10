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
    
    # --- 提速重构：组装高维特征批次，彻底消灭循环内 predict ---
    valid_syms = []
    valid_feats = []
    
    for sym in assets:
        feat = _get_features_for_date(sym, date, context)
        if feat is not None:
            valid_syms.append(sym)
            valid_feats.append(feat)
        else:
            masks[sym] = 0
            neutral_count += 1
            
    if valid_feats:
        # 将列表堆叠为 2D NumPy 矩阵，执行一次性全局推理
        X_batch = np.vstack(valid_feats)
        try:
            if hasattr(clf, "feature_name_") and clf.feature_name_ is not None:
                feat_df = pd.DataFrame(X_batch, columns=clf.feature_name_)
                probs = clf.predict_proba(feat_df)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    probs = clf.predict_proba(X_batch)
                    
            # 批量解析预测结果
            for i, sym in enumerate(valid_syms):
                prob = probs[i]
                prob_neg = prob[0]
                prob_pos = prob[2] if len(prob) == 3 else prob[1]
                
                if prob_pos >= gamma and prob_pos > prob_neg:
                    masks[sym] = 1; long_count += 1
                else:
                    masks[sym] = 0; neutral_count += 1
        except Exception as e:
            logger.debug(f"Batch prediction failed, fallback to zero mask: {e}")
            for sym in valid_syms:
                masks[sym] = 0; neutral_count += 1
            
    # 性能优化：引入交易日级别的动态看门狗
    if not hasattr(step_m_1_directional_mask, "_call_count"):
        step_m_1_directional_mask._call_count = 0
    step_m_1_directional_mask._call_count += 1
    
    if step_m_1_directional_mask._call_count == 1 or step_m_1_directional_mask._call_count % 100 == 0:
        logger.info("[操作] 执行条件胜率门控筛选 | [来源] 阶段5活体机器学习分类 network | [结果] 多头激发计入数: %s, 判定中性拦截数: %s | [意义] 完成横截面预测胜率置信度合规初筛 (当前累计执行: %d 天)", long_count, neutral_count, step_m_1_directional_mask._call_count)
        
    return masks