import logging
from datetime import datetime
import numpy as np
import pandas as pd
from .utils import _get_features_for_date

logger = logging.getLogger("PositionSizing.DirectionalMask")

def step_m_1_directional_mask(context: dict, date: datetime) -> dict:
    assets = context['assets']
    clf = context.get('direction_classifier')
    gamma = context.get('gamma_star', 0.5)
    
    if clf is None:
        logger.critical("[Orchestrator 断层] 全局上下文中找不到训练完成的 direction_classifier 机器学习模型！")
        raise RuntimeError("direction_classifier 未找到。")
        
    masks = {}
    long_n, neutral_n = 0, 0
    
    for sym in assets:
        logger.info(f"[Step M.1] 处理资产 {sym} 的方向性信号掩码, 总资产数: {len(assets)}, 当前索引: {assets.index(sym)+1}/{len(assets)}")
        feat = _get_features_for_date(sym, date, context)
        if feat is None:
            masks[sym] = 0
            neutral_n += 1
            continue
        try:
            # 🍏 方式 A：如果训练时使用了特征名，将一维特征动态包装为带有与模型一致的列名的 DataFrame
            if hasattr(clf, "feature_name_") and clf.feature_name_ is not None:
                feat_df = pd.DataFrame(feat.reshape(1, -1), columns=clf.feature_name_)
                prob = clf.predict_proba(feat_df)[0]
            else:
                # 🍏 方式 B：若不确定，采用局部上下文看门狗，强行在预测瞬间屏蔽警告
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    prob = clf.predict_proba(feat.reshape(1, -1))[0]
            # 兼容处理分类器的输出概率维度：[负向收益概率, 中性概率, 正向多头概率]
            prob_neg = prob[0]
            prob_pos = prob[2] if len(prob) == 3 else prob[1]
            
            # 严格遵循多头方向概率防御过滤：概率达到gamma阀值且优于负向概率
            if prob_pos >= gamma and prob_pos > prob_neg:
                masks[sym] = 1
                long_n += 1
            else:
                masks[sym] = 0
                neutral_n += 1
        except Exception as e:
            logger.warning(f"[{sym}] 模型预测概率矩阵转换崩溃: {e}，降级为中性。")
            masks[sym] = 0
            neutral_n += 1
            
    logger.debug(f"[信号转换面板] 日期: {date.strftime('%Y-%m-%d')} | 预测资产数: {len(assets)} | 信号分布 -> 现货多头: {long_n}, 中性锁仓: {neutral_n}")
    context['directional_symbol_masks'] = masks
    return masks