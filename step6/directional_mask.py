import logging
from datetime import datetime
from .utils import _get_features_for_date

logger = logging.getLogger("PositionSizing.DirectionalMask")

def step_m_1_directional_mask(context: dict, date: datetime) -> dict:
    assets = context['assets']
    clf = context.get('direction_classifier')
    gamma = context.get('gamma_star', 0.5)
    if clf is None:
        logger.critical("[Orchestrator 断层] 全局上下文中找不到训练完成的 direction_classifier 机器模型！")
        raise RuntimeError("direction_classifier 未找到。")
    borrowable = context.get('borrowable_today', set())
    masks = {}
    
    long_n, short_n, neutral_n = 0, 0, 0
    for sym in assets:
        feat = _get_features_for_date(sym, date, context)
        if feat is None:
            masks[sym] = 0
            neutral_n += 1
            continue
        try:
            prob = clf.predict_proba(feat.reshape(1, -1))[0]
            prob_neg, prob_zero, prob_pos = prob[0], prob[1], prob[2]
            if prob_pos >= gamma and prob_pos > prob_neg:
                masks[sym] = 1
                long_n += 1
            elif prob_neg >= gamma and prob_neg > prob_pos:
                if sym in borrowable:
                    masks[sym] = -1
                    short_n += 1
                else:
                    masks[sym] = 0
                    neutral_n += 1
            else:
                masks[sym] = 0
                neutral_n += 1
        except Exception as e:
            logger.warning(f"[{sym}] 模型预测概率矩阵转换崩溃: {e}，降级为中性。")
            masks[sym] = 0
            neutral_n += 1
            
    logger.debug(f"[信号转换面板] 日期: {date.strftime('%Y-%m-%d')} | 预测资产数: {len(assets)} | 信号分布 -> 多头: {long_n}, 空头(融券合规): {short_n}, 中性: {neutral_n}")
    context['directional_symbol_masks'] = masks
    return masks