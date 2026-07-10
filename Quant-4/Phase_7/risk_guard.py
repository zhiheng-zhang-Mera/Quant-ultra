# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_7 Micro-Liquidity Individual Capacity Sentinel
"""
import logging
import numpy as np

logger = logging.getLogger("FSMBacktest.RiskGuard")

def compute_individual_position_limit(sym, nav, current_date, data_bus, config):
    """
    个人流动性与资金量双重硬约束防线守门狗。
    单票最大持仓权重上限 = min(10%, 10% * ADV_20 / 账户可用总权益)
    """
    if nav <= 0:
        return 0.10

    try:
        # 优先使用 ADV_20，降级到 ADV
        adv_20 = data_bus.query_by_pit(sym, current_date, "adv_20")
        if adv_20 is None or np.isnan(adv_20) or adv_20 <= 0:
            adv_20 = data_bus.query_by_pit(sym, current_date, "adv")
        if adv_20 is None or np.isnan(adv_20) or adv_20 <= 0:
            adv_20 = 2e7  # 安全兜底

        liq_cap = (adv_20 * 0.10) / nav
        upper = float(np.clip(liq_cap, 0.01, 0.10))
        # logger.debug("[RISK] %s limit: %.4f (ADV_20=%.2e, NAV=%.2e)", sym, upper, adv_20, nav)
        logger.info("[风险] %s 单票最大持仓权重上限: %.4f (ADV_20=%.2e, NAV=%.2e)", sym, upper, adv_20, nav)
        return upper
    except Exception as e:
        # logger.warning("[RISK] Exception for %s: %s, returning 10%%", sym, e)
        logger.warning("[风险] %s 计算单票最大持仓权重上限时发生异常: %s, 返回默认值 10%%", sym, e)
        return 0.10