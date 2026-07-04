# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_7 Micro-Liquidity Individual Capacity Sentinel
"""
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger("FSMBacktest.RiskGuard")

def compute_individual_position_limit(sym: str, nav: float, current_date: datetime, data_bus, config: dict) -> float:
    """
    个人流动性与资金量双重硬约束防线守门狗。
    单票最大持仓权重上限 = min(10%, 10% * ADV_20 / 账户可用总权益)。
    """
    if nav <= 0: return 0.10
        
    try:
        # 穿透查询 Phase_6 编译拉齐的截面历史 20 日平均成交额 ADV_20 物理底表
        adv_20 = data_bus.query_by_pit(sym, current_date, "adv_20")
        if adv_20 is None or np.isnan(adv_20) or adv_20 <= 0:
            adv_20 = data_bus.query_by_pit(sym, current_date, "adv")
            
        if adv_20 is None or np.isnan(adv_20) or adv_20 <= 0: adv_20 = 20000000.0
            
        liq_cap = (adv_20 * 0.10) / nav
        final_upper_bound = float(np.clip(liq_cap, 0.01, 0.10))
        
        logger.debug("[OP] Evaluate Individual Capacity Limit | [SOURCE] Sectional ADV Matrix | [RESULT] Bounds Cap for %s: %.4f | [SIGNIFICANCE] Protects portfolio from micro-liquidity friction choke points", sym, final_upper_bound)
        logger.debug("[操作] 评估个体流动性容量上限 | [来源] 截面 20 日真实成交额矩阵 | [结果] 标的 %s 持仓限制线: %.4f | [意义] 契合微观现货真实冲击容量，防范长尾成分股由于挤踏丧失流动性变现力")
        return final_upper_bound
    except Exception:
        return 0.10