# -*- coding: utf-8 -*-
"""
step7/risk_guard.py
个人流动性与资金量双重硬约束防线守门狗（全面物理粉碎机构两融资管条款）
"""
import logging
import numpy as np

logger = logging.getLogger("FSMBacktest.RiskGuard")

def compute_individual_position_limit(sym, nav, current_date, data_bus, config):
    """
    [Flow-Pro M.3 / 6.3 核心要求重写]
    为契合个体账户的真实资产规模与标的微观流动性，刚性废除机构级大股东 4.5% 举牌限制。
    单票最大持仓权重上限强制重构为本地对开安全垫算子：
    单票权重上限 = min(10%, 10% * ADV_20 / 账户总权益)
    """
    if nav <= 0:
        return config.get('max_single_ticket_prop', 0.10)
        
    try:
        # 从边缘数据总线中点状查询该个股过去 20 个交易日的平均每日成交额 (ADV_20)
        adv_20 = data_bus.query_by_pit(sym, current_date, "adv_20")
        if adv_20 is None or np.isnan(adv_20) or adv_20 <= 0:
            # 降级兜底路径
            adv_20 = data_bus.query_by_pit(sym, current_date, "adv")
            
        if adv_20 is None or np.isnan(adv_20) or adv_20 <= 0:
            return config.get('max_single_ticket_prop', 0.10)
            
        # 实施本地对开安全垫算子公式，锁死本金在微观流动性个股上的暴露
        liquidity_bound = 0.10 * (adv_20 / nav)
        max_weight = min(config.get('max_single_ticket_prop', 0.10), liquidity_bound)
        return float(max(0.0, max_weight))
    except Exception as e:
        logger.debug(f"[{current_date}] 标的 {sym} 个体流动性双重硬约束算子解算异常，强制截断至10%. 原因: {str(e)}")
        return config.get('max_single_ticket_prop', 0.10)

# 💡 提示：原有的两融维持担保审计与信用负债计提函数 audit_credit_accrual 已按照规范完全物理碎尸粉碎，不予留存。