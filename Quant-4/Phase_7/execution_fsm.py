# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 7.4-7.6: FSM Order Matching & Bi-Party Reconciliation Engine
"""
import logging
import numpy as np
from Phase_7.config import STATIC_KAPPA_IMPACT, STATIC_ALPHA_IMPACT

logger = logging.getLogger("FSMBacktest.ExecutionFSM")

def process_state_4_execution(engine, target_weights: dict, prices: dict):
    """
    State 4: 撮合执行
    包含：100股整手截断、平方根冲击滑点、停牌超限减值惩罚、退市残值刚性清算。
    """
    total_nav = engine.calc_nav()
    target_values = {sym: total_nav * target_weights.get(sym, 0.0) for sym in engine.assets}

    # 1. 优先解算卖出调仓，腾出资金安全垫
    for sym in engine.assets:
        price = prices.get(sym)
        if engine._check_is_delisted(sym):
            if engine.holdings[sym] != 0:
                engine.cash += engine.holdings[sym] * engine.config.get('default_residual_rate', 0.0) * (price or 0.0)
                engine.holdings[sym] = 0.0
            continue

        if price is None or np.isnan(price) or price <= 0:
            engine.halt_counter[sym] += 1
            if engine.halt_counter[sym] >= engine.config.get('halt_days_limit', 20):
                engine.halt_status[sym] = True
            continue
        else: engine.halt_counter[sym] = 0

        if engine.halt_status[sym] or engine.holdings[sym] == 0: continue
            
        current_val = engine.holdings[sym] * price
        t_val = target_values.get(sym, 0.0)
        diff_val = t_val - current_val
        
        if diff_val < 0: # 触发正向卖出减仓
            sell_shares = min(abs(diff_val) / price, engine.holdings[sym])
            sell_shares = np.floor(sell_shares / 100.0) * 100.0 if engine.holdings[sym] - sell_shares >= 100.0 else engine.holdings[sym]
            if sell_shares > 0: engine._sell_asset_action(sym, sell_shares, price)

    # 2. 顺次解算买入调仓，实施平方根冲击滑点扣减 (Flow-Pro 7.2)
    for sym in engine.assets:
        price = prices.get(sym)
        if price is None or np.isnan(price) or price <= 0 or engine.halt_status[sym]: continue
            
        current_val = engine.holdings[sym] * price
        t_val = target_values.get(sym, 0.0)
        diff_val = t_val - current_val
        
        if diff_val > 0 and engine.cash > 1000:
            buy_shares = diff_val / price
            buy_shares = np.floor(buy_shares / 100.0) * 100.0
            if buy_shares <= 0: continue
                
            # 引入平方根市场冲击模型计算内生交易滑点惩罚权重
            turnover_ratio = (buy_shares * price) / total_nav
            slippage_impact = STATIC_KAPPA_IMPACT * (turnover_ratio ** STATIC_ALPHA_IMPACT)
            exec_price = price * (1.0 + max(engine.config.get('slippage_bps', 0.0002), slippage_impact))
            
            cost = buy_shares * exec_price
            fee = (engine.config.get('handling_fee', 0.0000487) + engine.config.get('management_fee', 0.00002)) * cost
            if engine.cash >= cost + fee:
                engine.cash -= (cost + fee)
                engine.holdings[sym] += buy_shares
                
    logger.info("[OP] Run Order Matching Loop | [SOURCE] Sectional Execution Queue | [RESULT] Matching pass completed | [SIGNIFICANCE] Enforces 100-share constraints and non-linear slippage friction pricing")
    logger.info("[操作] 执行订单撮合大循环 | [来源] 截面调仓执行队列 | [结果] 撮合步进计算完成 | [意义] 强行实施 100 股整手截断限制与非线性市场冲击滑点惩罚，逼真实盘摩擦损耗")

def process_state_5_equity(engine):
    """State 5: 权益/分红送配处理 (生产级标准抽象占位接口)"""
    pass

def process_state_6_reconciliation(engine, prices: dict, prev_nav: float):
    """
    State 6: 账目核对与风控清算检测
    双端现金+持仓市值总账强核对，若对账不平触发致命红线，立即执行一键物理熔断。
    """
    nav_calc = engine.calc_nav()
    total_mv = sum(engine.holdings[s] * (prices.get(s) or 0.0) for s in engine.assets if not engine.halt_status[s])
    total_halt_mv = sum(engine.holdings[s] * (prices.get(s) or 0.0) * (1.0 - engine.config.get('impairment_rate', 0.1)) for s in engine.assets if engine.halt_status[s])
    
    reconciled_balance = engine.cash + total_mv + total_halt_mv
    discrepancy = abs(nav_calc - reconciled_balance)
    
    if discrepancy > 1.0:
        logger.critical("[OP] Reconcile Bi-Party Ledger | [SOURCE] Double-Entry Clearing Vault | [RESULT] Mismatch Catastrophe! Diff: %.4f CNY | [SIGNIFICANCE] Discrepancy violates accounting law, triggering hard panic shutdown", discrepancy)
        logger.critical("[操作] 强行执行双端总账对账 | [来源] 级联清算小金库余额明细 | [结果] 错账账目不平灾难！绝对误差金额: %.4f 元 | [意义] 违反底线会计复式借贷平衡守恒守恒红线，强行切断策略流抛出异常熔断")
        raise RuntimeError(f"FSM Bookkeeping Catastrophe: Ledger unbalanced by {discrepancy} CNY.")
        
    logger.info("[OP] Complete EOD Balance Verification | [SOURCE] Compliant Audited Ledger | [RESULT] NAV: %.2f, Discrepancy: %.4f | [SIGNIFICANCE] Confirms capital integrity before advancing chronology axis", nav_calc, discrepancy)
    logger.info("[操作] 完成盘后资金安全核销对账 | [来源] 经验证的合规交易账本 | [结果] 最终清算总权益NAV: %.2f, 勾稽差额: %.4f | [意义] 证明无常态资金漏损，清盘安全交割放行，允许时间轴推进行走")