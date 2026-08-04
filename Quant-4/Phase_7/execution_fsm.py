# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 7.4-7.6: FSM Order Matching & Bi-Party Reconciliation Engine
"""
import logging
import numpy as np
from Phase_7.config import STATIC_KAPPA_IMPACT, STATIC_ALPHA_IMPACT, DEFAULT_RESIDUAL_RATE
from Main.trading_costs import explicit_order_fees, merged_cost_config

logger = logging.getLogger("FSMBacktest.ExecutionFSM")

def process_state_4_execution(engine, target_weights: dict, prices: dict):
    """
    State 4: 撮合执行
    包含：先卖后买、整手截断、平方根冲击滑点、停牌超限减值惩罚、退市残值刚性清算。
    """
    total_nav = engine.calc_nav()
    target_values = {sym: total_nav * target_weights.get(sym, 0.0) for sym in engine.assets}

    # ---- 1. 退市资产强制清算 ----
    for sym in engine.assets:
        if engine._check_is_delisted(sym):
            if engine.holdings[sym] != 0:
                price = prices.get(sym) or 0.0
                residual = engine.config.get('default_residual_rate', DEFAULT_RESIDUAL_RATE)
                engine.cash += engine.holdings[sym] * residual * price
                # logger.info("[DELIST] %s liquidated at residual rate %.2f", sym, residual)
                logger.info("由FSM回测引擎强制清算退市资产 | 当前资产: %s | 清算残值率: %.2f", sym, residual)
                engine.holdings[sym] = 0.0
            continue

    # ---- 2. 停牌状态更新 & 减值标记 ----
    for sym in engine.assets:
        price = prices.get(sym)
        if price is None or np.isnan(price) or price <= 0:
            engine.halt_counter[sym] += 1
            if engine.halt_counter[sym] >= engine.config.get('halt_days_limit', 20):
                engine.halt_status[sym] = True
                if not engine.impairment_applied.get(sym, False):
                    # 一次性计提减值
                    engine.impairment_factor[sym] = 1.0 - engine.config.get('impairment_rate', 0.1)
                    engine.impairment_applied[sym] = True
                    # logger.warning("[HALT] %s halted >= %d days, impairment factor applied: %.2f",
                    #               sym, engine.config.get('halt_days_limit', 20),
                    #               engine.impairment_factor[sym])
                    logger.warning("由FSM回测引擎标记停牌资产并计提减值 | 当前资产: %s | 停牌天数: %d | 减值率: %.2f",
                                   sym, engine.config.get('halt_days_limit', 20), engine.impairment_factor[sym])
        else:
            engine.halt_counter[sym] = 0
            if engine.halt_status.get(sym, False):
                engine.halt_status[sym] = False
                engine.impairment_factor[sym] = 1.0
                engine.impairment_applied[sym] = False
                #logger.info("[RESUME] %s resumed trading, impairment cleared", sym)
                logger.info("由FSM回测引擎标记停牌资产恢复交易并清除减值 | 当前资产: %s", sym)
    # ---- 3. 优先处理卖出（释放现金） ----
    for sym in engine.assets:
        price = prices.get(sym)
        if price is None or np.isnan(price) or price <= 0 or engine.halt_status.get(sym, False):
            continue
        current_shares = engine.holdings[sym]
        if current_shares == 0:
            continue
        current_val = current_shares * price
        t_val = target_values.get(sym, 0.0)
        diff_val = t_val - current_val
        if diff_val < 0:  # 需要卖出
            # 整手截断
            lot = engine._get_lot_size(sym)
            sell_shares = min(abs(diff_val) / price, current_shares)
            # 保留至少一手，如果剩余不足一手则全清
            if current_shares - sell_shares < lot and current_shares - sell_shares > 0:
                sell_shares = current_shares
            else:
                sell_shares = np.floor(sell_shares / lot) * lot
            if sell_shares > 0:
                engine._sell_asset_action(sym, sell_shares, price)

    # ---- 4. 再处理买入 ----
    for sym in engine.assets:
        price = prices.get(sym)
        if price is None or np.isnan(price) or price <= 0 or engine.halt_status.get(sym, False):
            continue
        current_shares = engine.holdings[sym]
        current_val = current_shares * price
        t_val = target_values.get(sym, 0.0)
        diff_val = t_val - current_val
        if diff_val > 0 and engine.cash > 1000:
            lot = engine._get_lot_size(sym)
            buy_shares = np.floor(diff_val / price / lot) * lot
            if buy_shares <= 0:
                continue
            # 平方根市场冲击
            adv = engine.bus.query_by_pit(sym, engine.current_date, "adv")
            adv = adv if (adv is not None and adv > 0) else 1e7
            turnover = (buy_shares * price) / adv
            slippage_impact = STATIC_KAPPA_IMPACT * (turnover ** STATIC_ALPHA_IMPACT)
            base_slippage = float(merged_cost_config(engine.config)['slippage_rate'])
            exec_price = price * (1.0 + slippage_impact + base_slippage)
            cost = buy_shares * exec_price
            fees = explicit_order_fees(cost, "buy", symbol=sym, config=engine.config)
            if engine.cash >= cost + fees['total']:
                engine.cash -= cost + fees['total']
                engine.holdings[sym] += buy_shares
                for key in ("commission", "exchange_fee", "regulatory_fee", "stamp_tax"):
                    engine.cost_ledger[key] += fees[key]
                engine.cost_ledger["slippage"] += buy_shares * price * (slippage_impact + base_slippage)
                logger.debug("[BUY] %s %d shares @ %.4f (impact %.4f)", sym, buy_shares, exec_price, slippage_impact)

    # logger.info("[EXEC] Order matching completed for %d assets", len(engine.assets))
    logger.info("由FSM回测引擎Phase 4完成撮合执行 | 涉及资产数量: %d", len(engine.assets))

def process_state_5_equity(engine):
    """State 5: 权益/分红送配处理 (生产级标准抽象占位接口)"""
    logger.info("由FSM回测引擎Phase 5处理权益分红送配")
    pass


def process_state_6_reconciliation(engine, prices: dict, prev_nav: float):
    """
    State 6: 账目核对与风控清算检测
    双端现金+持仓市值总账强核对，若对账不平触发致命红线。
    """
    nav_calc = engine.calc_nav()
    total_mv = 0.0
    for sym in engine.assets:
        p = prices.get(sym) or 0.0
        if engine.halt_status.get(sym, False):
            # 停牌资产按减值后估值
            p *= engine.impairment_factor.get(sym, 1.0)
        total_mv += engine.holdings[sym] * p

    reconciled_balance = engine.cash + total_mv
    discrepancy = abs(nav_calc - reconciled_balance)

    if discrepancy > 1.0:
        # logger.critical("[RECON] Ledger mismatch! Diff=%.4f, NAV=%.2f, Reconciled=%.2f",
        #                discrepancy, nav_calc, reconciled_balance)
        logger.critical("由FSM回测引擎Phase 6触发致命红线 | 账本不平衡 | 差额: %.4f | 计算净值: %.2f | 对账总额: %.2f", discrepancy, nav_calc, reconciled_balance)
        # raise RuntimeError(f"FSM Bookkeeping Catastrophe: Ledger unbalanced by {discrepancy} CNY.")
        raise RuntimeError(f"FSM回测引擎Phase 6账本灾难: 账本不平衡，差额为 {discrepancy} CNY。")
    else:
        # logger.debug("[RECON] Verified: NAV=%.2f, Diff=%.4f", nav_calc, discrepancy)
        logger.debug("由FSM回测引擎Phase 6核对账目 | 账本平衡 | 计算净值: %.2f | 差额: %.4f", nav_calc, discrepancy)
