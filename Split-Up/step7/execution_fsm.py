# -*- coding: utf-8 -*-
"""
step7/execution_fsm.py
有限状态机交易执行内核（Decoupled Phase 7 State Machine Core）
包含：状态4撮合执行、状态5权益分红占位、状态6账目核对与对账清算
"""
import logging
import numpy as np

logger = logging.getLogger("FSMBacktest.ExecutionFSM")

def process_state_4_execution(engine, target_weights, prices):
    """
    State 4: 撮合执行
    包含：整手截断、平方根冲击滑点、停牌特殊计提减值、退市刚性残值清算
    """
    total_nav = engine.calc_nav()
    target_values = {sym: total_nav * target_weights.get(sym, 0.0) for sym in engine.assets}

    for sym in engine.assets:
        price = prices.get(sym)

        # 1. 退市刚性残值清算
        if engine._check_is_delisted(sym):
            if engine.holdings[sym] != 0:
                residual = engine.config.get('default_residual_rate', 0.0)
                engine.cash += engine.holdings[sym] * residual * (price or 0.0)
                engine.holdings[sym] = 0.0
            continue

        # 2. 停牌超限及特殊计提减值
        if price is None or np.isnan(price) or price <= 0:
            engine.halt_counter[sym] += 1
            engine.halt_status[sym] = True
            if engine.halt_counter[sym] >= engine.config.get('halt_days', 20) and not engine.impairment_applied[sym]:
                rate = engine.config.get('impairment_rate', 0.1)
                engine.impairment_factor[sym] *= (1.0 - rate)
                engine.impairment_applied[sym] = True
                logger.warning(f"⚠️ {sym} 停牌超限（>= {engine.config.get('halt_days', 20)}天），流动性计提减值 {rate*100:.1f}%")
            continue
        else:
            # 恢复交易时，重置停牌计数器并清除减值状态
            engine.halt_counter[sym] = 0
            engine.halt_status[sym] = False
            if engine.impairment_applied[sym]:
                engine.impairment_factor[sym] = 1.0
                engine.impairment_applied[sym] = False
                logger.info(f"🔄 {sym} 恢复交易，主轨自动清除流动性减值因子。")

        # 3. 股本整手与目标股数解算
        current_shares = engine.holdings[sym]
        target_shares = target_values[sym] / price if price > 0 else 0.0
        diff = target_shares - current_shares
        
        # 判定科创板（200股起）与主板（100股起）整手单位限制
        lot_unit = engine.config.get('star_market_lot', 200) if sym.startswith("688") else engine.config.get('main_board_lot', 100)
        
        # 微量仓位清理门槛判定（碎股微调过滤）
        if abs(diff) < engine.config.get('clearing_threshold', 0.001) * total_nav / price:
            if current_shares > 0:
                engine._sell_asset_action(sym, current_shares, price)
            continue

        # 4. 买入调仓撮合解算
        if diff > 0:
            exec_shares = np.floor(diff / lot_unit) * lot_unit
            if exec_shares >= lot_unit:
                adv = engine.bus.query_by_pit(sym, engine.current_date, "adv")
                adv = adv if (adv is not None and adv > 0) else 1e7
                kappa = engine.config.get('static_kappa_impact', 0.001)
                alpha = engine.config.get('static_alpha_impact', 0.5)
                # 平方根市场冲击定律
                impact_factor = kappa * ((exec_shares * price / adv) ** alpha)
                exec_price = price * (1.0 + impact_factor) + engine.config.get('auction_slippage_bps', 0.0002) * price
                
                # 扣除手续费 + 经手费/管理费
                fee = (engine.config.get('handling_fee', 0.0000487) + engine.config.get('management_fee', 0.00002)) * exec_shares * exec_price
                engine.cash -= (exec_shares * exec_price + fee)
                engine.holdings[sym] += exec_shares
                
        # 5. 卖出调仓撮合解算
        elif diff < 0:
            sell_shares = min(abs(diff), current_shares)
            if current_shares - sell_shares < lot_unit and current_shares - sell_shares > 0:
                sell_shares = current_shares
            else:
                sell_shares = np.floor(sell_shares / lot_unit) * lot_unit
            if sell_shares > 0:
                engine._sell_asset_action(sym, sell_shares, price)

def process_state_5_equity(engine):
    """
    State 5: 权益/分红送配处理
    生产级框架标准占位接口，用于后续扩展企业红利公告的除权因果追溯
    """
    pass

def process_state_6_reconciliation(engine, prices, prev_nav):
    """
    State 6: 账目核对与风控清算检测
    双端现金+持仓市值强对账，确保账目不平当场拦截熔断
    """
    nav_calc = engine.calc_nav()
    total_mv = 0.0
    for sym in engine.assets:
        price = prices.get(sym)
        if price is None or np.isnan(price):
            continue
        adj_price = price * engine.impairment_factor.get(sym, 1.0)
        total_mv += engine.holdings[sym] * adj_price
        
    # 双端强实对账强校验
    diff = abs((engine.cash + total_mv) - nav_calc)
    if diff > 0.01:
        raise AssertionError(f"❌ 状态机 [State 6] 动态时序对账不平！差值高达 {diff:.4f} 元，触发系统级核算熔断。")