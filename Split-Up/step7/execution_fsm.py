# -*- coding: utf-8 -*-
"""
step7/execution_fsm.py
有限状态机流水线：撮合撮合、特殊停牌计提减值、退市残值归结算处理器
"""
import logging
import numpy as np
from .market_utils import estimate_adaptive_kappa
from .risk_guard import audit_credit_accrual

logger = logging.getLogger("FSMBacktest.FSMProcessor")

def process_state_4_execution(engine, target_weights, prices):
    """
    有限状态机 State 4: 撮合指令执行。支持整手截断、自适应冲击、持续停牌资产减值与退市强制归零。
    """
    total_nav = engine.calc_nav()
    target_values = {sym: total_nav * target_weights.get(sym, 0.0) for sym in engine.assets}
    
    for sym in engine.assets:
        price = prices.get(sym)
        
        # 1. 物理退市资产剔除合规检测
        if engine._check_is_delisted(sym):
            if engine.holdings[sym] != 0:
                residual_rate = engine.config.get('default_residual_rate', 0.0)
                residual_value = engine.holdings[sym] * residual_rate * (price if price else 0.0)
                engine.cash += residual_value
                logger.warning(f"[{engine.current_date}] 🚨 资产 {sym} 确认物理退市! 强制清算持仓，回收残值: {residual_value:.2f}")
                engine.holdings[sym] = 0.0
            continue
            
        # 2. 特殊停牌监控机制检测
        if price is None or np.isnan(price) or price <= 0:
            engine.halt_counter[sym] += 1
            engine.halt_status[sym] = True
            if engine.halt_counter[sym] >= engine.config.get('halt_days', 20) and not engine.impairment_applied[sym]:
                # 连续停牌超限，物理执行资产负债表下的减值标记
                rate = engine.config.get('impairment_rate', 0.1)
                engine.impairment_factor[sym] *= (1.0 - rate)
                engine.impairment_applied[sym] = True
                logger.warning(f"[{engine.current_date}] 📉 标的 {sym} 连续停牌达 {engine.halt_counter[sym]} 日，触发账面资产减值 {rate*100:.1f}%, 减值乘数修正为: {engine.impairment_factor[sym]:.3f}")
            continue
        else:
            # 恢复正常交易状态，平滑重置计数防线
            engine.halt_counter[sym] = 0
            engine.halt_status[sym] = False
            if engine.impairment_applied[sym]:
                engine.impairment_factor[sym] = 1.0
                engine.impairment_applied[sym] = False
                logger.info(f"[{engine.current_date}] 🔄 标的 {sym} 恢复市价交易，账面资产减值因子全额充刷重置为 1.0")
                
        # 3. 产生期望仓位差值并进行多空匹配
        current_shares = engine.holdings[sym]
        target_shares = target_values[sym] / price if price > 0 else 0.0
        diff = target_shares - current_shares
        
        # 确定特定市场的个股整手单元限制
        lot_unit = 200 if sym.startswith("688") else 100
        
        # 小微交易过滤防御线，防止碎片订单摩擦无意义规费
        if abs(diff) < engine.config.get('clearing_threshold', 0.001) * total_nav / price:
            if current_shares != 0 and target_weights.get(sym, 0.0) == 0.0:
                engine._sell_asset_action(sym, current_shares, price)
            continue
            
        # 4. 执行多头买入 / 空头平仓
        kappa = estimate_adaptive_kappa(sym, engine.current_date, engine.bus, engine.config, engine.impact_kappa_base)
        if diff > 0:
            exec_shares = np.floor(diff / lot_unit) * lot_unit
            if exec_shares >= lot_unit:
                adv = engine.bus.query_by_pit(sym, engine.current_date, "adv")
                adv = adv if (adv is not None and adv > 0) else 1e7
                
                # 冲击与滑点叠加成本公式: Price * (1 + kappa * (Exec / ADV)^alpha) + slippage
                impact_factor = kappa * ((exec_shares * price / adv) ** engine.config.get('impact_alpha', 0.5))
                exec_price = price * (1.0 + impact_factor) + engine.config.get('auction_slippage_bps', 0.0002) * price
                
                fees = (engine.config.get('handling_fee', 0.0000487) + engine.config.get('management_fee', 0.00002)) * exec_shares * exec_price
                engine.cash -= (exec_shares * exec_price + fees)
                engine.holdings[sym] += exec_shares
        # 5. 执行卖出减仓
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
    有限状态机 State 5: 除权除息、红利派发等分红权益自动登记过户中心 (暂列存根)
    """
    pass

def process_state_6_credit(engine, target_weights, prices):
    """
    有限状态机 State 6: 利息计提与两融合规审查调度桥接
    """
    audit_credit_accrual(engine, target_weights, prices)

def process_state_7_reconciliation(engine, prices):
    """
    有限状态机 State 7: 每日清算后对账审计。
    严格执行单张资产负债表守恒定律: 现金 + 减值修正后总市值 == 系统级动态 NAV。
    若对账差额超出精算精度限制，实施物理报错熔断。
    """
    nav_calc = engine.calc_nav()
    total_mv = 0.0
    for sym in engine.assets:
        price = prices.get(sym)
        if price is None or np.isnan(price):
            continue
        adj_price = price * engine.impairment_factor.get(sym, 1.0)
        total_mv += engine.holdings[sym] * adj_price
        
    diff = abs((engine.cash + total_mv) - nav_calc)
    if diff > 0.01:
        err_msg = f"[{engine.current_date}] ❌ 总账会计审计异常! 资产负债表无法强行配平! 差值: {diff:.6f}, 现金账户: {engine.cash:.2f}, 计算市值: {total_mv:.2f}, 登记净值: {nav_calc:.2f}"
        logger.critical(err_msg)
        raise AssertionError(err_msg)