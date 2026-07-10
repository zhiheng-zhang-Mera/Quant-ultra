# -*- coding: utf-8 -*-
"""
step7/execution_fsm.py
有限状态机硬化流水线：单向纯多头撮合（含资金绝对截断与追高防御）、特殊停牌处理、总账配平验证器
"""
import logging
import numpy as np
from .market_utils import estimate_adaptive_kappa, get_previous_close_price

logger = logging.getLogger("FSMBacktest.FSMProcessor")

def process_state_4_execution(engine, target_weights, prices):
    """
    有限状态机 State 4: 执行撮合。
    工程硬化：100%纯现货多头锁死、跳空高开拒绝买入协议挂载、资金绝对截断防透支哨兵。
    """
    total_nav = engine.calc_nav()
    # 刚性锁死：由于100%采用单边现货，所有融券负数偏置在此处强制向 0.0 截断洗刷
    target_values = {sym: total_nav * max(0.0, target_weights.get(sym, 0.0)) for sym in engine.assets}
    
    for sym in engine.assets:
        price_open = prices.get(sym)
        
        # 1. 物理退市资产剔除合规检测 (Flow-Pro 1.2)
        if engine._check_is_delisted(sym):
            if engine.holdings[sym] != 0:
                residual_rate = engine.config.get('default_residual_rate', 0.0)
                residual_value = engine.holdings[sym] * residual_rate * (price_open if price_open else 0.0)
                engine.cash += residual_value
                logger.warning(f"[{engine.current_date}] 🚨 资产 {sym} 确认物理退市! 强制清算持仓，回收默认残值: {residual_value:.2f}")
                engine.holdings[sym] = 0.0
            continue
            
        # 2. 特殊停牌监控机制检测
        if price_open is None or np.isnan(price_open) or price_open <= 0:
            engine.halt_counter[sym] += 1
            engine.halt_status[sym] = True
            if engine.halt_counter[sym] >= engine.config.get('halt_days', 20) and not engine.impairment_applied[sym]:
                rate = engine.config.get('impairment_rate', 0.1)
                engine.impairment_factor[sym] *= (1.0 - rate)
                engine.impairment_applied[sym] = True
                logger.warning(f"[{engine.current_date}] 📉 标的 {sym} 连续停牌达 {engine.halt_counter[sym]} 日，触发账面资产减值 {rate*100:.1f}%")
            continue
        else:
            engine.halt_counter[sym] = 0
            engine.halt_status[sym] = False
            if engine.impairment_applied[sym]:
                engine.impairment_factor[sym] = 1.0
                engine.impairment_applied[sym] = False
                logger.info(f"[{engine.current_date}] 🔄 标的 {sym} 恢复市价交易，账面资产减值因子全额充刷重置为 1.0")
                
        # 3. 产生期望仓位差值并进行单向多头分配
        current_shares = engine.holdings[sym]
        target_shares = target_values[sym] / price_open if price_open > 0 else 0.0
        diff = target_shares - current_shares
        
        lot_unit = engine.config.get('star_market_lot', 200) if sym.startswith("688") else engine.config.get('main_board_lot', 100)
        
        # 小微交易过滤防御线
        if abs(diff) < engine.config.get('clearing_threshold', 0.001) * total_nav / price_open:
            if current_shares != 0 and target_weights.get(sym, 0.0) == 0.0:
                engine._sell_asset_action(sym, current_shares, price_open)
            continue
            
        # 4. 执行单边多头买入开仓 / 增仓
        if diff > 0:
            # 🚨 [Flow-Pro 7.2] 跳空高开拒绝买入协议（追高防御）
            price_close_prev = get_previous_close_price(sym, engine.current_date, engine.bus, engine.price_cache)
            if price_close_prev is not None and price_close_prev > 0:
                gap_up_ratio = (price_open / price_close_prev) - 1.0
                if gap_up_ratio > engine.config.get('gap_up_threshold', 0.03):
                    logger.warning(f"[{engine.current_date}] 🛡️ 触发追高防御协议！{sym} 今日开盘跳空高开 {gap_up_ratio*100:.2f}% > 3%，强行挂起并拒绝开仓买入！")
                    continue

            exec_shares = np.floor(diff / lot_unit) * lot_unit
            if exec_shares >= lot_unit:
                adv = engine.bus.query_by_pit(sym, engine.current_date, "adv")
                adv = adv if (adv is not None and adv > 0) else 1e7
                
                kappa = estimate_adaptive_kappa(sym, engine.current_date, engine.bus, engine.config, engine.impact_kappa_base)
                alpha = engine.config.get('static_alpha_impact', 0.5)
                
                # 冲击与滑点叠加成本计算
                impact_factor = kappa * ((exec_shares * price_open / adv) ** alpha)
                exec_price = price_open * (1.0 + impact_factor) + engine.config.get('auction_slippage_bps', 0.0002) * price_open
                
                fee_rate = engine.config.get('handling_fee', 0.0000487) + engine.config.get('management_fee', 0.00002)
                total_cost = exec_shares * exec_price * (1.0 + fee_rate)
                
                # 🚨 [Flow-Pro 7.3] 资金绝对截断哨兵（Cash Absolute Truncation Sentinel）
                if total_cost > engine.cash:
                    logger.info(f"[{engine.current_date}] ⚠️ 触发资金绝对截断哨兵！可用现金不足以覆盖非线性买入开销。自动执行同比例降维扣减。")
                    max_available_shares = engine.cash / (exec_price * (1.0 + fee_rate))
                    exec_shares = np.floor(max_available_shares / lot_unit) * lot_unit
                    if exec_shares < lot_unit:
                        logger.info(f"[{engine.current_date}] ❌ 扣减后股数低于单手限制，就地对该标的放弃买入操作。")
                        continue
                    # 重新进行精确二次非线性开销标定
                    impact_factor = kappa * ((exec_shares * price_open / adv) ** alpha)
                    exec_price = price_open * (1.0 + impact_factor) + engine.config.get('auction_slippage_bps', 0.0002) * price_open
                    total_cost = exec_shares * exec_price * (1.0 + fee_rate)
                    
                engine.cash -= total_cost
                engine.holdings[sym] += exec_shares
                
        # 5. 执行现货单边卖出减仓 / 平仓
        elif diff < 0:
            sell_shares = min(abs(diff), current_shares)
            if current_shares - sell_shares < lot_unit and current_shares - sell_shares > 0:
                sell_shares = current_shares
            else:
                sell_shares = np.floor(sell_shares / lot_unit) * lot_unit
            if sell_shares > 0:
                engine._sell_asset_action(sym, sell_shares, price_open)

def process_state_5_equity(engine):
    """
    有限状态机 State 5: 除权除息、红利派发自动过户登记中心（个体全多头权益清算存根）
    """
    pass

def process_state_6_reconciliation(engine, prices, prev_nav):
    """
    有限状态机 State 6: 纯现金清算与安全哨兵隐式总账对账（完全吞噬并替代原有信用资产层）。
    严格执行单张资产负债表守恒定律：现金 + 减值修正后总市值 == 系统级动态测算 NAV。
    """
    nav_calc = engine.calc_nav()
    total_mv = 0.0
    for sym in engine.assets:
        price = prices.get(sym)
        if price is None or np.isnan(price):
            continue
        adj_price = price * engine.impairment_factor.get(sym, 1.0)
        total_mv += engine.holdings[sym] * adj_price
        
    # 精算级两端严格对账配平 (Flow-Pro 7.4)
    diff = abs((engine.cash + total_mv) - nav_calc)
    if diff > 0.01:
        err_msg = f"[{engine.current_date}] ❌ 总账会计审计异常! 资产负债表无法物理强行配平! 差值: {diff:.6f}, 现金账户: {engine.cash:.2f}, 计算市值: {total_mv:.2f}, 登记净值: {nav_calc:.2f}"
        logger.critical(err_msg)
        raise AssertionError(err_msg)
        
    if prev_nav is not None:
        delta_nav = nav_calc - prev_nav
        logger.debug(f"[{engine.current_date}] ⚖️ 隐式前向清算完成。当期可用现金池: {engine.cash:.2f} 元, 时序增量变动守恒。")