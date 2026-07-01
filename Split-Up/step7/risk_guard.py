# -*- coding: utf-8 -*-
"""
step7/risk_guard.py
大盘股本合规举牌红线刺探器、两融维持担保比例全时段信用审计守门狗
"""
import logging
import akshare as ak
import numpy as np

logger = logging.getLogger("FSMBacktest.RiskGuard")

# 全局高隔离度的局部单例内存字典，彻底物理封杀每日重复网络请求的开销
_GLOBAL_SHARES_CACHE = {}

def compute_dynamic_upper_bound(sym, nav, current_date, data_bus):
    """
    根据标的最新总股本和PIT全收益价格，计算单一投资组合允许暴露的最大持仓上限权重
    严密对齐大盘红线约束 (不允许超过总市值的 4.5%)
    """
    if nav <= 0:
        return 0.045
        
    try:
        # 优先读取内存物理缓存，拦截频繁请求导致的异常
        if sym in _GLOBAL_SHARES_CACHE:
            total_shares = _GLOBAL_SHARES_CACHE[sym]
        else:
            info = ak.stock_individual_info_em(symbol=sym)
            total_shares = info[info['item'] == '总股本']['value'].values[0]
            _GLOBAL_SHARES_CACHE[sym] = total_shares
            
        price = data_bus.query_by_pit(sym, current_date, "total_return_price")
        if price is None or np.isnan(price) or price <= 0:
            return 0.045
            
        market_value = total_shares * price
        max_weight = (0.045 * market_value) / nav
        return float(min(max_weight, 0.045))
    except Exception as e:
        logger.debug(f"[{current_date}] 标的 {sym} 动态总股本红线穿透失败，安全保底截断至 4.5%. 原因: {str(e)}")
        return 0.045

def audit_credit_accrual(engine, target_weights, prices):
    """
    有限状态机 State 6: 信用利息计提与动态维持担保比例强平合规审计
    """
    total_debt = 0.0
    nav = engine.calc_nav()
    
    for sym in engine.assets:
        price = prices.get(sym)
        if price is None or price <= 0 or np.isnan(price):
            continue
            
        adj_price = price * engine.impairment_factor.get(sym, 1.0)
        mv = engine.holdings[sym] * adj_price
        w = target_weights.get(sym, 0.0)
        
        # 杠杆多头多余部分的负债计提
        if w > 1.0 and engine.holdings[sym] > 0:
            debt = (w - 1.0) * nav
            total_debt += debt
            engine.cash -= (engine.config.get('margin_interest', 0.06) / 252.0) * debt
        # 融券空头头寸的负债利息计提
        elif engine.holdings[sym] < 0:
            short_value = -engine.holdings[sym] * adj_price
            engine.cash -= (engine.config.get('short_interest', 0.08) / 252.0) * short_value
            
    if total_debt > 0.0:
        maintenance_ratio = nav / total_debt
        if maintenance_ratio < engine.config.get('maintenance_ratio', 1.3):
            logger.warning(f"[{engine.current_date}] ⚠️ 触发风控熔断! 维持担保比例 {maintenance_ratio:.2f} 低于警戒线 {engine.config.get('maintenance_ratio', 1.3)}，执行物理强平!")
            # 暴力降维强平：全部头寸市价清刷
            for sym in engine.assets:
                if engine.holdings[sym] > 0:
                    p = prices.get(sym, 0)
                    if p > 0:
                        engine._sell_asset_action(sym, engine.holdings[sym], p)
                elif engine.holdings[sym] < 0:
                    p = prices.get(sym, 0)
                    if p > 0:
                        buy_shares = -engine.holdings[sym]
                        slippage = engine.config.get('auction_slippage_bps', 0.0002) * p
                        engine.cash -= buy_shares * (p + slippage)
                        engine.holdings[sym] = 0.0