# -*- coding: utf-8 -*-
"""
step7/market_utils.py
点时对齐价格追溯总线、隔夜收盘价穿透器与硬编码静态冲击系数评估内核
"""
import logging
import numpy as np
from datetime import timedelta

logger = logging.getLogger("FSMBacktest.MarketUtils")

def get_prices_for_date(sym_list, date, data_bus, price_cache, lookback_days=10):
    """
    点时对齐价格追溯总线。当T日由于停牌无法拉取价格时，向后平滑穿透lookback_days个物理日
    """
    prices = {}
    for sym in sym_list:
        cache_key = (sym, date, "total_return_price")
        if cache_key in price_cache:
            prices[sym] = price_cache[cache_key]
            continue
            
        p = data_bus.query_by_pit(sym, date, "total_return_price")
        if p is None or np.isnan(p):
            found = False
            for delta in range(1, lookback_days + 1):
                alt_date = date - timedelta(days=delta)
                p = data_bus.query_by_pit(sym, alt_date, "total_return_price")
                if p is not None and not np.isnan(p):
                    found = True
                    break
            if not found:
                p = None
                
        price_cache[cache_key] = p
        prices[sym] = p
    return prices

def get_previous_close_price(sym, date, data_bus, price_cache, lookback_days=10):
    """
    [Flow-Pro 7.2 专属配套] 获取当前交易日之前的最新有效收盘价，用于隔夜跳空追高防御过滤器
    """
    for delta in range(1, lookback_days + 1):
        alt_date = date - timedelta(days=delta)
        cache_key = (sym, alt_date, "close")
        if cache_key in price_cache:
            if price_cache[cache_key] is not None:
                return price_cache[cache_key]
        p = data_bus.query_by_pit(sym, alt_date, "close")
        if p is not None and not np.isnan(p) and p > 0:
            price_cache[cache_key] = p
            return p
    return None

def estimate_adaptive_kappa(sym, current_date, data_bus, config, base_kappa):
    """
    [Flow-Pro 7.2 刚性重构降级]
    彻底废除基于历史高低买卖价差的动态自适应估计机制，全面斩断网络刺探。
    硬性遵循平方根市场冲击定律，刚性返回全局静态硬编码冲击常数。
    """
    return config.get('static_kappa_impact', 0.001)