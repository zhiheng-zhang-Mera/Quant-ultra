# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_7 Point-In-Time Price Alignment & Piercing Subsystem
"""
import logging
import numpy as np
from datetime import timedelta

logger = logging.getLogger("FSMBacktest.MarketUtils")

def get_prices_for_date(sym_list, date, data_bus, price_cache, lookback_days=10):
    """
    点时对齐价格追溯总线。停牌时向后穿透，避免返回 None。
    """
    prices = {}
    for sym in sym_list:
        cache_key = (sym, date, "total_return_price")
        if cache_key in price_cache:
            prices[sym] = price_cache[cache_key]
            continue

        p = data_bus.query_by_pit(sym, date, "total_return_price")
        if p is None or np.isnan(p) or p <= 0:
            found = False
            for delta in range(1, lookback_days + 1):
                alt_date = date - timedelta(days=delta)
                p = data_bus.query_by_pit(sym, alt_date, "total_return_price")
                if p is not None and not np.isnan(p) and p > 0:
                    found = True
                    break
            if not found:
                p = None
        price_cache[cache_key] = p
        prices[sym] = p

    logger.debug("[PIT] Retrieved prices for %d symbols on %s", len(prices), date.date())
    return prices


def get_previous_close_price(sym, date, data_bus, price_cache, lookback_days=10):
    """提取前一有效交易日收盘价，用于隔夜跳空检测"""
    for delta in range(1, lookback_days + 1):
        alt_date = date - timedelta(days=delta)
        cache_key = (sym, alt_date, "close")
        if cache_key in price_cache and price_cache[cache_key] is not None:
            return price_cache[cache_key]

        p = data_bus.query_by_pit(sym, alt_date, "close")
        if p is not None and not np.isnan(p) and p > 0:
            price_cache[cache_key] = p
            return p
    return None