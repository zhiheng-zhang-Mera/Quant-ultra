# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_7 Point-In-Time Price Alignment & Piercing Subsystem
"""
import logging
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger("FSMBacktest.MarketUtils")

def get_prices_for_date(sym_list: list, date: datetime, data_bus, price_cache: dict, lookback_days: int = 10) -> dict:
    """
    点时对齐价格追溯总线。
    当T日标的由于停牌无法拉取全收益收盘价时，向后安全平滑穿透，杜绝前瞻透视。
    """
    prices = {}
    for sym in sym_list:
        cache_key = (sym, date, "total_return_price")
        if cache_key in price_cache:
            prices[sym] = price_cache[cache_key]; continue
            
        # 穿透 Point-In-Time 总线获取指定时点的历史状态，锁死 15:00 盘后边界
        p = data_bus.query_by_pit(sym, date, "total_return_price")
        if p is None or np.isnan(p) or p <= 0:
            found = False
            for delta in range(1, lookback_days + 1):
                alt_date = date - timedelta(days=delta)
                p = data_bus.query_by_pit(sym, alt_date, "total_return_price")
                if p is not None and not np.isnan(p) and p > 0:
                    found = True; break
            if not found: p = None
                
        price_cache[cache_key] = p
        prices[sym] = p
        
    logger.debug("[OP] Query PIT Price Vectors | [SOURCE] Immutable TSDB Data Bus | [RESULT] Extracted assets count: %s | [SIGNIFICANCE] Guarantees aligned matching references without lookahead leaks", len(prices))
    logger.debug("[操作] 查询时点行情价格向量 | [来源] 不可变时序数据总线 | [结果] 提取有效资产数: %s | [意义] 确保撮合价格在时间轴上强对齐，防止跨期未来信息漏损")
    return prices

def get_previous_close_price(sym: str, date: datetime, data_bus, price_cache: dict, lookback_days: int = 10):
    """提取前一有效交易日收盘价，服务于隔夜跳空追高防御物理熔断器"""
    for delta in range(1, lookback_days + 1):
        alt_date = date - timedelta(days=delta)
        cache_key = (sym, alt_date, "close")
        if cache_key in price_cache and price_cache[cache_key] is not None:
            return price_cache[cache_key]
            
        p = data_bus.query_by_pit(sym, alt_date, "close")
        if p is not None and not np.isnan(p) and p > 0:
            price_cache[cache_key] = p; return p
    return None