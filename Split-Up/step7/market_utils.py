# -*- coding: utf-8 -*-
"""
step7/market_utils.py
价格追溯总线、历史买卖价差穿透与自适应冲击系数评估内核
"""
import logging
import numpy as np
import pandas as pd
from datetime import timedelta

logger = logging.getLogger("FSMBacktest.MarketUtils")

def get_prices_for_date(sym_list, date, data_bus, price_cache, lookback_days=10):
    """
    点时对齐价格追溯总线。当T日由于停牌无法拉取价格时，向后平滑穿透lookback_days个物理日
    """
    prices = {}
    for sym in sym_list:
        cache_key = (sym, date)
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

def estimate_adaptive_kappa(sym, current_date, data_bus, config, base_kappa):
    """
    自适应微观结构冲击估算器：
    根据过去60个交易日的高低相对价差，动态逼近市场微观流动性冲击系数 kappa
    """
    lookback = config.get('spread_lookback_days', 60)
    try:
        start_date = current_date - timedelta(days=lookback * 2)
        df = data_bus.load_asset_history(sym, start_date.strftime('%Y-%m-%d'), current_date.strftime('%Y-%m-%d'))
        
        if df is None or len(df) < 20:
            return base_kappa
            
        # 计算横截面滚动相对价差: (high - low) / close
        spread = (df['high'] - df['low']) / df['close']
        avg_spread = spread.tail(lookback).mean()
        
        if np.isnan(avg_spread) or avg_spread <= 0:
            return base_kappa
            
        # 严密数学公式约束: kappa = 5% * 滚动平均价差
        kappa = 0.05 * avg_spread
        
        # 施加边界防御防御区，防止离群值打乱算法
        return float(np.clip(kappa, base_kappa * 0.5, base_kappa * 1.5))
    except Exception as e:
        logger.debug(f"[{current_date}] 资产 {sym} 估算自适应 kappa 失败，降级至基准. 原因: {str(e)}")
        return base_kappa