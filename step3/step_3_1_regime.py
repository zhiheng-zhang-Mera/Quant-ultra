# -*- coding: utf-8 -*-
"""
step3/step_3_1_regime.py
横截面分位数自适应波动率体制计算器
"""

import logging
import numpy as np
from step3.config import DEFAULT_VOL_WINDOW

logger = logging.getLogger("Phase3.Regime")


def run_online_regime_labels(context: dict):
    """根据过去指定窗口内的对数波动率，对资产实施横截面分级标签划分"""
    logger.info("[Step 3.1] Constructing online volatility regimes with cross-sectional quantile bins.")
    
    assets = context.get('assets', [])
    trading_days = context.get('trading_days_dt', [])
    if not assets or not trading_days:
        raise ValueError("Missing assets or trading calendar in context.")
        
    config = context.get('config', {})
    vol_window = config.get('vol_window', DEFAULT_VOL_WINDOW)
    
    current_date = trading_days[-1]
    asset_data = context.get('asset_ohlcv', {})
    vol_dict = {}
    
    for sym in assets:
        df = asset_data.get(sym)
        if df is None or df.empty:
            continue
            
        idx = df.index.searchsorted(current_date, side='right') - 1
        if idx < 4 or idx < vol_window - 1:
            continue
            
        start_idx = max(0, idx - vol_window + 1)
        prices = df.iloc[start_idx:idx+1]['close'].values
        if len(prices) < 5:
            continue
            
        # 稳健无偏对数收益率波动率计算
        rets = np.diff(np.log(prices))
        if len(rets) > 0:
            vol = np.std(rets) * np.sqrt(252)
            vol_dict[sym] = vol
            
    if not vol_dict:
        logger.warning("No valid volatility computed; fallback all assets to medium regime (1).")
        regime_map = {sym: 1 for sym in assets}
    else:
        vols = np.array(list(vol_dict.values()))
        lower_q = np.percentile(vols, 33)
        upper_q = np.percentile(vols, 67)
        
        regime_map = {}
        for sym in assets:
            vol = vol_dict.get(sym)
            if vol is None or np.isnan(vol):
                regime_map[sym] = 1  # 填充中等风险兜底
            elif vol < lower_q:
                regime_map[sym] = 0  # 低波
            elif vol > upper_q:
                regime_map[sym] = 2  # 高波
            else:
                regime_map[sym] = 1  # 中波
                
    context['online_regime_state'] = regime_map
    logger.info(f"Regime labels successfully built: low={sum(1 for v in regime_map.values() if v==0)}, "
                f"medium={sum(1 for v in regime_map.values() if v==1)}, "
                f"high={sum(1 for v in regime_map.values() if v==2)}")