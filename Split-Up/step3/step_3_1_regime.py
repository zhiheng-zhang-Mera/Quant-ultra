# -*- coding: utf-8 -*-
"""
step3/step_3_1_regime.py
横截面分位数自适应波动率体制计算器（双轨历轴序号硬对齐升级版）
"""

import logging
import numpy as np
import pandas as pd
from step3.config import DEFAULT_VOL_WINDOW

logger = logging.getLogger("Phase3.Regime")


def run_online_regime_labels(context: dict):
    """根据过去指定窗口内的对数波动率，结合双轨日历序号映射服务对资产实施横截面分级体制标签划分"""
    logger.info("[Step 3.1] Constructing online volatility regimes with dual-market sequential alignment.")
    
    assets = context.get('assets', [])
    trading_days_cn = context.get('trading_days_dt_cn', [])
    if not assets or not trading_days_cn:
        raise ValueError("Missing assets or China trading calendar in context.")
        
    config = context.get('config', {})
    vol_window = config.get('vol_window', DEFAULT_VOL_WINDOW)
    
    # 策略全局推进时间轴由 A 股主轨锚定
    current_date_cn = trading_days_cn[-1]
    calendar_alignment = context.get('calendar_alignment', {})
    asset_data = context.get('asset_ohlcv', {})
    vol_dict = {}
    
    for sym in assets:
        df = asset_data.get(sym)
        if df is None or df.empty:
            continue
        
        is_a_share = any(suffix in sym.upper() for suffix in [".SH", ".SZ", ".BJ"]) or sym.isdigit()
        
        # Flow-Pro 1.4: 跨市场特征交互放弃日历日对齐，全面激活序号令牌对齐机制
        if is_a_share:
            target_date = current_date_cn
        else:
            if calendar_alignment:
                date_str_cn = current_date_cn.strftime("%Y-%m-%d") if hasattr(current_date_cn, 'strftime') else str(current_date_cn)[:10]
                seq = calendar_alignment["date_to_seq_cn"].get(date_str_cn)
                if seq is not None and seq in calendar_alignment["seq_to_date_us"]:
                    target_date = pd.to_datetime(calendar_alignment["seq_to_date_us"][seq])
                else:
                    target_date = current_date_cn
            else:
                target_date = current_date_cn
                
        # 时区同步矫正
        if df.index.tz is not None and getattr(target_date, 'tz', None) is None:
            target_date = pd.to_datetime(target_date).tz_localize(df.index.tz)
        elif df.index.tz is None and getattr(target_date, 'tz', None) is not None:
            target_date = pd.to_datetime(target_date).tz_localize(None)
            
        idx = df.index.searchsorted(target_date, side='right') - 1
        if idx < 4 or idx < vol_window - 1:
            continue
            
        start_idx = max(0, idx - vol_window + 1)
        prices = df.iloc[start_idx:idx+1]['close'].values
        if len(prices) < 5:
            continue
            
        # 稳健无偏对数收益率时序波动率计算
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
                regime_map[sym] = 1  # 风险中等兜底
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