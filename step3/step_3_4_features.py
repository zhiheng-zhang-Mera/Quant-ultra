# -*- coding: utf-8 -*-
"""
step3/step_3_4_features.py
高细粒度白盒特征核心矩阵计算模块
"""

import logging
import numpy as np
from step3.data_loader import get_last_valid_value, get_value_at_offset

logger = logging.getLogger("Phase3.Features")


def run_whitebox_feature_panel(context: dict):
    """编译五维白盒高保真核心特征面板"""
    logger.info("[Step 3.4] Compiling white-box feature panel (Mom, GK_Vol, Turnover_Shock).")
    
    assets = context.get('current_tradable_universe', context.get('assets', []))
    trading_days = context.get('trading_days_dt', [])
    if not assets or not trading_days:
        raise ValueError("Missing tradable universe or calendar matrix.")
        
    current_date = trading_days[-1]
    asset_data = context.get('asset_ohlcv', {})
    feature_registry = {}
    
    for sym in assets:
        df = asset_data.get(sym)
        if df is None or df.empty:
            continue
            
        # 1. 获取 PIT 截面收盘价
        close_T = get_last_valid_value(df, current_date, 'close')
        if close_T is None or close_T <= 0:
            continue
            
        # 2. 严格时点对齐多跨度收盘价追溯
        close_1 = get_value_at_offset(df, current_date, 'close', 1)
        close_5 = get_value_at_offset(df, current_date, 'close', 5)
        close_20 = get_value_at_offset(df, current_date, 'close', 20)
        if None in [close_1, close_5, close_20] or any(c <= 0 for c in [close_1, close_5, close_20]):
            continue
            
        # 动量计算 (对数变换实现结构平稳性)
        mom_1d = np.log(close_T / close_1)
        mom_5d = np.log(close_T / close_5)
        mom_20d = np.log(close_T / close_20)
        
        # 3. Garman-Klass 波动率计算 (融合开高低收的日内结构)
        open_T = get_last_valid_value(df, current_date, 'open')
        high_T = get_last_valid_value(df, current_date, 'high')
        low_T = get_last_valid_value(df, current_date, 'low')
        if None in [open_T, high_T, low_T] or any(p <= 0 for p in [open_T, high_T, low_T]):
            continue
            
        log_hl = np.log(high_T / low_T)
        log_co = np.log(close_T / open_T)
        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        gk_vol = 0.0 if gk_var < 0 else np.sqrt(gk_var)
        
        # 4. 流动性冲击冲击计算 (Turnover Shock)
        adv_T = get_last_valid_value(df, current_date, 'adv')
        if adv_T is None:
            continue
            
        idx_T = df.index.searchsorted(current_date, side='right') - 1
        if idx_T < 1:
            continue
            
        prev_date = df.index[idx_T - 1]
        adv_ma20_prev = get_last_valid_value(df, prev_date, 'adv_ma20')
        if adv_ma20_prev is None or adv_ma20_prev == 0:
            continue
            
        turnover_shock = (adv_T - adv_ma20_prev) / adv_ma20_prev
        
        # 5. 特征向量灌入与数学有限值校验
        feature_vector = np.array([mom_1d, mom_5d, mom_20d, gk_vol, turnover_shock], dtype=np.float64)
        if np.isfinite(feature_vector).all():
            feature_registry[sym] = feature_vector
            
    context['feature_panel'] = feature_registry
    logger.info(f"Feature panel built successfully for {len(feature_registry)} stocks.")