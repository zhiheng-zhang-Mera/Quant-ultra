# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_6 Robust Covariance & Dual-Liquidity Bounds Analyzer
"""
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from sklearn.covariance import LedoitWolf

logger = logging.getLogger("PositionSizing.Utils")

def _get_features_for_date(asset: str, date: datetime, context: dict) -> np.ndarray:
    """直接从 Phase_5 交付的高维特征魔方中检索截面切片，绝对禁止重复特征变换 (Flaw A-4)"""
    feat_cube = context.get('fractional_features_cube')
    trading_days = context.get('trading_days_dt')
    assets = context.get('assets')
    selected = context.get('selected_features', [0, 1, 2, 3, 4])
    
    if feat_cube is None or trading_days is None or assets is None: return None
    ts = pd.Timestamp(date).tz_localize(None)
    
    if ts not in trading_days: return None
    t_idx = trading_days.get_loc(ts)
    if asset not in assets: return None
    a_idx = assets.index(asset)
    
    scaler = context.get('feature_scaler')
    raw_vec = feat_cube[t_idx, a_idx, selected]
    
    if scaler: return scaler.transform(raw_vec.reshape(1, -1)).flatten()
    return raw_vec

def _compute_robust_covariance(bus, assets: list, date: datetime, lookback: int = 252) -> np.ndarray:
    """使用 Ledoit-Wolf 经验收缩估计器解算高维横截面资产协方差，全面适配在途高速内存切片"""
    end_date_str = date.strftime('%Y-%m-%d')
    start_date_str = (date - timedelta(days=int(lookback * 1.5))).strftime('%Y-%m-%d')
    
    # 提取高速预载句柄
    bulk_cache = bus.context.get('bulk_history_cache') if hasattr(bus, 'context') else None
    if not bulk_cache and 'bulk_history_cache' in bus.__dict__: # 针对 context 传递链多样性适配
        bulk_cache = bus.bulk_history_cache
        
    ret_series_list = []
    ts_target = pd.Timestamp(date)

    for sym in assets:
        df = None
        # ⭐ 核心性能提升点：优先撞击全量在途内存表，将 I/O 耗时彻底清零
        if bulk_cache and sym in bulk_cache:
            df_full = bulk_cache[sym]
            df = df_full.loc[start_date_str:end_date_str] # 毫秒级内存检索
        else:
            df = bus.load_asset_history(sym, start_date=start_date_str, end_date=end_date_str)
            
        if df is not None and not df.empty:
            col = 'actual_log_return' if 'actual_log_return' in df.columns else 'log_return'
            # 采用极致平滑的 tail 截取
            s = df[col].tail(lookback)
            if len(s) < lookback:
                # 长度不达标时执行刚性零值对齐补齐
                pad = np.zeros(lookback - len(s))
                ret_series_list.append(np.concatenate([pad, s.values]))
            else:
                ret_series_list.append(s.values)
        else:
            ret_series_list.append(np.zeros(lookback))
            
    X = np.column_stack(ret_series_list)
    X_clean = np.nan_to_num(X, nan=0.0)
    
    try:
        lw = LedoitWolf().fit(X_clean)
        
        if not hasattr(_compute_robust_covariance, "_call_count"):
            _compute_robust_covariance._call_count = 0
        _compute_robust_covariance._call_count += 1
        
        if _compute_robust_covariance._call_count == 1 or _compute_robust_covariance._call_count % 100 == 0:
            logger.info("[OP] Standardize Shrinkage Covariance | [RESULT] Robust Cov Shape: %s | [DAYS] %d", 
                        lw.covariance_.shape, _compute_robust_covariance._call_count)
            
        return lw.covariance_
    except Exception:
        return np.eye(len(assets)) * 0.01

def _compute_market_weights(bus, assets: list, date: datetime) -> np.ndarray:
    """提取各资产自由流通市值比率，构建切片内标准化市场基准先验组合权重"""
    mcap_list = []
    for sym in assets:
        try: mcap_list.append(max(1.0, bus.get_free_float_market_cap(sym, date)))
        except: mcap_list.append(10000.0)
        
    mcap_arr = np.array(mcap_list, dtype=np.float64)
    total_mcap = np.sum(mcap_arr)
    return mcap_arr / (total_mcap if total_mcap > 0 else 1.0)

def _compute_individual_shares_upper(bus, asset: str, date: datetime, nav: float, config: dict) -> float:
    """个人流动性优化限额：双重限额防线（全时序内存切片提速版）"""
    lookback_adv = config.get('lookback_adv', 20)
    end_date_str = date.strftime('%Y-%m-%d')
    start_date_str = (date - timedelta(days=45)).strftime('%Y-%m-%d')
    
    bulk_cache = bus.context.get('bulk_history_cache') if hasattr(bus, 'context') else None
    adv_20 = 0.0
    
    try:
        df = None
        if bulk_cache and asset in bulk_cache:
            df = bulk_cache[asset].loc[start_date_str:end_date_str] # 内存切片
        else:
            df = bus.load_asset_history(asset, start_date=start_date_str, end_date=end_date_str)
            
        if df is not None and not df.empty:
            col = 'amount' if 'amount' in df.columns else 'volume'
            adv_20 = df[col].tail(lookback_adv).mean()
    except Exception: 
        adv_20 = 0.0
        
    if pd.isna(adv_20) or adv_20 <= 0: adv_20 = 20000000.0  # 灾备自愈均值底座
        
    liq_upper = (adv_20 * 0.10) / (nav if nav > 0 else config.get('individual_account_equity', 1e7))
    final_bound = float(np.clip(liq_upper, 0.01, 0.10))
    return final_bound