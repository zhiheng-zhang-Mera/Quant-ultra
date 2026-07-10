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

# def _compute_robust_covariance(bus, assets: list, date: datetime, lookback: int = 252) -> np.ndarray:
#     """使用 Ledoit-Wolf 经验收缩估计器解算高维横截面资产协方差，全面适配在途高速内存切片"""
#     end_date_str = date.strftime('%Y-%m-%d')
#     start_date_str = (date - timedelta(days=int(lookback * 1.5))).strftime('%Y-%m-%d')
    
#     # 提取高速预载句柄
#     bulk_cache = bus.context.get('bulk_history_cache') if hasattr(bus, 'context') else None
#     if not bulk_cache and 'bulk_history_cache' in bus.__dict__: # 针对 context 传递链多样性适配
#         bulk_cache = bus.bulk_history_cache
        
#     ret_series_list = []
#     ts_target = pd.Timestamp(date)

#     for sym in assets:
#         df = None
#         # ⭐ 核心性能提升点：优先撞击全量在途内存表，将 I/O 耗时彻底清零
#         # if bulk_cache and sym in bulk_cache:
#         #    df_full = bulk_cache[sym]
#         #    df = df_full.loc[start_date_str:end_date_str] # 毫秒级内存检索
#         #else:
#         #    df = bus.load_asset_history(sym, start_date=start_date_str, end_date=end_date_str)

#         if bulk_cache and sym in bulk_cache:
#             df = bulk_cache[sym]
#             # 直接提取预计算好的安全切片，规避 DataFrame 内部复杂的引擎解析
#             df_slice = df.loc[start_date_str:end_date_str]
#             s = df_slice['precalc_return'].tail(lookback)
            
#             if len(s) < lookback:
#                 pad = np.zeros(lookback - len(s))
#                 ret_series_list.append(np.concatenate([pad, s.values]))
#             else:
#                 ret_series_list.append(s.values)
#         else:
#             ret_series_list.append(np.zeros(lookback))

#         if df is not None and not df.empty:
#             # --------------------------------------------------------
#             # 🔄 收益率列动态嗅探与现场推导防线
#             # --------------------------------------------------------
#             if 'actual_log_return' in df.columns:
#                 col = 'actual_log_return'
#             elif 'log_return' in df.columns:
#                 col = 'log_return'
#             else:
#                 # 寻找收盘价列现场计算对数收益率
#                 price_cols = [c for c in ['adj_close', 'adjclose', 'total_return_price', 'close', 'Close'] if c in df.columns]
#                 if price_cols:
#                     price_col = price_cols[0]
#                     col = 'derived_log_return'
#                     # 现场计算对数收益率：ln(P_t / P_{t-1})
#                     df[col] = np.log(df[price_col].astype(float) / df[price_col].astype(float).shift(1))
#                     # 强力清洗计算产生的无穷大和空值
#                     df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)
#                 else:
#                     # 极端兜底：连价格列都找不到，退化为0序列以防止协方差矩阵崩溃
#                     col = 'fallback_zeros'
#                     df[col] = 0.0
                    
#             # 采用极致平滑的 tail 截取
#             s = df[col].tail(lookback)
#             if len(s) < lookback:
#                 # 长度不达标时执行刚性零值对齐补齐
#                 pad = np.zeros(lookback - len(s))
#                 ret_series_list.append(np.concatenate([pad, s.values]))
#             else:
#                 ret_series_list.append(s.values)
#         else:
#             ret_series_list.append(np.zeros(lookback))
            
#     X = np.column_stack(ret_series_list)
#     X_clean = np.nan_to_num(X, nan=0.0)
    
#     try:
#         lw = LedoitWolf().fit(X_clean)
        
#         if not hasattr(_compute_robust_covariance, "_call_count"):
#             _compute_robust_covariance._call_count = 0
#         _compute_robust_covariance._call_count += 1
        
#         if _compute_robust_covariance._call_count == 1 or _compute_robust_covariance._call_count % 100 == 0:
#             logger.info("[OP] Standardize Shrinkage Covariance | [RESULT] Robust Cov Shape: %s | [DAYS] %d", 
#                         lw.covariance_.shape, _compute_robust_covariance._call_count)
            
#         return lw.covariance_
#     except Exception:
#         return np.eye(len(assets)) * 0.01

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
    
    # try:
    #    df = None
    #    if bulk_cache and asset in bulk_cache:
    #        df = bulk_cache[asset].loc[start_date_str:end_date_str] # 内存切片
    #    else:
    #        df = bus.load_asset_history(asset, start_date=start_date_str, end_date=end_date_str)
    #        
    #    if df is not None and not df.empty:
    #        col = 'amount' if 'amount' in df.columns else 'volume'
    #        adv_20 = df[col].tail(lookback_adv).mean()
    #except Exception: 
    #    adv_20 = 0.0
    
    try:
        # ⚡ 极速路径：直接从预计算缓存命中单点标量
        if bulk_cache and asset in bulk_cache:
            df = bulk_cache[asset]
            if date in df.index and pd.notna(df.at[date, 'precalc_adv20']):
                adv_20 = df.at[date, 'precalc_adv20']
        else:
            # 灾备路径保留原样
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

# def _compute_robust_covariance(bus, assets: list, date: datetime, lookback: int = 252) -> np.ndarray:
#     """使用 Ledoit-Wolf 经验收缩估计器解算高维横截面资产协方差，融合高速切片与自愈推导"""
#     end_date_str = date.strftime('%Y-%m-%d')
#     start_date_str = (date - timedelta(days=int(lookback * 1.5))).strftime('%Y-%m-%d')
    
#     # 提取高速预载句柄
#     bulk_cache = bus.context.get('bulk_history_cache') if hasattr(bus, 'context') else None
#     if not bulk_cache and 'bulk_history_cache' in bus.__dict__: 
#         bulk_cache = bus.bulk_history_cache
        
#     ret_series_list = []

#     for sym in assets:
#         s_vals = None
        
#         # ⚡ 第一道防线：极速内存预计算列截取
#         if bulk_cache and sym in bulk_cache:
#             df_slice = bulk_cache[sym].loc[start_date_str:end_date_str]
#             if 'precalc_return' in df_slice.columns:
#                 s_vals = df_slice['precalc_return'].tail(lookback).values
                
#         # 🛡️ 第二道防线：内存穿透或预计算缺失时，现场提取 K 线并推导
#         if s_vals is None or len(s_vals) == 0:
#             df = bus.load_asset_history(sym, start_date=start_date_str, end_date=end_date_str)
#             if df is not None and not df.empty:
#                 if 'actual_log_return' in df.columns:
#                     col_s = df['actual_log_return']
#                 elif 'log_return' in df.columns:
#                     col_s = df['log_return']
#                 else:
#                     # 现场推导收益率
#                     price_cols = [c for c in ['adj_close', 'adjclose', 'total_return_price', 'close', 'Close'] if c in df.columns]
#                     if price_cols:
#                         price_col = price_cols[0]
#                         col_s = np.log(df[price_col].astype(float) / df[price_col].astype(float).shift(1))
#                         col_s = col_s.replace([np.inf, -np.inf], 0.0).fillna(0.0)
#                     else:
#                         col_s = pd.Series(0.0, index=df.index)
#                 s_vals = col_s.tail(lookback).values
                
#         # ⚖️ 终端拦截器：刚性对齐长度，绝不执行重复 Append！
#         if s_vals is None:
#             ret_series_list.append(np.zeros(lookback))
#         elif len(s_vals) < lookback:
#             pad = np.zeros(lookback - len(s_vals))
#             ret_series_list.append(np.concatenate([pad, s_vals]))
#         else:
#             ret_series_list.append(s_vals)
            
#     # 横向堆叠并处理脏值
#     X = np.column_stack(ret_series_list)
#     X_clean = np.nan_to_num(X, nan=0.0)
    
#     try:
#         lw = LedoitWolf().fit(X_clean)
        
#         if not hasattr(_compute_robust_covariance, "_call_count"):
#             _compute_robust_covariance._call_count = 0
#         _compute_robust_covariance._call_count += 1
        
#         if _compute_robust_covariance._call_count == 1 or _compute_robust_covariance._call_count % 100 == 0:
#             logger.info("[OP] Standardize Shrinkage Covariance | [RESULT] Robust Cov Shape: %s | [DAYS] %d", 
#                         lw.covariance_.shape, _compute_robust_covariance._call_count)
            
#         return lw.covariance_
#     except Exception:
#         return np.eye(len(assets)) * 0.01

try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

def _compute_robust_covariance(bus, assets: list, date: datetime, lookback: int = 252) -> np.ndarray:
    """使用异构加速与 Tikhonov 代数平替，彻底消除 sklearn 单线程瓶颈"""
    end_date_str = date.strftime('%Y-%m-%d')
    start_date_str = (date - timedelta(days=int(lookback * 1.5))).strftime('%Y-%m-%d')
    
    bulk_cache = bus.context.get('bulk_history_cache') if hasattr(bus, 'context') else None
    if not bulk_cache and 'bulk_history_cache' in bus.__dict__: 
        bulk_cache = bus.bulk_history_cache
        
    ret_series_list = []
    for sym in assets:
        s_vals = None
        if bulk_cache and sym in bulk_cache:
            df_slice = bulk_cache[sym].loc[start_date_str:end_date_str]
            if 'precalc_return' in df_slice.columns:
                s_vals = df_slice['precalc_return'].tail(lookback).values
                
        if s_vals is None or len(s_vals) == 0:
            df = bus.load_asset_history(sym, start_date=start_date_str, end_date=end_date_str)
            if df is not None and not df.empty:
                col_s = df.get('actual_log_return', df.get('log_return'))
                if col_s is None:
                    price_cols = [c for c in ['adj_close', 'adjclose', 'total_return_price', 'close', 'Close'] if c in df.columns]
                    if price_cols:
                        col_s = np.log(df[price_cols[0]].astype(float) / df[price_cols[0]].astype(float).shift(1))
                        col_s = col_s.replace([np.inf, -np.inf], 0.0).fillna(0.0)
                    else: col_s = pd.Series(0.0, index=df.index)
                s_vals = col_s.tail(lookback).values
                
        if s_vals is None: ret_series_list.append(np.zeros(lookback))
        elif len(s_vals) < lookback: ret_series_list.append(np.concatenate([np.zeros(lookback - len(s_vals)), s_vals]))
        else: ret_series_list.append(s_vals)
            
    X_clean = np.nan_to_num(np.column_stack(ret_series_list), nan=0.0)
    
    # ==============================================================================
    # ⚡ 核心提速：代数平替与 GPU/多核并行协方差解算
    # 原理：直接进行高维矩阵乘法 S = X^T @ X，并采用先验常数替代昂贵的逐步收缩寻找
    # ==============================================================================
    try:
        if GPU_AVAILABLE:
            cp_X = cp.array(X_clean)
            cp_X_centered = cp_X - cp.mean(cp_X, axis=0)
            # 毫秒级 GPU 样本协方差
            cp_S = (cp_X_centered.T @ cp_X_centered) / (lookback - 1)
            
            # Ledoit-Wolf 平替：向均值方差对角矩阵收缩 (Shrinkage=0.1 经验高维优值)
            mean_var = cp.mean(cp.diag(cp_S))
            cp_target = cp.eye(len(assets)) * mean_var
            cp_Sigma = 0.9 * cp_S + 0.1 * cp_target
            
            cov_mat = cp.asnumpy(cp_Sigma)
        else:
            X_centered = X_clean - np.mean(X_clean, axis=0)
            # 激活 OS 环境变量后的多核 CPU 极限乘法
            S = (X_centered.T @ X_centered) / (lookback - 1)
            
            mean_var = np.mean(np.diag(S))
            target = np.eye(len(assets)) * mean_var
            cov_mat = 0.9 * S + 0.1 * target
            
        return cov_mat
    except Exception as e:
        logger.warning(f"协方差平替运算降级: {e}")
        return np.eye(len(assets)) * 0.01