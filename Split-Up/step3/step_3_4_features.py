# -*- coding: utf-8 -*-
"""
step3/step_3_4_features.py
高细粒度分层分域白盒特征核心矩阵计算模块（完全联邦化重构版）
"""

import logging
import numpy as np
import pandas as pd
from step3.data_loader import get_last_valid_value, get_value_at_offset

logger = logging.getLogger("Phase3.Features")


def run_whitebox_feature_panel(context: dict):
    """编译分层双架构模型特征矩阵（模块一共享明文特征 + 模块二垂直私有隔离特征）"""
    logger.info("[Step 3.4] Compiling split hierarchical feature panel (Shared Matrix + Isolated Local Node Panels).")
    
    assets = context.get('current_tradable_universe', context.get('assets', []))
    trading_days_cn = context.get('trading_days_dt_cn', [])
    if not assets or not trading_days_cn:
        raise ValueError("Missing tradable universe or calendar matrix.")
        
    current_date_cn = trading_days_cn[-1]
    calendar_alignment = context.get('calendar_alignment', {})
    asset_data = context.get('asset_ohlcv', {})
    
    # 初始化三个解耦分流的局部节点特征面板字典
    feature_panel_shared = {}     # 共享特征层（跨市场通用因子，明文直通至协调器）
    feature_panel_private_a = {}  # 垂直私有层 - A股节点（特有制度特征，严格本地隔离）
    feature_panel_private_us = {} # 垂直私有层 - 美股节点（特有制度特征，严格本地隔离）
    
    for sym in assets:
        df = asset_data.get(sym)
        if df is None or df.empty:
            continue
            
        is_a_share = any(suffix in sym.upper() for suffix in [".SH", ".SZ", ".BJ"]) or sym.isdigit()
        
        # Flow-Pro 1.4: 双轨时间窗口序号硬对齐
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
                
        # 时区物理对齐
        if df.index.tz is not None and getattr(target_date, 'tz', None) is None:
            target_date = pd.to_datetime(target_date).tz_localize(df.index.tz)
        elif df.index.tz is None and getattr(target_date, 'tz', None) is not None:
            target_date = pd.to_datetime(target_date).tz_localize(None)
            
        # --------------------------------------------------
        # 【模块一：共享特征层计算（明文通用时序规律因子）】
        # --------------------------------------------------
        close_T = get_last_valid_value(df, target_date, 'close')
        if close_T is None or close_T <= 0:
            continue
            
        close_1 = get_value_at_offset(df, target_date, 'close', 1)
        close_5 = get_value_at_offset(df, target_date, 'close', 5)
        close_20 = get_value_at_offset(df, target_date, 'close', 20)
        if None in [close_1, close_5, close_20] or any(c <= 0 for c in [close_1, close_5, close_20]):
            continue
            
        mom_1d = np.log(close_T / close_1)
        mom_5d = np.log(close_T / close_5)
        mom_20d = np.log(close_T / close_20)
        
        # Garman-Klass 波动率
        open_T = get_last_valid_value(df, target_date, 'open')
        high_T = get_last_valid_value(df, target_date, 'high')
        low_T = get_last_valid_value(df, target_date, 'low')
        if None in [open_T, high_T, low_T] or any(p <= 0 for p in [open_T, high_T, low_T]):
            continue
            
        log_hl = np.log(high_T / low_T)
        log_co = np.log(close_T / open_T)
        gk_var = 0.5 * (log_hl ** 2) - (2 * np.log(2) - 1) * (log_co ** 2)
        gk_vol = 0.0 if gk_var < 0 else np.sqrt(gk_var)
        
        # 流动性冲击 (Turnover Shock)
        adv_T = get_last_valid_value(df, target_date, 'adv')
        if adv_T is None:
            continue
            
        idx_T = df.index.searchsorted(target_date, side='right') - 1
        if idx_T < 1:
            continue
            
        prev_date = df.index[idx_T - 1]
        adv_ma20_prev = get_last_valid_value(df, prev_date, 'adv_ma20')
        if adv_ma20_prev is None or adv_ma20_prev == 0:
            continue
            
        turnover_shock = (adv_T - adv_ma20_prev) / adv_ma20_prev
        
        shared_vector = np.array([mom_1d, mom_5d, mom_20d, gk_vol, turnover_shock], dtype=np.float64)
        if np.isfinite(shared_vector).all():
            feature_panel_shared[sym] = shared_vector
            
        # --------------------------------------------------
        # 【模块二：垂直私有特征层计算（独立节点防火墙留痕）】
        # --------------------------------------------------
        if is_a_share:
            # A股节点私有数据提取：有效涨跌停价格矩阵、ST状态、自由流通市值、北向资金流向、龙虎榜席位数据
            limit_mat = get_last_valid_value(df, target_date, 'Limit_Price_Matrix')
            st_stat = get_last_valid_value(df, target_date, 'ST_Status')
            free_cap = get_last_valid_value(df, target_date, 'Free_Float_Cap')
            north_flow = get_last_valid_value(df, target_date, 'Northbound_Flow')
            seats_data = get_last_valid_value(df, target_date, 'Dragon_Tiger_Seats')
            
            p_vec = np.array([limit_mat, st_stat, free_cap, north_flow, seats_data], dtype=np.float64)
            if np.isfinite(p_vec).all():
                feature_panel_private_a[sym] = p_vec
        else:
            # 美股节点私有数据提取：做空比例、期权隐含波动率、财报季窗口分布、Insider高管交易数据
            short_int = get_last_valid_value(df, target_date, 'Short_Interest')
            vix_imp = get_last_valid_value(df, target_date, 'VIX_Implied')
            earn_win = get_last_valid_value(df, target_date, 'Earnings_Window')
            insider = get_last_valid_value(df, target_date, 'Insider_Trading')
            
            p_vec = np.array([short_int, vix_imp, earn_win, insider], dtype=np.float64)
            if np.isfinite(p_vec).all():
                feature_panel_private_us[sym] = p_vec
                
    # 全量结果独立广播并挂载到环境总线上下文
    context['feature_panel_shared'] = feature_panel_shared
    context['feature_panel_private_a'] = feature_panel_private_a
    context['feature_panel_private_us'] = feature_panel_private_us
    
    # 兼容下游遗留调用
    context['feature_panel'] = feature_panel_shared
    
    logger.info(f"Hierarchical Feature Panel Compilation complete. "
                f"Shared: {len(feature_panel_shared)}, Private A: {len(feature_panel_private_a)}, Private US: {len(feature_panel_private_us)}")