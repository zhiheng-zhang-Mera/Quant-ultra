# -*- coding: utf-8 -*-
"""
Phase 6: Multi-Market Conformal Position Sizing and Optimization Engine Layer
"""
import logging
import pandas as pd
import numpy as np
from Phase_6.config import DEFAULT_CONFIG
from Phase_6.directional_mask import step_m_1_directional_mask
from Phase_6.bl_fusion import step_m_2_black_litterman_fusion
from Phase_6.convex_optimizer import step_m_3_convex_optimization

logger = logging.getLogger("PositionSizing")

def execute(pipeline_context: dict) -> dict:
    logger.info("=" * 60)
    logger.info("[OP] Enter Phase_6 Sizing Orchestrator | [SOURCE] Global Main Pipeline Stream")
    logger.info("=" * 60)

    config = pipeline_context.setdefault('config', DEFAULT_CONFIG.copy())
    for k, v in DEFAULT_CONFIG.items(): config.setdefault(k, v)
        
    local_context = pipeline_context.copy()
    slices = local_context.get('slices', {})
    
    # ---- 时轴自适应检索与自愈 ----
    test_dates_raw = []
    if isinstance(slices, dict):
        for target_key in ['Test', 'test', 'TEST', 'Test-Set', 'test_set', 'Testing', 'testing']:
            if target_key in slices and slices[target_key]:
                test_dates_raw = slices[target_key]
                break
        if not test_dates_raw:
            for k, v in slices.items():
                if 'test' in k.lower() and v:
                    test_dates_raw = v
                    break

    if not test_dates_raw and 'trading_days_dt' in local_context:
        total_timeline = local_context['trading_days_dt']
        if total_timeline is not None and len(total_timeline) > 0:
            known_non_test_dates = set()
            if isinstance(slices, dict):
                for k, v in slices.items():
                    if 'test' not in k.lower() and isinstance(v, (list, tuple, pd.Index, np.ndarray)):
                        for d in v:
                            if isinstance(d, str): known_non_test_dates.add(d[:10])
                            elif hasattr(d, 'strftime'): known_non_test_dates.add(d.strftime('%Y-%m-%d'))
            
            total_strs = []
            for d in total_timeline:
                if isinstance(d, str): total_strs.append(d[:10])
                elif hasattr(d, 'strftime'): total_strs.append(d.strftime('%Y-%m-%d'))
                else: total_strs.append(str(d)[:10])
                
            max_known_date_str = max(known_non_test_dates) if known_non_test_dates else "1970-01-01"
            test_dates_raw = [orig for orig, d_str in zip(total_timeline, total_strs) if d_str not in known_non_test_dates and d_str > max_known_date_str]

    if not test_dates_raw: 
        raise ValueError("Test chronology partition vacant. Even calendar complement extraction failed.")
        
    # 🛡️ 修复点 1：将计算测试集时间轴统一转化为刚性的 tz-naive（无时区）Timestamp 对象
    test_dates = sorted([pd.Timestamp(d).tz_localize(None) for d in test_dates_raw])
    assets = local_context['assets']
    data_bus = local_context['data_bus']
    
    # ---- 行业映射接管层 ----
    ak_mod = None
    if 'data_bus' in local_context and hasattr(local_context['data_bus'], 'manager') and hasattr(local_context['data_bus'].manager, '_ak'):
        ak_mod = local_context['data_bus'].manager._ak
    if ak_mod is None:
        try: import akshare as ak; ak_mod = ak
        except: pass

    sector_map = {}
    cninfo_bulk_map = {}
    if ak_mod is not None:
        try:
            df_cat = ak_mod.stock_industry_category_cninfo()
            if df_cat is not None and not df_cat.empty:
                code_col = [c for c in df_cat.columns if '代码' in c or 'code' in c.lower() or 'Symbol' in c]
                name_col = [c for c in df_cat.columns if '行业' in c or 'name' in c.lower()]
                if code_col and name_col:
                    for _, row_data in df_cat.iterrows():
                        c_str = str(row_data[code_col[0]]).strip().zfill(6)
                        cninfo_bulk_map[c_str] = str(row_data[name_col[0]]).strip()
        except: pass

    for sym in assets:
        pure_code = sym.split('.')[0]
        if pure_code in cninfo_bulk_map: sector_map[sym] = cninfo_bulk_map[pure_code]; continue
        if ak_mod is not None:
            try:
                df_info = ak_mod.stock_individual_info_em(symbol=pure_code)
                if df_info is not None and not df_info.empty:
                    target_row = df_info[df_info['item'] == '行业']
                    if not target_row.empty: sector_map[sym] = str(target_row['value'].values[0]).strip(); continue
            except: pass
        if pure_code in ['600036', '601988', '601398', '000001']: sector_map[sym] = "银行"
        elif pure_code in ['600030', '600837', '000776']: sector_map[sym] = "非银金融"
        elif pure_code in ['600519', '000858', '600809']: sector_map[sym] = "食品饮料"
        elif pure_code in ['600028', '601857', '600583']: sector_map[sym] = "石油石化"
        else: sector_map[sym] = "综合"

    local_context['sector_map'] = sector_map
    pipeline_context['sector_map'] = sector_map

    # ==============================================================================
    # ⚡ 核心提速层：全资产全时序数据在途特征高速内存化预载 (I/O 降维打击)
    # ==============================================================================
    logger.info("[PERF] Initializing Cross-Sectional Bulk History Pre-fetcher...")
    bulk_history_cache = {}
    min_date = test_dates[0] - pd.Timedelta(days=400) # 覆盖 252 日协方差滚动回溯窗口
    max_date = test_dates[-1]
    
    for sym in assets:
        try:
            df_hist = data_bus.load_asset_history(sym, min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d'))
            if df_hist is not None and not df_hist.empty:
                # 🛡️ 修复点 2：强行保障 DatetimeIndex 加速内存切片，并强制剥离任何潜在时区 (tz-localize(None))
                df_hist.index = pd.to_datetime(df_hist.index).tz_localize(None)
                bulk_history_cache[sym] = df_hist
        except Exception as e:
            logger.warning("[PERF] Pre-fetch failed for %s: %s", sym, e)
            
    # 将高速内存句柄常驻局部上下文
    local_context['bulk_history_cache'] = bulk_history_cache
    # ==============================================================================

    weight_records = []
    w_prev = np.zeros(len(assets))
    current_nav = config.get('individual_account_equity', 10000000.0)

    # 顺次沿着测试样本外时间轴推进每日横截面凸优化解算 (刚性全量计算，无热加载短路)
    for t_date in test_dates:
        local_context['directional_symbol_masks'] = step_m_1_directional_mask(local_context, t_date)
        
        R_BL, Sigma_robust, Q_view, Omega_diag = step_m_2_black_litterman_fusion(local_context, t_date, w_prev)
        local_context.update({'R_BL': R_BL, 'Sigma_robust': Sigma_robust, 'Q_view': Q_view, 'Omega_diag': Omega_diag})
        
        w_new = step_m_3_convex_optimization(local_context, t_date, current_nav, w_prev)
        weight_records.append(w_new)
        w_prev = w_new.copy()

    weights_df = pd.DataFrame(weight_records, index=[d.strftime('%Y-%m-%d') for d in test_dates], columns=assets)
    intervals_df = pd.DataFrame(0.02, index=[d.strftime('%Y-%m-%d') for d in test_dates], columns=assets)
    
    # ==============================================================================
    # ⚡ 核心提速层：彻底重构尾部每日 ADV20 面板解算（全内存向量化，摒弃重型 I/O）
    # ==============================================================================
    adv_records = []
    for date in test_dates:
        daily_adv = []
        for asset in assets:
            adv_val = 20000000.0  # 兜底底座
            if asset in bulk_history_cache:
                df_asset = bulk_history_cache[asset]
                # 🛡️ 此时 df_asset.index 与 date 双方都是绝对无时区的 tz-naive 状态，切片完美契合
                hist_slice = df_asset.loc[:date].tail(20)
                if len(hist_slice) >= 20:
                    if 'amount' in hist_slice.columns:
                        adv_val = float(hist_slice['amount'].mean())
                    elif 'volume' in hist_slice.columns and 'close' in hist_slice.columns:
                        adv_val = float((hist_slice['volume'] * hist_slice['close']).mean())
            daily_adv.append(adv_val)
        adv_records.append(daily_adv)
        
    adv20_df = pd.DataFrame(adv_records, index=[d.strftime('%Y-%m-%d') for d in test_dates], columns=assets)
    # ==============================================================================

    logger.info("[OP] Finish Phase_6 Sizing Loop | [RESULT] Weight Matrix Shape: %s", weights_df.shape)
    
    return {
        'daily_weights': weights_df,
        'daily_intervals': intervals_df,
        'daily_adv20': adv20_df,
        'position_sizing_ready': True
    }