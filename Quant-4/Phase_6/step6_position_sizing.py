# -*- coding: utf-8 -*-
"""
Phase 6: Multi-Market Conformal Position Sizing and Optimization Engine Layer
"""
import logging
import os
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import concurrent.futures
from Phase_6.config import DEFAULT_CONFIG
from Phase_6.directional_mask import step_m_1_directional_mask
from Phase_6.bl_fusion import step_m_2_black_litterman_fusion
from Phase_6.convex_optimizer import step_m_3_convex_optimization
from Phase_6.utils import _get_features_for_date, _compute_robust_covariance

num_cores = str(os.cpu_count() or 4)
os.environ["OMP_NUM_THREADS"] = num_cores
os.environ["OPENBLAS_NUM_THREADS"] = num_cores
os.environ["MKL_NUM_THREADS"] = num_cores
os.environ["VECLIB_MAXIMUM_THREADS"] = num_cores
os.environ["NUMEXPR_NUM_THREADS"] = num_cores

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
        print("=== Phase 6: 测试集时间轴检索 ===")
        for target_key in ['Test', 'test', 'TEST', 'Test-Set', 'test_set', 'Testing', 'testing']:
            if target_key in slices and slices[target_key]:
                test_dates_raw = slices[target_key]
                break
        if not test_dates_raw:
            print("=== Phase 6: 测试集时间轴检索失败，从切片字典中提取非测试集的补集 ===")
            for k, v in slices.items():
                if 'test' in k.lower() and v:
                    test_dates_raw = v
                    break

    if not test_dates_raw and 'trading_days_dt' in local_context:
        print("=== Phase 6: 测试集时间轴检索失败，从全局交易日历中提取非测试集的补集 ===")
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
        
    test_dates = sorted([pd.Timestamp(d).tz_localize(None) for d in test_dates_raw])
    if len(test_dates) > 3:
        logger.warning("🚨 [DEBUG MODE] 触发短时间测试剪裁，已跳过前 %d 天，仅运行最后 3 天！", len(test_dates) - 3)
        test_dates = test_dates[-3:]

    assets = local_context['assets']
    data_bus = local_context['data_bus']
    
    # ==============================================================================
    # ⚡ 行业映射加速模块：缓存 + 多线程 + 扩充批量源
    # ==============================================================================
    CACHE_PATH = Path("data/sector_map_cache.pkl")
    sector_map = None

    # 尝试加载缓存
    if CACHE_PATH.exists():
        try:
            with open(CACHE_PATH, "rb") as f:
                sector_map = pickle.load(f)
            logger.info("Loaded sector_map from cache (%d symbols)", len(sector_map))
            # 检查是否涵盖所有资产（若资产列表变更则重新构建）
            if all(asset in sector_map for asset in assets):
                local_context['sector_map'] = sector_map
                pipeline_context['sector_map'] = sector_map
            else:
                logger.warning("Cache missing some assets, will rebuild.")
                sector_map = None
        except Exception as e:
            logger.warning("Cache load failed: %s, rebuild...", e)
            sector_map = None

    if sector_map is None:
        # ---- 获取akshare模块 ----
        ak_mod = None
        if 'data_bus' in local_context and hasattr(local_context['data_bus'], 'manager') and hasattr(local_context['data_bus'].manager, '_ak'):
            ak_mod = local_context['data_bus'].manager._ak
        if ak_mod is None:
            try:
                import akshare as ak
                ak_mod = ak
            except:
                pass

        # ---- 批量获取行业分类（cninfo + sw） ----
        bulk_map = {}
        if ak_mod is not None:
            # cninfo
            try:
                logger.info("=== Phase 6: Akshare批量获取全市场行业分类映射 (cninfo) ===")
                df_cat = ak_mod.stock_industry_category_cninfo()
                if df_cat is not None and not df_cat.empty:
                    code_col = [c for c in df_cat.columns if '代码' in c or 'code' in c.lower() or 'Symbol' in c]
                    name_col = [c for c in df_cat.columns if '行业' in c or 'name' in c.lower()]
                    if code_col and name_col:
                        for _, row in df_cat.iterrows():
                            c_str = str(row[code_col[0]]).strip().zfill(6)
                            bulk_map[c_str] = str(row[name_col[0]]).strip()
            except Exception as e:
                logger.warning("cninfo bulk fetch failed: %s", e)

            # sw (申万)
            try:
                logger.info("=== Phase 6: Akshare批量获取全市场行业分类映射 (申万) ===")
                df_sw = ak_mod.stock_industry_category_sw()
                if df_sw is not None and not df_sw.empty:
                    code_col = [c for c in df_sw.columns if '代码' in c or 'code' in c.lower() or 'Symbol' in c]
                    name_col = [c for c in df_sw.columns if '行业' in c or 'name' in c.lower()]
                    if code_col and name_col:
                        for _, row in df_sw.iterrows():
                            c_str = str(row[code_col[0]]).strip().zfill(6)
                            # 仅当未覆盖时补充（cninfo优先）
                            if c_str not in bulk_map:
                                bulk_map[c_str] = str(row[name_col[0]]).strip()
            except Exception as e:
                logger.warning("SW bulk fetch failed: %s", e)

        # ---- 第一遍：直接命中批量映射 ----
        sector_map = {}
        missing_syms = []
        for sym in assets:
            pure_code = sym.split('.')[0]
            if pure_code in bulk_map:
                sector_map[sym] = bulk_map[pure_code]
            else:
                missing_syms.append(sym)

        # ---- 第二遍：多线程并发获取缺失股票 ----
        if missing_syms and ak_mod is not None:
            logger.info("Fetching sector for %d missing symbols via multi-thread...", len(missing_syms))
            def fetch_sector(sym):
                pure_code = sym.split('.')[0]
                try:
                    df_info = ak_mod.stock_individual_info_em(symbol=pure_code)
                    if df_info is not None and not df_info.empty:
                        row = df_info[df_info['item'] == '行业']
                        if not row.empty:
                            return sym, str(row['value'].values[0]).strip()
                except Exception:
                    pass
                # fallback硬编码映射
                if pure_code in ['600036', '601988', '601398', '000001']:
                    return sym, "银行"
                elif pure_code in ['600030', '600837', '000776']:
                    return sym, "非银金融"
                elif pure_code in ['600519', '000858', '600809']:
                    return sym, "食品饮料"
                elif pure_code in ['600028', '601857', '600583']:
                    return sym, "石油石化"
                else:
                    return sym, "综合"

            with concurrent.futures.ThreadPoolExecutor(max_workers=int(config.get('optimization_workers', 16))) as executor:
                futures = {executor.submit(fetch_sector, sym): sym for sym in missing_syms}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        sym, sec = future.result()
                        sector_map[sym] = sec
                    except Exception:
                        sym = futures[future]
                        sector_map[sym] = "综合"

        # ---- 确保所有资产都有行业，未覆盖的置为“综合” ----
        for sym in assets:
            if sym not in sector_map:
                sector_map[sym] = "综合"

        # ---- 保存缓存 ----
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(CACHE_PATH, "wb") as f:
                pickle.dump(sector_map, f)
            logger.info("Saved sector_map cache to %s", CACHE_PATH)
        except Exception as e:
            logger.warning("Failed to save cache: %s", e)

    # 注入上下文
    local_context['sector_map'] = sector_map
    pipeline_context['sector_map'] = sector_map

    # ==============================================================================
    # ⚡ 核心提速层：全资产全时序数据在途特征高速内存化预载 (I/O 降维打击)
    # ==============================================================================
    logger.info("[性能] 初始化横截面全量历史数据预载器...")
    bulk_history_cache = {}
    min_date = test_dates[0] - pd.Timedelta(days=400)
    max_date = test_dates[-1]
    
    counter = 0
    for sym in assets:
        if counter % 200 == 0: logger.info(f"=== Phase 6: 历史数据预载进度 {counter}/{len(assets)} ===")
        counter += 1
        try:
            df_hist = data_bus.load_asset_history(sym, min_date.strftime('%Y-%m-%d'), max_date.strftime('%Y-%m-%d'))
            if df_hist is not None and not df_hist.empty:
                df_hist.index = pd.to_datetime(df_hist.index).tz_localize(None)
                # 预计算ADV20
                if 'amount' in df_hist.columns:
                    df_hist['precalc_adv20'] = df_hist['amount'].rolling(20).mean()
                elif 'volume' in df_hist.columns and 'close' in df_hist.columns:
                    df_hist['precalc_adv20'] = (df_hist['volume'] * df_hist['close']).rolling(20).mean()
                else:
                    df_hist['precalc_adv20'] = 20000000.0
                # 预计算收益率
                if 'actual_log_return' in df_hist.columns:
                    df_hist['precalc_return'] = df_hist['actual_log_return']
                elif 'log_return' in df_hist.columns:
                    df_hist['precalc_return'] = df_hist['log_return']
                else:
                    price_cols = [c for c in ['adj_close', 'adjclose', 'total_return_price', 'close', 'Close'] if c in df_hist.columns]
                    if price_cols:
                        df_hist['precalc_return'] = np.log(df_hist[price_cols[0]].astype(float) / df_hist[price_cols[0]].astype(float).shift(1))
                        df_hist['precalc_return'] = df_hist['precalc_return'].replace([np.inf, -np.inf], 0.0).fillna(0.0)
                    else:
                        df_hist['precalc_return'] = 0.0
                bulk_history_cache[sym] = df_hist
        except Exception as e:
            logger.warning("[性能] 预载 %s 历史数据失败: %s", sym, e)
            
    local_context['bulk_history_cache'] = bulk_history_cache

    weight_records = []
    w_prev = np.zeros(len(assets))
    current_nav = config.get('individual_account_equity', 10000000.0)

    counter = 0
    for t_date in test_dates:
        logger.info(f"=== Phase 6: 正在计算 {t_date.strftime('%Y-%m-%d')} 的每日权重向量 | 进度 {counter}/{len(test_dates)} ===")
        counter += 1
        local_context['directional_symbol_masks'] = step_m_1_directional_mask(local_context, t_date)
        
        R_BL, Sigma_robust, Q_view, Omega_diag = step_m_2_black_litterman_fusion(local_context, t_date, w_prev)
        local_context.update({'R_BL': R_BL, 'Sigma_robust': Sigma_robust, 'Q_view': Q_view, 'Omega_diag': Omega_diag})
        
        w_new = step_m_3_convex_optimization(local_context, t_date, current_nav, w_prev)
        weight_records.append(w_new)
        w_prev = w_new.copy()

    weights_df = pd.DataFrame(weight_records, index=[d.strftime('%Y-%m-%d') for d in test_dates], columns=assets)
    intervals_df = pd.DataFrame(0.02, index=[d.strftime('%Y-%m-%d') for d in test_dates], columns=assets)
    
    # ==============================================================================
    # ADV20 面板计算（全内存向量化）
    # ==============================================================================
    logger.info("=== Phase 6: 全资产全时序数据在途特征高速内存化预载,开始计算 ADV20 ===")
    adv_records = []
    for date in test_dates:
        daily_adv = []
        for asset in assets:
            adv_val = 20000000.0
            if asset in bulk_history_cache:
                df_asset = bulk_history_cache[asset]
                if date in df_asset.index and pd.notna(df_asset.at[date, 'precalc_adv20']):
                    adv_val = float(df_asset.at[date, 'precalc_adv20'])
            daily_adv.append(adv_val)
        if len(daily_adv) % 50 == 0: logger.info(f"=== Phase 6: 已完成 {len(daily_adv)} / {len(assets)} 个资产的 ADV20 计算 ===")
        adv_records.append(daily_adv)
        
    adv20_df = pd.DataFrame(adv_records, index=[d.strftime('%Y-%m-%d') for d in test_dates], columns=assets)

    logger.info("Phase 6结束,每日权重矩阵形状: %s", weights_df.shape)
    
    return {
        'daily_weights': weights_df,
        'daily_intervals': intervals_df,
        'daily_adv20': adv20_df,
        'position_sizing_ready': True
    }
