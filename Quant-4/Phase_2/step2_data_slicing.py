# -*- coding: utf-8 -*-
"""
Phase 2: Full-Flow Two-Tier Data Slicing and Isolation Architecture
"""
import os
import logging
import pandas as pd
from datetime import datetime
from Phase_2.step2_1_slicing import run_moving_window_slicing
from Phase_2.step2_2_validation import run_purge_and_embargo_validation

logger = logging.getLogger("DataSlicing")

def _determine_start_year(pipeline_context: dict) -> int:
    data_bus = pipeline_context.get('data_bus')
    data_manager = pipeline_context.get('data_manager')
    if not data_manager and data_bus and hasattr(data_bus, 'manager'): data_manager = data_bus.manager
        
    assets = pipeline_context.get('assets', [])
    fallback_year = 2022
    if not data_manager or not hasattr(data_manager, 'cache_dir'): return fallback_year
    cache_dir = data_manager.cache_dir
    if not os.path.exists(cache_dir): return fallback_year

    raw_samples = assets[:100] if assets else [f.split('_')[0] for f in os.listdir(cache_dir) if f.endswith('.parquet')][:100]
    if not raw_samples: return fallback_year

    earliest_dates = []
    for sym in raw_samples:
        file_path = os.path.join(cache_dir, f"{sym}_history.parquet")
        if os.path.exists(file_path):
            try:
                df_head = pd.read_parquet(file_path, columns=['date'], engine='pyarrow').head(1)
                if not df_head.empty:
                    first_date = df_head['date'].iloc[0]
                    if pd.to_datetime(first_date).year < 2024: earliest_dates.append(first_date)
            except Exception: continue

    if earliest_dates: return int(pd.to_datetime(max(earliest_dates)).year)
    return fallback_year

def execute(pipeline_context: dict) -> dict:
    logger.info("=" * 60)
    logger.info("[操作] 进入 Phase_2 生命周期主控核心 | [来源] 主中央计算工作流干线 | [结果] 正在执行缓存净化与日历自愈校准循环 | [意义] 锁定阶段1清洗底座与下游资产预测/策略层之间的跨级时有时空隔离带")
    logger.info("=" * 60)
    
    # --------------------------------------------------------
    # 🔄 上下文穿透与同步：捕获 Phase_1 动态注入总线的存活资产池
    # --------------------------------------------------------
    data_bus = pipeline_context.get('data_bus')
    if data_bus and hasattr(data_bus, 'get_universe'):
        latest_assets = data_bus.get_universe()
        # 只有当总线中有资产，且当前字典中的 assets 为空时，才执行状态回插同步
        if latest_assets and not pipeline_context.get('assets'):
            pipeline_context['assets'] = latest_assets
            logger.info(f"[状态对齐] 已从底层总线同步最新 Universe，共提取 {len(latest_assets)} 只存活标的，消除冷启动脱节隐患")

    # --------------------------------------------------------
    # 🛡️ 刚性内存冷冲刷防线：粉碎 Phase_1 残留的毒化及非常态内存对象缓存
    # --------------------------------------------------------
    if data_bus and hasattr(data_bus, '_cache'):
        data_bus._cache.clear()
        logger.info("[操作] 执行内存核心冷冲刷 | [来源] 易失性内存时序数据库缓存区 | [结果] 级联常驻高速缓存已完全清空 | [意义] 强迫下游模块彻底通过边缘数据总线解密健康的物理数据，断绝因残留脏字典引起的数据漏损")

    if 'trading_days_dt_cn' not in pipeline_context or 'trading_days_dt_us' not in pipeline_context:
        logger.warning("[核心警报] 拦截到旧版单轨主控上下文遗留！启动生产级自愈补全程序...")
        detected_start_year = _determine_start_year(pipeline_context)
        current_year = datetime.now().year
        
        if data_bus and hasattr(data_bus, 'manager'):
            cal_cn = data_bus.manager.fetch_trading_calendar(detected_start_year, current_year)
            pipeline_context['trading_days_dt_cn'] = cal_cn.tolist()
            pipeline_context['trading_days_dt_us'] = cal_cn.tolist() 
        else: raise ValueError("Unable to extract active PIT Data Bus instance. Pipeline Validation Halt.")

    run_moving_window_slicing(pipeline_context)
    run_purge_and_embargo_validation(pipeline_context)
    
    pipeline_context['slices_isolated'] = True
    
    logger.info("[操作] 终结 Phase_2 隔离上下文 | [来源] 联邦多市场切片总体蓝图 | [结果] 级联状态要素 slices_isolated 标记锁死为 True | [意义] 确认多市场双轨时空序号防火墙已全面部署并交付完毕")
    return pipeline_context