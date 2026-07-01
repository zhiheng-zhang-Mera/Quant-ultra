# -*- coding: utf-8 -*-
"""
Phase 3: Point-in-Time Setup and White-Box Feature Panel Compilation
高层流水线执行总线 - 完全适配大工程解耦流
"""

import logging
from step3.data_loader import load_all_assets_parallel
from step3.step_3_1_regime import run_online_regime_labels
from step3.step_3_2_3_guards import run_preserve_raw_prices_check, run_cross_sectional_guard
from step3.step_3_4_features import run_whitebox_feature_panel

logger = logging.getLogger("Phase3")


def execute(pipeline_context: dict) -> dict:
    """
    Phase 3 统一主程序调度入口
    """
    logger.info("=" * 60)
    logger.info("EXECUTING PHASE 3: POINT-IN-TIME SETUP & FEATURE MATRIX")
    logger.info("=" * 60)
    
    # ---- 强依赖前置卡点审查 ----
    if 'trading_days_dt' not in pipeline_context:
        raise ValueError("Missing 'trading_days_dt' in context. Please ensure Phase 1 & 2 run successfully.")
    if 'assets' not in pipeline_context or not pipeline_context['assets']:
        raise ValueError("No valid assets list transferred to Phase 3.")
        
    # 1. 物理层并发加载数据源
    asset_ohlcv = load_all_assets_parallel(pipeline_context)
    pipeline_context['asset_ohlcv'] = asset_ohlcv
    
    # 2. 执行波动率自适应体制标签生成
    run_online_regime_labels(pipeline_context)
    
    # 3. 运行前瞻穿透风控审查与交易宇宙锁死守门狗
    run_preserve_raw_prices_check(pipeline_context)
    run_cross_sectional_guard(pipeline_context)
    
    # 4. 深度编译时点对齐特征面板矩阵
    run_whitebox_feature_panel(pipeline_context)
    
    pipeline_context['pit_setup_ready'] = True
    logger.info("Phase 3 completed successfully with absolute alignment.")
    
    return pipeline_context