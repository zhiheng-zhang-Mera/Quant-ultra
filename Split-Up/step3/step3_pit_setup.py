# -*- coding: utf-8 -*-
"""
Phase 3: Point-in-Time Setup and White-Box Feature Panel Compilation
高层流水线执行总线 - 完全联邦化与隐私分流适配版
"""

import logging
from step3.data_loader import load_all_assets_parallel
from step3.step_3_1_regime import run_online_regime_labels
from step3.step_3_2_3_guards import run_preserve_raw_prices_check, run_cross_sectional_guard, run_federated_privacy_firewall
from step3.step_3_4_features import run_whitebox_feature_panel

logger = logging.getLogger("Phase3")


def execute(pipeline_context: dict) -> dict:
    """
    Phase 3 统一主程序调度入口
    """
    logger.info("=" * 60)
    logger.info("EXECUTING PHASE 3: POINT-IN-TIME SETUP & FEDERATED FEATURE MATRIX")
    logger.info("=" * 60)
    
    # ---- 强依赖前置卡点审查 ----
    if 'trading_days_dt_cn' not in pipeline_context or 'trading_days_dt_us' not in pipeline_context:
        raise ValueError("Missing 'trading_days_dt_cn' or 'trading_days_dt_us' in context. Please ensure Phase 1 & 2 run successfully.")
    if 'assets' not in pipeline_context or not pipeline_context['assets']:
        raise ValueError("No valid assets list transferred to Phase 3.")
        
    # 1. 物理层并发加载数据源并自适应生成特有区域私有底座
    asset_ohlcv = load_all_assets_parallel(pipeline_context)
    pipeline_context['asset_ohlcv'] = asset_ohlcv
    
    # 2. 执行双轨历轴序号硬对齐之自适应体制标签横截面划分
    run_online_regime_labels(pipeline_context)
    
    # 3. 运行前瞻穿透风控审查与交易宇宙锁死守门狗
    run_preserve_raw_prices_check(pipeline_context)
    run_cross_sectional_guard(pipeline_context)
    
    # 4. 深度分层编译多市场隐私分流白盒特征面板矩阵
    run_whitebox_feature_panel(pipeline_context)
    
    # 5. 强力安全熔断卡点：执行联邦隐私防火墙硬合规审计（防止私有特征泄露）
    run_federated_privacy_firewall(pipeline_context)
    
    pipeline_context['pit_setup_ready'] = True
    logger.info("Phase 3 completed successfully with absolute alignment and physical domain isolation.")
    
    return pipeline_context