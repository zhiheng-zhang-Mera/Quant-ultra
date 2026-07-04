# -*- coding: utf-8 -*-
"""
Phase 3: Point-in-Time Setup and White-Box Feature Panel Compilation
"""
import logging
from Phase_3.data_loader import load_all_assets_parallel
from Phase_3.step_3_1_regime import run_online_regime_labels
from Phase_3.step_3_2_3_guards import run_preserve_raw_prices_check, run_cross_sectional_guard, run_federated_privacy_firewall
from Phase_3.step_3_4_features import run_whitebox_feature_panel

logger = logging.getLogger("Phase3")

def execute(pipeline_context: dict) -> dict:
    logger.info("=" * 60)
    logger.info("[OP] Deploy Phase_3 Computational Core | [SOURCE] Main Workflow Control Sequence | [RESULT] Activating PIT matrix builder | [SIGNIFICANCE] Compiles whitebox shared vectors and forces local domain boundaries to guard features")
    logger.info("[操作] 部署 Phase_3 核心计算引擎 | [来源] 主控制流编排序列 | [结果] 激活时点信息总线底座 | [意义] 编译跨市场共享特征明文并死锁本土私有特征边界，夯实联邦多域基石")
    logger.info("=" * 60)
    
    if 'trading_days_dt_cn' not in pipeline_context or 'trading_days_dt_us' not in pipeline_context:
        raise ValueError("Missing chronological alignment table structures. Phase 1 & 2 pass required.")
    if 'assets' not in pipeline_context or not pipeline_context['assets']:
        raise ValueError("Shared component target assets vector is empty.")
        
    asset_ohlcv = load_all_assets_parallel(pipeline_context)
    pipeline_context['asset_ohlcv'] = asset_ohlcv
    
    run_online_regime_labels(pipeline_context)
    run_preserve_raw_prices_check(pipeline_context)
    run_cross_sectional_guard(pipeline_context)
    run_whitebox_feature_panel(pipeline_context)
    run_federated_privacy_firewall(pipeline_context)
    
    pipeline_context['pit_setup_ready'] = True
    
    logger.info("[OP] Terminate Phase_3 Core Engine Context | [SOURCE] Purified Hierarchical Panels Trunk | [RESULT] Status set: pit_setup_ready = True | [SIGNIFICANCE] Satisfies rigorous cross-phase data contracts for down-stream machine learning training inputs")
    logger.info("[操作] 终结 Phase_3 核心引擎上下文 | [来源] 纯净化分层特征面板主干 | [结果] 级联状态要素 pit_setup_ready 锁死为 True | [意义] 满足跨域建模的全局 Schema 输入契约，为主干网络的对抗训练和非线性逼近安全交底")
    return pipeline_context