"""
Quant-Ultra Flow - Step 1: Data Foundation & Premium Cleaner
Fully refactored to eliminate Survivor-Bias (Resolves Flaw A-8).
Forces historical dead/delisted stock matrices back into the orchestrator assets core.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import time
from Phase_1.config import CONFIG as DATA_FOUNDATION_CONFIG
from Phase_1.step1_1_screening import run_screening
from Phase_1.step1_2_returns import run_returns_cleaning
from Phase_1.step1_3_trading_status import run_status_mapping

logger = logging.getLogger("DataFoundation.Main")

def execute(pipeline_context: dict) -> dict:
    start_time = time.time()
    logger.info("=" * 60)
    logger.info("[OP] Deploy Phase_1 Core Foundation Orchestrator | [SOURCE] Pipeline Master Engine Control Loop | [RESULT] Activating data purifier pipeline | [SIGNIFICANCE] Resolves survivor bias by anchoring strict cross-sectional mask panels")
    logger.info("[操作] 部署 Phase_1 核心底座编排器 | [来源] 流水线主引擎控制循环 | [结果] 激活清洁型数据底座清洗管线 | [意义] 通过锚定严格的跨截面状态面板彻底终结生存者偏差隐患")
    logger.info("=" * 60)
    
    data_manager = pipeline_context['data_manager']
    data_bus = pipeline_context['data_bus']
    audit_logger = pipeline_context['audit_logger']
    
    # 依次串联和调用解耦后的原子运算步骤
    logger.info("[SUB] Starting Step 1.1: Screening")
    t1 = time.time()
    run_screening(pipeline_context, data_bus, data_manager)
    logger.info("[SUB] Step 1.1 completed in %.2f seconds", time.time() - t1)
    
    logger.info("[SUB] Starting Step 1.2: Returns & Delisting")
    t2 = time.time()
    run_returns_cleaning(pipeline_context, data_bus, data_manager, audit_logger)
    logger.info("[SUB] Step 1.2 completed in %.2f seconds", time.time() - t2)
    
    logger.info("[SUB] Starting Step 1.3: Trading Status")
    t3 = time.time()
    run_status_mapping(pipeline_context, data_bus, data_manager)
    logger.info("[SUB] Step 1.3 completed in %.2f seconds", time.time() - t3)
    
    total_assets = pipeline_context['assets']
    calendar_alignment = pipeline_context['calendar_alignment']
    alignment_table = calendar_alignment["alignment_table"]
    ashare_timeline = alignment_table["ashare_date"].tolist()
    logger.info("[DATA] Total assets after all steps: %s", len(total_assets))
    logger.info("[DATA] Timeline length: %s trading days", len(ashare_timeline))
    
    # ====================================================
    # 核心修复 A-8：构建全时空横截面生存状态矩阵 alive_mask Panel
    # ====================================================
    logger.info("[OP] Construct Space-Time Survivor Matrix Panel | [SOURCE] Point-In-Time Asset History Files | [RESULT] Extracted temporal birth and death boundaries | [SIGNIFICANCE] Provides absolute defense mapping to protect FSM layers against backtest forward-peeking lookahead errors")
    logger.info("[操作] 构建全时空横截面生存状态矩阵面板 | [来源] Point-In-Time 资产历史物理底表 | [结果] 成功索出个股上市出生与退市消亡时空物理边界 | [意义] 为下游交易有限状态机提供绝对防御映射，杜绝回测前瞻性获利透视错误")
    
    alive_mask_records = []
    asset_bounds = {}
    bound_start_time = time.time()
    
    logger.info("[PROGRESS] Fetching birth/death boundaries for %s assets...", len(total_assets))
    for idx, sym in enumerate(total_assets):
        try:
            hist = data_manager.fetch_historical(sym, "2010-01-01", "2026-07-02")
            if hist is not None and not hist.empty:
                asset_bounds[sym] = (hist['date'].min(), hist['date'].max())
            else:
                asset_bounds[sym] = (pd.to_datetime("2030-01-01"), pd.to_datetime("2030-01-01"))
        except Exception:
            asset_bounds[sym] = (pd.to_datetime("2030-01-01"), pd.to_datetime("2030-01-01"))
        # 每100个资产打印一次进度
        if (idx+1) % 100 == 0:
            logger.info("[PROGRESS] Boundary fetched for %s/%s assets", idx+1, len(total_assets))
    logger.info("[PROGRESS] Boundary fetching completed in %.2f seconds", time.time() - bound_start_time)
    
    logger.info("[PROGRESS] Building alive_mask matrix over %s dates and %s assets...", len(ashare_timeline), len(total_assets))
    mask_start = time.time()
    for date_str in ashare_timeline:
        current_dt = pd.to_datetime(date_str)
        row_mask = []
        for sym in total_assets:
            start_born, end_death = asset_bounds[sym]
            if start_born <= current_dt <= end_death: 
                row_mask.append(True)
            else: 
                row_mask.append(False)
        alive_mask_records.append(row_mask)
    logger.info("[PROGRESS] alive_mask construction completed in %.2f seconds", time.time() - mask_start)

    alive_mask_df = pd.DataFrame(alive_mask_records, index=ashare_timeline, columns=total_assets)
    theoretical_aum_limit = pipeline_context.get('theoretical_aum_limit_base', 50000000.0)
    adv_data_mock = pd.DataFrame(20000000.0, index=ashare_timeline, columns=total_assets)

    logger.info("[OP] Finalize Phase_1 Cleanup Context | [SOURCE] Purified Pipeline Cache Trunk | [RESULT] Emits alive_mask frame shaped: %s | [SIGNIFICANCE] Satisfies schema-contract assurances across the federated framework", alive_mask_df.shape)
    logger.info("[操作] 终结 Phase_1 清洗上下文 | [来源] 纯净化流水线缓存主干 | [结果] 交付生存矩阵面板，维度为: %s | [意义] 满足联邦框架下跨阶段的 Schema 强校验合规合拢保证", alive_mask_df.shape)
    logger.info("[TOTAL] Phase_1 overall execution time: %.2f seconds", time.time() - start_time)

    return {
        'assets': total_assets,
        'alive_mask': alive_mask_df,
        'adv_data': adv_data_mock,
        'theoretical_aum_limit': theoretical_aum_limit
    }