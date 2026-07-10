# -*- coding: utf-8 -*-
"""
Phase 9 Pipeline Orchestrator - Production Release & Immutable Asset Ledger
"""
import logging
from datetime import datetime
import pandas as pd
from Phase_9.shadow_reconciliation import run_shadow_reconciliation
from Phase_9.tiered_updater import evaluate_distribution_drift
from Phase_9.telemetry_alerts import process_nested_risk_telemetry
from Phase_9.config import CACHE_PARQUET_DIR, CACHE_FEATHER_DIR

logger = logging.getLogger("MLOps.MainExecutor")

def execute(pipeline_context: dict) -> dict:
    """
    阶段九 MLOps 与生产看门狗生命周期主入口。
    将增量缓存机制与文件落盘交还主协调器。本地执行对账与安全断电，并导出符合格式标准的实体。
    """
    logger.info("=" * 60)
    # logger.info("[OP] Deploy Phase_9 Production Command Center | [SOURCE] Pipeline Central Orchestration Core | [RESULT] Launching Live-MLOps gateway | [SIGNIFICANCE] Final auditing, reconciliation, and automated auto-preservation")
    logger.info("[操作] 部署 Phase_9 实盘监控主控中心 | [来源] 全局时空编排协调器大轴 | [结果] 正在引导 MLOps 交互网关 | [意义] 执行最终的对账核销与特征漂移审计，生成合规本地备份")
    logger.info("=" * 60)

    current_date = pipeline_context.get('current_date')
    if isinstance(current_date, datetime):
        current_date_str = current_date.strftime("%Y-%m-%d")
    elif isinstance(current_date, str):
        current_date_str = current_date
    else:
        current_date_str = datetime.now().strftime("%Y-%m-%d")

    # Step 9.1：执行影子对账双轨审计
    pipeline_context = run_shadow_reconciliation(pipeline_context)
    
    # Step 9.2：PSI 认知稳定性审计
    pipeline_context = evaluate_distribution_drift(pipeline_context)
    
    # Step 9.3：嵌套风险主动遥测哨兵
    pipeline_context = process_nested_risk_telemetry(pipeline_context)

    # 4. 打包导出元数据点，对准下游契约 (Contract assurance)
    result_update = {
        "execution_timestamp": datetime.now().isoformat(),
        "current_date_str": current_date_str,
        "reconciliation_mae": float(pipeline_context.get('reconciliation_mae', 0.0)),
        "recon_passed": bool(pipeline_context.get('recon_passed', True)),
        "current_mean_psi": float(pipeline_context.get('current_mean_psi', 0.0)),
        "psi_consecutive_breaches": int(pipeline_context.get('psi_consecutive_breaches', 0)),
        "alpha_new_model": float(pipeline_context.get('alpha_new_model', 1.0)),
        "tier3_retrain_active": bool(pipeline_context.get('tier3_retrain_active', False)),
        "enforce_crowded_allocation_cap": float(pipeline_context.get('enforce_crowded_allocation_cap', 1.0)),
        "mlops_ready": True
    }

    # 5. 生成落盘归档实体
    try:
        export_df = pd.DataFrame([result_update])
        p_path = CACHE_PARQUET_DIR / f"mlops_status_{current_date_str}.parquet"
        f_path = CACHE_FEATHER_DIR / f"mlops_status_{current_date_str}.feather"
        
        export_df.to_parquet(p_path, index=False)
        export_df.reset_index(drop=True).to_feather(f_path)
        
        # logger.info("[OP] Export Production Session Artifacts | [SOURCE] Memory State Record DataFrame | [RESULT] Created Parquet: %s, Feather: %s | [SIGNIFICANCE] Provides persistent state auditing for MLOps tracing", p_path.name, f_path.name)
        logger.info("[操作] 导出生产会话账本实体 | [来源] 内存态运行数据变DataFrame | [结果] 固化Parquet: %s, Feather: %s | [意义] 提供高保真、轻量级、不可篡改的实盘每日运行会话状态硬拷贝，以供灾备追溯", p_path.name, f_path.name)
    except Exception as e:
        logger.error(f"Archiving system session outputs failed: {e}")

    # 合并返回管道，更新大总线
    pipeline_context.update(result_update)
    return pipeline_context