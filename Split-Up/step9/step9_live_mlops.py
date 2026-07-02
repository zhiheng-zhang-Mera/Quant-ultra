"""
Quant-Ultra Flow - Step 9: Dual-Track MLOps Framework and Production Life Cycle Management
Fully compliant with Flow-Pro.md and README.md persistence cache constraints.
"""
import os
import logging
import pandas as pd
from datetime import datetime
from .config import CACHE_PARQUET_DIR, CACHE_FEATHER_DIR
from .shadow_recon import run_shadow_reconciliation
from .tiered_updater import evaluate_distribution_drift
from .telemetry_alerts import process_nested_risk_telemetry

logger = logging.getLogger("MLOps.MainExecutor")

def check_session_cache(current_date_str: str) -> tuple:
    """
    检查指定日期在本地磁盘中是否存在双格式（.parquet / .feather）中间状态文件。
    """
    parquet_file = CACHE_PARQUET_DIR / f"step9_result_{current_date_str}.parquet"
    feather_file = CACHE_FEATHER_DIR / f"step9_result_{current_date_str}.feather"
    
    if parquet_file.exists() and feather_file.exists():
        try:
            # 采用高保真 Parquet 格式进行 Python 内部会话消费与读取
            df_cache = pd.read_parquet(parquet_file)
            cached_context = df_cache.to_dict(orient='records')[0]
            logger.info(f"💾 [会话缓存命中] 成功检测到 Step 9 在 {current_date_str} 的完备落盘缓存。一键载入，准备跳过。")
            return True, cached_context
        except Exception as e:
            logger.warning(f"读取或解析本地中间缓存发生异常: {str(e)}。强行回归正常推演计算。")
            return False, {}
    return False, {}

def save_session_cache(current_date_str: str, context: dict):
    """
    完成正常处理流程后，将核心指标同时固化为 .parquet (Python消费) 与 .feather (供未来C语言外部调用) 双格式。
    """
    parquet_file = CACHE_PARQUET_DIR / f"step9_result_{current_date_str}.parquet"
    feather_file = CACHE_FEATHER_DIR / f"step9_result_{current_date_str}.feather"
    
    try:
        # 将点状标量与标志位提炼封装为单行特征 DataFrame
        summary_payload = {
            "execution_timestamp": [datetime.now().isoformat()],
            "current_date": [current_date_str],
            "reconciliation_mae": [context.get('reconciliation_mae', 0.0)],
            "recon_passed": [int(context.get('recon_passed', True))],
            "current_mean_psi": [context.get('current_mean_psi', 0.0)],
            "psi_consecutive_breaches": [context.get('psi_consecutive_breaches', 0)],
            "alpha_new_model": [context.get('alpha_new_model', 0.0)],
            "condition_a_active": [int(context.get('condition_a_active', False))],
            "condition_b_active": [int(context.get('condition_b_active', False))],
            "enforce_crowded_allocation_cap": [int(context.get('enforce_crowded_allocation_cap', False))]
        }
        df_to_persist = pd.DataFrame(summary_payload)
        
        # 严格执行双格式原子化落盘固化体系
        df_to_persist.to_parquet(parquet_file, index=False)
        df_to_persist.to_feather(feather_file)
        logger.info(f"💾 [会话缓存固化完成] 成功写入本地双格式存储。日期节点: {current_date_str}")
    except Exception as e:
        logger.error(f"双格式会话落盘持久化生命周期管理执行失败: {str(e)}")

def execute(pipeline_context: dict) -> dict:
    """
    阶段九 MLOps 顶层调度核心入口。
    顺序激活影子对账、平稳度漂移、拥挤泡沫审计，并完美闭环生命周期缓存。
    """
    logger.info("==========================================================================")
    logger.info("初始化阶段九：双轨分布式 MLOps 最终冻结、影子对账与生产看门狗应急机制")
    logger.info("==========================================================================")
    
    current_date = pipeline_context.get('current_date')
    if isinstance(current_date, datetime):
        current_date_str = current_date.strftime("%Y-%m-%d")
    elif isinstance(current_date, str):
        current_date_str = current_date
    else:
        current_date_str = datetime.now().strftime("%Y-%m-%d")

    # 刚性响应 README 约定的缓存挂起逻辑核验
    update_required = pipeline_context.get('update_required', True)
    if not update_required:
        cache_hit, cached_data = check_session_cache(current_date_str)
        if cache_hit:
            pipeline_context.update(cached_data)
            pipeline_context['step9_skipped_by_cache'] = True
            logger.info("根据项目生命周期控制律，已通过本地高效缓存直接复用跳过阶段九。")
            return pipeline_context

    # Step 9.1：运行确定性影子对账协议与多资产 LLM 报文冻结
    pipeline_context = run_shadow_reconciliation(pipeline_context)
    
    # Step 9.2：运行阶梯式运维平稳度 PSI 审计与新老模型混合步进混配
    pipeline_context = evaluate_distribution_drift(pipeline_context)
    
    # Step 9.3：运行多层嵌套哨兵预警，实时审查低波及拥挤度变动
    pipeline_context = process_nested_risk_telemetry(pipeline_context)
    
    # 正常推演结束后，自动导出双格式持久化会话中间状态
    save_session_cache(current_date_str, pipeline_context)
    
    pipeline_context['mlops_ready'] = True
    pipeline_context['step9_completed'] = True
    
    logger.info("阶段九全套分布式生产级运维及风险看门狗校验任务圆满闭环。")
    return pipeline_context