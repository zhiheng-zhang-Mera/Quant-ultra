"""
Quant-Ultra + Conformal-BL Investment Workflow Engine
Phase 1: Asset Screening and Basic Data Cleaning (Data Foundation Orchestrator)
"""
import logging

# 导入打散后的原子原子业务子组件
from step1.step1_1_screening import run_screening
from step1.step1_2_returns import run_returns_cleaning
from step1.step1_3_trading_status import run_status_mapping

logger = logging.getLogger("Orchestrator.Step1")

def execute(pipeline_context: dict) -> dict:
    """
    Step1 标准管道入口
    """
    logger.info("=" * 60)
    logger.info("📡 QUANT-ULTRA PIPELINE -> STARTING STEP 1 (DATA FOUNDATION)")
    logger.info("=" * 60)
    
    # 提取 Main 注入的基础数据总线与组件
    data_bus = pipeline_context['data_bus']
    data_manager = pipeline_context['data_manager']
    audit_logger = pipeline_context['audit_logger']

    # 1. 执行资产初筛与容量评测
    run_screening(pipeline_context, data_bus, data_manager)

    # 2. 执行全收益价格流增量对齐清洗
    run_returns_cleaning(pipeline_context, data_bus, data_manager, audit_logger)

    # 3. 执行涨跌停边界控制安全字典映射
    run_status_mapping(pipeline_context, data_bus, data_manager)

    # 4. 交付下游验证标记
    pipeline_context['data_foundation_ready'] = True
    logger.info(f"✅ Step 1 数据基建层构建完毕，清洗出安全可交易核心池: {len(pipeline_context.get('assets', []))} 只股票")
    
    return pipeline_context