# -*- coding: utf-8 -*-
from .step7_fsm_backtest import FSMEngine, logger

def execute(pipeline_context: dict) -> dict:
    """
    阶段 7 统一原子规范接口外曝
    """
    logger.info("=" * 60)
    logger.info("🚀 启动 Phase 7: 确定性 FSM 状态机引擎与多空资产负债回测流水线")
    logger.info("=" * 60)
    
    engine = FSMEngine(pipeline_context)
    engine.run_engine_pipeline()
    
    pipeline_context['fsm_engine'] = engine
    final_nav = pipeline_context.get('final_nav', 0.0)
    
    logger.info(f"✅ Phase 7 顺利通关。回测流清算收尾，资产包最终总净值: {final_nav:.2f}")
    logger.info("=" * 60)
    
    return pipeline_context