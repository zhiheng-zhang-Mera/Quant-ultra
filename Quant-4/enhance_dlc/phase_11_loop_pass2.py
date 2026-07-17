# Phase_11_pass02.py
import logging
from Phase_6.bl_fusion import PosteriorFusion
from Phase_6.convex_optimizer import ConvexOptimizer
from Phase_7.execution_fsm import ExecutionFSM
from Phase_9.shadow_reconciliation import ShadowLedger

def execute_pass02_loop(pass1_baseline, llm_views, max_retries=3):
    logger = logging.getLogger("Pass02_Orchestrator")
    fallback_allocation = pass1_baseline['weights']

    for attempt in range(max_retries):
        try:
            # 融合LLM视图生成后验参数
            posterior_params = PosteriorFusion.fuse(
                prior=pass1_baseline['priors'],
                views=llm_views
            )

            # 调用严格凸优化求解
            adjusted_weights = ConvexOptimizer.optimize(posterior_params)

            # 执行FSM物理沙盒验证
            fsm_results = ExecutionFSM.run_sandbox(adjusted_weights)

            # 处理影子账本发散度审计
            divergence = ShadowLedger.calculate_divergence(
                model_weights=adjusted_weights,
                shadow_weights=fsm_results['shadow_weights']
            )

            if divergence > ShadowLedger.SAFETY_THRESHOLD:
                raise Exception("LedgerDivergenceException")

            logger.info("Pass_2_Execution_Successful")
            return adjusted_weights

        except Exception as e:
            logger.warning(f"Penalty_Triggered_Attempt:_{attempt + 1}")

    logger.error("Max_Failures_Reached_System_Fallback")
    return fallback_allocation