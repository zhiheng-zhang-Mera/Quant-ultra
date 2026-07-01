"""
Phase 8 Pipeline Driver - Fully Decoupled Entry
Fully compliant with Final-Flow.md [2026 Production Release]
"""

import logging
from .dsr_audit import run_dsr_audit
from .coverage_test import run_christoffersen_test
from .stress_test import run_stress_test
from .capacity_audit import run_capacity_audit

logger = logging.getLogger("AuditStressTest")

def execute(pipeline_context: dict) -> dict:
    """
    Execute Phase 8 Atomic Sub-pipeline.
    Receives global context bus, extracts read-only shadows internally, 
    and returns aggregated audit results safely.
    """
    logger.info("=" * 60)
    logger.info("📡 [PHASE 8] 启动 DSR递减夏普 / CHRISTOFFERSEN条件覆盖 / 极端压力 / 扩容审计")
    logger.info("=" * 60)

    # 依序调度原子物理核心，内部包含影子副本防火墙，严禁篡改主数据源
    run_dsr_audit(pipeline_context)
    run_christoffersen_test(pipeline_context)
    run_stress_test(pipeline_context)
    run_capacity_audit(pipeline_context)

    # 判定全局宏观合规红绿灯状态
    dsr_ok = pipeline_context.get("dsr_pass", False)
    christoffersen_ok = pipeline_context.get("christoffersen_pass", False)
    overall_pass = bool(dsr_ok and christoffersen_ok)
    
    pipeline_context["audit_passed"] = overall_pass

    if overall_pass:
        logger.info("✅ 阶段审计通过 [AUDIT PASSED]。多重共线试验无前瞻过拟合，VaR 体制完全独立。策略达到生产准入级，可平滑滑入 Live MLOps 实盘总线。")
    else:
        logger.error("❌ 阶段审计未通过 [AUDIT FAILED]。DSR 衰减不合规或违约聚集度过高。策略具备高度回测欺骗性，禁止投产，请立刻拦截回溯！")

    # 收拢纯写出格式的审计快照字典
    summary = {
        "nominal_sharpe": pipeline_context.get("nominal_sharpe"),
        "dsr_pval": pipeline_context.get("dsr_pval"),
        "empirical_coverage": pipeline_context.get("empirical_coverage"),
        "christoffersen_pvals": pipeline_context.get("christoffersen_regime_pvals"),
        "stress_drawdowns": pipeline_context.get("stress_drawdowns"),
        "aum_capacity_multiplier": pipeline_context.get("aum_capacity_multiplier"),
        "audit_passed": overall_pass,
    }
    pipeline_context["audit_summary"] = summary
    
    return pipeline_context