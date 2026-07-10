"""
Phase 8 Pipeline Driver - Fully Decoupled & High-Cohesion Entry
Fully compliant with Flow-Pro.md and README.md [2026 纯计算内核版]
"""

import logging
from .dsr_audit import run_dsr_audit
from .coverage_test import run_christoffersen_test
from .stress_test import run_stress_test
from .capacity_audit import run_capacity_audit

logger = logging.getLogger("AuditStressTest")

def execute(pipeline_context: dict) -> dict:
    """
    执行阶段八原子子管线核心驱动。
    串联各分项审计算子，挂载一票否决合规硬红线，剔除任何本地手写读写盘。
    """
    logger.info("=" * 60)
    logger.info("📡 [PHASE 8] 启动综合多维风控联合终审总线 (纯内存运算)")
    logger.info("=" * 60)

    # 1. 依序调度物理核心审计算子
    run_dsr_audit(pipeline_context)
    run_christoffersen_test(pipeline_context)
    run_stress_test(pipeline_context)
    run_capacity_audit(pipeline_context)

    # 2. 提取分项合规判定，布设宏观一票否决安全红绿灯
    dsr_ok = pipeline_context.get("dsr_pass", False)
    christoffersen_ok = pipeline_context.get("christoffersen_pass", False)
    capacity_ok = pipeline_context.get("capacity_audit_pass", False)
    
    overall_pass = bool(dsr_ok and christoffersen_ok and capacity_ok)
    pipeline_context["audit_passed"] = overall_pass

    if overall_pass:
        logger.info("✅ 阶段八综合风控终审通过 [AUDIT PASSED]！策略准入，可移交 Live MLOps 实盘总线。")
    else:
        logger.error("❌ 阶段八综合风控终审被拦截 [AUDIT FAILED]！严禁投产！")

    # 3. 收拢审计成果，向总线打包标准快照字典
    summary = {
        "nominal_sharpe": pipeline_context.get("nominal_sharpe"),
        "dsr_pval": pipeline_context.get("dsr_pval"),
        "empirical_coverage": pipeline_context.get("empirical_coverage"),
        "christoffersen_pvals": pipeline_context.get("christoffersen_regime_pvals"),
        "stress_drawdowns": pipeline_context.get("stress_drawdowns"),
        "max_participation_rate": pipeline_context.get("max_participation_rate"),
        "total_impact_loss_nav": pipeline_context.get("total_impact_loss_nav"),
        "audit_passed": overall_pass,
    }
    
    # 💡 架构演进：物理割除了 flat_snapshot 构造和极其臃肿的 to_parquet/to_feather 手写逻辑！
    
    # 4. 组装仅需更新并持久化的资产字典，全量交由 main.py 实现规范的中心化双格式存盘
    result_update = {
        "audit_passed": overall_pass,
        "audit_summary": summary,
        "phase8_completed": True
    }

    return result_update