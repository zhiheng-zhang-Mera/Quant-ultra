"""
Phase 8 Pipeline Driver - Fully Decoupled & High-Cohesion Entry
Fully compliant with Flow-Pro.md and README.md [2026 Production Release]
"""

import os
import logging
import pandas as pd
from .dsr_audit import run_dsr_audit
from .coverage_test import run_christoffersen_test
from .stress_test import run_stress_test
from .capacity_audit import run_capacity_audit

logger = logging.getLogger("AuditStressTest")

def execute(pipeline_context: dict) -> dict:
    """
    执行阶段八原子子管线核心驱动。
    串联各分项审计算子，挂载一票否决合规硬红线，并强制执行双格式会话落盘持久化管理。
    """
    logger.info("=" * 60)
    logger.info("📡 [PHASE 8] 启动 DSR递减夏普 / CHRISTOFFERSEN条件覆盖 / 极端压力 / 纯静态容量终审总线")
    logger.info("=" * 60)

    # 1. 依序调度物理核心审计算子，内部由内存影子副本防火墙托管保护
    run_dsr_audit(pipeline_context)
    run_christoffersen_test(pipeline_context)
    run_stress_test(pipeline_context)
    run_capacity_audit(pipeline_context)

    # 2. 提取分项合规判定，布设宏观一票否决（Overall Door Guard）安全红绿灯
    dsr_ok = pipeline_context.get("dsr_pass", False)
    christoffersen_ok = pipeline_context.get("christoffersen_pass", False)
    capacity_ok = pipeline_context.get("capacity_audit_pass", False)
    
    # 三轨联合审计全部放行方可准入投产
    overall_pass = bool(dsr_ok and christoffersen_ok and capacity_ok)
    pipeline_context["audit_passed"] = overall_pass

    if overall_pass:
        logger.info("✅ 阶段八综合风控终审通过 [AUDIT PASSED]！多重共线过拟合已过滤，VaR独立，静态容量高度合规，策略可向 Live MLOps 实盘总线滑入。")
    else:
        logger.error("❌ 阶段八综合风控终审被拦截 [AUDIT FAILED]！DSR 严重衰减、违约聚集度过高或大单交易极易引发踩踏。策略具备高度欺骗性，严禁投产！")

    # 3. 收拢并打包纯写出格式的审计快照字典
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
    pipeline_context["audit_summary"] = summary

    # 4. 响应 README 挂起任务，强制实施会话缓存双格式（.parquet/.feather）高保真持久化落盘
    try:
        # 平面化高维嵌套字典以完美兼容跨语言底层存储格式要求
        flat_snapshot = {
            "nominal_sharpe": [summary["nominal_sharpe"]],
            "dsr_pval": [summary["dsr_pval"]],
            "empirical_coverage": [summary["empirical_coverage"]],
            "max_participation_rate": [summary["max_participation_rate"]],
            "total_impact_loss_nav": [summary["total_impact_loss_nav"]],
            "audit_passed": [int(overall_pass)],
            "dsr_pass": [int(dsr_ok)],
            "christoffersen_pass": [int(christoffersen_ok)],
            "capacity_audit_pass": [int(capacity_ok)]
        }
        
        # 解压条件覆盖马尔可夫体制 p-value 列表至基础列
        if summary["christoffersen_pvals"]:
            for k, v in summary["christoffersen_pvals"].items():
                flat_snapshot[f"christoffersen_pval_{k}"] = [v]
                
        # 解压极端巨震压力测试回撤至基础列
        if summary["stress_drawdowns"]:
            for k, v in summary["stress_drawdowns"].items():
                flat_snapshot[f"stress_drawdown_{k}"] = [v]

        df_save = pd.DataFrame(flat_snapshot)

        # 动态解析构建符合全局生命周期定义的多级存储物理骨架
        current_dir = os.path.dirname(os.path.abspath(__file__))
        split_up_root = os.path.dirname(current_dir)  # 定位到 Split-Up 根节点
        
        parquet_path = os.path.join(split_up_root, "Phase Result", "parquet", "Phase 8")
        feather_path = os.path.join(split_up_root, "Phase Result", "feather", "Phase 8")
        
        os.makedirs(parquet_path, exist_ok=True)
        os.makedirs(feather_path, exist_ok=True)

        # 无损执行双格式物理落盘固化
        df_save.to_parquet(os.path.join(parquet_path, "step8_summary.parquet"), index=False)
        df_save.to_feather(os.path.join(feather_path, "step8_summary.feather"))
        logger.info(f"💾 [会话缓存双格式持久化成功] 成果报告已落盘固化于：\n   - Parquet: {parquet_path}\n   - Feather: {feather_path}")
        
    except Exception as e:
        logger.error(f"❌ [会话缓存持久化故障] 细节原因: {str(e)}", exc_info=True)

    return pipeline_context