# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 8.5: Static Account Capacity & Non-Linear Market Impact Auditor
"""
import logging
import numpy as np
import pandas as pd
from Phase_8.config import DEFAULT_CONFIG
from Phase_8.utils import safe_get_shadow

logger = logging.getLogger("AuditStressTest.Capacity")

def run_capacity_audit(context: dict) -> None:
    """
    个体账户静态容量边界终审算子。
    摒弃两融或大股东举牌条款。代入既定 1000 万本金，在平方根冲击模型下测算极端换手大单参与率。
    """
    logger.info("[OP] Deploy Account Capacity Auditor | [SOURCE] Personal Wealth Baseline | [RESULT] Initializing static impact evaluation | [SIGNIFICANCE] Replaces dynamic looping to guarantee deterministic execution analysis")
    logger.info("[操作] 部署个人账户容量终审器 | [来源] 个体本金资产底座 | [结果] 正在初始化静态市场冲击评估 | [意义] 废除动态资产缩放外循环，确保大单参与率及滑点评估具备确定性的工程分析结果")
    
    config = context.get('config', {})
    total_equity = config.get('total_equity', DEFAULT_CONFIG['total_equity'])
    max_participation_threshold = config.get('max_participation_threshold', DEFAULT_CONFIG['max_participation_threshold'])
    
    # 刚性对齐 Phase_6/Phase_7 指定的平方根静态冲击常数降级基准
    kappa_impact = 0.001  # 10bp 冲击基准
    alpha_impact = 0.5    # 平方根弹性指数

    try:
        weights = safe_get_shadow(context, "daily_weights")
        nav_series = safe_get_shadow(context, "daily_nav")
        adv20_df = context.get("daily_adv20")

        if adv20_df is None or adv20_df.empty:
            # 灾备自愈：无可用 ADV 表时，自动初始化对齐 2000 万均值底座，阻断除零异常
            assets = context.get('assets', list(weights.columns)) if weights is not None else []
            adv20_df = pd.DataFrame(20000000.0, index=weights.index, columns=assets)
            logger.warning("[OP] Probe ADV Matrix | [SOURCE] Data Bus Cache Router | [RESULT] Absent! Switched to 20M defensive fallback pool | [SIGNIFICANCE] Guards math kernels against zero divisor traps")
            logger.warning("[操作] 探测日均成交额矩阵 | [来源] 数据总线缓存路由 | [结果] 未命中！安全切往2000万均值降级灾备池 | [意义] 防范截面个股流动性空置导致的除零溢出致命陷阱")

        # 1. 求解每日个股理论订单规模 (CNY)
        daily_diff = weights.diff().fillna(weights)
        order_amounts = daily_diff.abs() * total_equity

        # 2. 计算每日个股换手大单占市场 20D ADV 的参与率 (Participation Rate)
        participation_rates = (order_amounts / adv20_df).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        max_part_per_stock = participation_rates.max()
        global_max_part = float(max_part_per_stock.max()) if not max_part_per_stock.empty else 0.0

        # 3. 统计学平方根冲击模型：滑点损耗 = 订单金额 * [kappa * (participation_rate ** alpha)]
        impact_loss_matrix = (participation_rates ** alpha_impact) * kappa_impact * order_amounts
        total_loss_by_date = impact_loss_matrix.sum(axis=1)
        global_total_loss_nav = float(total_loss_by_date.sum())

        context["max_participation_rate"] = global_max_part
        context["total_impact_loss_nav"] = global_total_loss_nav
        
        # 4. 一票否决硬红线：大单参与率峰值必须 <= 预设红线阈值 (默认5%)
        capacity_pass = global_max_part <= max_participation_threshold
        context["capacity_audit_pass"] = capacity_pass

        logger.info("[OP] Finalize Capacity Conformal Audit | [SOURCE] Square-root Slippage Ledger | [RESULT] Base Equity: %.2f, Max Part: %.4f%%, Total Slippage Drag: %.2f CNY, Pass: %s | [SIGNIFICANCE] Guarantees strategy execution validity inside real market spreads", total_equity, global_max_part*100, global_total_loss_nav, capacity_pass)
        logger.info("[操作] 终结容量共形审计 | [来源] 平方根滑点损耗账本 | [结果] 账户本金: %.2f, 峰值参与率: %.4f%%, 累计非线性滑点总耗: %.2f 元, 审查通过: %s | [意义] 证明大额资金换仓订单在真实交易环境下的承载可行性，拒接高频滑点磨损导致的策略隐性亏损")

    except Exception as e:
        logger.error("[OP] Analyze Portfolio Capacity Bounds | [SOURCE] Liquidity Estimator Kernel | [RESULT] Failure: %s | [SIGNIFICANCE] Forced capacity rejection to enforce extreme defense rules", str(e))
        logger.error("[操作] 分析组合容量边界 | [来源] 流动性估值内核 | [结果] 算法失败: %s | [意义] 强制判为容量不通过，启用最高级一票否决风控拒接本策略上线")
        context["capacity_audit_pass"] = False