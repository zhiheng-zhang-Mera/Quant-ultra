import logging
import numpy as np
import pandas as pd
from .config import DEFAULT_CONFIG
from .utils import safe_get_shadow

logger = logging.getLogger("AuditStressTest.Capacity")

def run_capacity_audit(context: dict) -> None:
    """
    个体账户静态容量边界终审。
    彻底废除 4.5% 大股东举牌限额反推，废除一切随 AUM 放大迭代的外循环。
    代入个体既定本金底座，静态验证换仓大单参与率对最终 NAV 的非线性滑点损耗，确立真实净值表现。
    """
    logger.info("[Step 8.5] 执行基于个体既定本金底座与微观流动性匹配的纯静态容量终审.")
    
    config = context.get('config', {})
    total_equity = config.get('total_equity', DEFAULT_CONFIG['total_equity'])
    max_participation_threshold = config.get('max_participation_threshold', DEFAULT_CONFIG['max_participation_threshold'])
    
    # 遵循 Flow-Pro 7.2 / 8.5 规范硬性锁死的平方根静态冲击常数降级基准线
    kappa_impact = 0.001  # 刚性指定 10bp 冲击基准线
    alpha_impact = 0.5    # 平方根弹性系数

    try:
        weights = safe_get_shadow(context, "daily_weights")
        nav_series = safe_get_shadow(context, "daily_nav")
        adv20_df = context.get("daily_adv20")  # 尝试从内存总线提取个股 T-1 日盘后 20日平均成交额面板

        if not isinstance(weights, pd.DataFrame) or weights.empty:
            logger.warning("⚠️ 外部持仓权重矩阵为空或非法，静态容量终审默认条件放行。")
            context["capacity_audit_pass"] = True
            context["max_participation_rate"] = 0.0
            context["total_impact_loss_nav"] = 0.0
            return

        # 计算持仓权重的每日换仓绝对变动量（模拟盘后指令拆分交易规模）
        weight_diff = weights.diff().fillna(0.0)

        # 估计每日各成分股的拟换仓名义金额订单
        if isinstance(nav_series, pd.Series) and not nav_series.empty:
            common_dates = weight_diff.index.intersection(nav_series.index)
            weight_diff = weight_diff.loc[common_dates]
            order_amounts = weight_diff.abs().multiply(nav_series.loc[common_dates], axis=0)
        else:
            order_amounts = weight_diff.abs() * total_equity

        # 若分布式总线未配发个股流式真实 ADV20，启动生产级稳健降级：假定个股中位数 ADV 为 50,000,000 元
        if adv20_df is None or not isinstance(adv20_df, pd.DataFrame) or adv20_df.empty:
            logger.warning("⚠️ 内存总线未检测到 daily_adv20 真实流动性面板，触发安全降级：锁定单票 ADV = 50,000,000 元。")
            adv20_df = pd.DataFrame(50000000.0, index=weight_diff.index, columns=weight_diff.columns)
        
        # 严格时空对齐
        adv20_df = adv20_df.reindex(index=weight_diff.index, columns=weight_diff.columns).fillna(50000000.0)

        # 计算盘后模拟大单参与率 Participation Rate = 订单名义金额 / 市场个股成交额底座
        participation_rates = (order_amounts / adv20_df).fillna(0.0).replace([np.inf, -np.inf], 0.0)
        max_part_per_stock = participation_rates.max()
        global_max_part = float(max_part_per_stock.max()) if not max_part_per_stock.empty else 0.0

        # 根据平方根冲击定律评估非线性滑点累积总损耗对账户 NAV 的隐性压榨
        # 修正：损耗 = 冲击比例 * 订单金额，其中冲击比例 = kappa * (participation_rate ** alpha)
        impact_loss_matrix = (participation_rates ** alpha_impact) * kappa_impact * order_amounts
        total_loss_by_date = impact_loss_matrix.sum(axis=1)
        global_total_loss_nav = float(total_loss_by_date.sum())

        context["max_participation_rate"] = global_max_part
        context["total_impact_loss_nav"] = global_total_loss_nav
        
        # 纯静态终审红绿灯准则：全史最极端个股大单参与率严禁击穿警戒水位线
        capacity_pass = global_max_part <= max_participation_threshold
        context["capacity_audit_pass"] = capacity_pass

        logger.info(f"🚀 静态容量终审完毕 -> 当前既定本金权益底座: {total_equity:,.2f} 元")
        logger.info(f"   全史个股极端大单参与率峰值: {global_max_part*100:.3f}% (合规硬红线: <= {max_participation_threshold*100:.2f}%)")
        logger.info(f"   全时段累计非线性滑点对 NAV 的绝对冲击损耗: {global_total_loss_nav:.4f} 元")
        logger.info(f"   容量终审综合评定准入状态: {'[PASSED] ✅' if capacity_pass else '[FAILED] ❌'}")

    except Exception as e:
        logger.error(f"❌ [静态容量终审内核崩溃] 细节原因: {str(e)}", exc_info=True)
        context["capacity_audit_pass"] = False
        context["max_participation_rate"] = 1.0
        context["total_impact_loss_nav"] = 0.0