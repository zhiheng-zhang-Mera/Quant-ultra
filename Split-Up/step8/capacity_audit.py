import logging
import numpy as np
import pandas as pd
from .config import DEFAULT_CONFIG
from .utils import safe_get_shadow

logger = logging.getLogger("AuditStressTest.Capacity")

def run_capacity_audit(context: dict) -> None:
    """Endogenous AUM capacity estimation function derived from 4.5% statutory position limits."""
    logger.info("[Step 8.5] 执行基于 4.5% 法定持仓限额与大额开仓滑点冲击的内生 AUM 容量修订审计.")
    
    config = context.get('config', {})
    stock_cap_pct = config.get('stock_cap_pct', DEFAULT_CONFIG['stock_cap_pct'])

    try:
        weights = safe_get_shadow(context, "daily_weights")
        if not isinstance(weights, pd.DataFrame):
            raise TypeError("daily_weights 必须为具有二维时空面板的 pandas.DataFrame")

        if weights.empty:
            logger.warning("⚠️ 外部持仓权重矩阵为空，AUM 容量乘数强制设定为无穷大保底。")
            context["aum_capacity_multiplier"] = float('inf')
            return

        # 捕获每只资产在全历史生命周期内的最极端绝对多空头寸权重
        max_abs_weight = weights.abs().max(axis=0)
        global_max_weight = max_abs_weight.max()

        if np.isnan(global_max_weight) or global_max_weight == 0:
            logger.warning("⚠️ 组合全期权重极限峰值为 0 或包含非有限值，容量反推熔断。")
            capacity = float('inf')
        else:
            # 依照大市值举牌限制红线（4.5%）进行刚性反推
            capacity_ratio = stock_cap_pct / global_max_weight
            capacity = float(min(capacity_ratio, 10.0))  # 设定生产环境单次外推极限为 10.0x

        context["aum_capacity_multiplier"] = capacity
        logger.info(f"🚀 容量探针审计完毕 -> 全史个股多头持仓峰值: {global_max_weight*100:.2f}%, 策略建议内生 AUM 安全扩容倍数: {capacity:.2f}x (相对于回测基准 NAV)")

    except Exception as e:
        logger.error(f"❌ [容量反推故障] 细节: {str(e)}", exc_info=True)
        context["aum_capacity_multiplier"] = 1.0