import logging
import pandas as pd
from .config import DEFAULT_CONFIG
from .utils import safe_get_shadow, compute_max_drawdown

logger = logging.getLogger("AuditStressTest.Stress")

def run_stress_test(context: dict) -> None:
    """Stress test on predefined historical crisis windows (e.g. 2015 Liquidity, 2024 Microcap Melt)."""
    logger.info("[Step 8.4] 开启历史极端体制崩溃窗口巨震压力测试.")
    
    config = context.get('config', {})
    stress_scenarios = config.get('stress_scenarios', DEFAULT_CONFIG['stress_scenarios'])

    try:
        nav_series = safe_get_shadow(context, "daily_nav")
        if not isinstance(nav_series, pd.Series):
            raise TypeError("daily_nav 必须为具有 DatetimeIndex 的 pandas.Series")

        stress_drawdowns = {}
        for scenario, (start_str, end_str) in stress_scenarios.items():
            start = pd.Timestamp(start_str)
            end = pd.Timestamp(end_str)
            
            # 严格提取影子闭合时间截面，不改变任何外围日历结构
            window_nav = nav_series.loc[start:end]
            
            if len(window_nav) < 2:
                logger.warning(f"⚠️ 压力测试窗口 [{scenario}] ({start_str} 至 {end_str}) 在回测周期内查无足够行情数据，自动剔除。")
                dd = float('nan')
            else:
                dd = compute_max_drawdown(window_nav)
                logger.info(f"🔥 危机场景 [{scenario}] 压榨性最大回撤测试测得: {dd*100:.2f}%")
            
            stress_drawdowns[scenario] = dd

        context["stress_drawdowns"] = stress_drawdowns

    except Exception as e:
        logger.error(f"❌ [压力测试流中断] 原因: {str(e)}", exc_info=True)
        context["stress_drawdowns"] = {}