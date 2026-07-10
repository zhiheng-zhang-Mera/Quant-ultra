# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 8.4: Multi-Scenario Historical Crisis Stress Tester
"""
import logging
import pandas as pd
from Phase_8.config import DEFAULT_CONFIG
from Phase_8.utils import safe_get_shadow, compute_max_drawdown

logger = logging.getLogger("AuditStressTest.Stress")

def run_stress_test(context: dict) -> None:
    """
    危机体制极限压力测试算子。
    穿透预设的重大危机时空窗口（2015流动性异常、2016熔断、2024微盘股踩踏），评估最大回撤弹性。
    """
    # logger.info("[OP] Deploy Stress Test Engine | [SOURCE] Predefined Systematic Windows | [RESULT] Activating scenario parser | [SIGNIFICANCE] Verifies strategy survivability inside historically true market collapses")
    logger.info("[操作] 部署极端场景压力测试引擎 | [来源] 预设的历史系统性危机时空窗口 | [结果] 正在激活场景解析算子 | [意义] 检验策略在历史真实极值崩溃下的物理存活能力，拒绝幸存者偏差假象")
    
    config = context.get('config', {})
    stress_scenarios = config.get('stress_scenarios', DEFAULT_CONFIG['stress_scenarios'])

    try:
        nav_series = safe_get_shadow(context, "daily_nav")
        if not isinstance(nav_series, pd.Series):
            raise TypeError("daily_nav must be a DatetimeIndexed pandas Series.")

        stress_drawdowns = {}
        for scenario, (start_str, end_str) in stress_scenarios.items():
            start = pd.Timestamp(start_str)
            end = pd.Timestamp(end_str)
            
            # 🛡️ 刚性时区对齐自愈栅栏：净化比对，消灭 naive 与 aware 时间轴比较死锁
            if nav_series.index.tz is not None:
                if start.tz is None: start = start.tz_localize(nav_series.index.tz)
                else: start = start.tz_convert(nav_series.index.tz)
                if end.tz is None: end = end.tz_localize(nav_series.index.tz)
                else: end = end.tz_convert(nav_series.index.tz)
            else:
                if start.tz is not None: start = start.tz_localize(None)
                if end.tz is not None: end = end.tz_localize(None)

            # 严格截取危机影子时间跨度
            window_nav = nav_series.loc[start:end]
            
            if len(window_nav) < 2:
                # logger.warning("[OP] Slice Crisis Window | [SOURCE] Scenario Calendar: %s | [RESULT] Vacuum Period (0 or 1 row) | [SIGNIFICANCE] No overlapping dates in backtest; safely bypasses", scenario)
                logger.warning("[操作] 截取历史危机时间切片 | [来源] 极端场景日历轴: %s | [结果] 数据真空（零行或单行） | [意义] 当前回测窗口未覆盖该危机区间，安全跳过此项场景模拟", scenario)
                dd = float('nan')
            else:
                dd = compute_max_drawdown(window_nav)
                # logger.info("[OP] Compute Crisis Max Drawdown | [SOURCE] Sliced Period: %s | [RESULT] Drawdown: %.4f | [SIGNIFICANCE] Quantifies historical tail damage", scenario, dd)
                logger.info("[操作] 求解危机最大回撤 | [来源] 截取后的危机高波波时段: %s | [结果] 期间最大回撤: %.4f | [意义] 量化历史真实系统性尾部崩塌对本策略净值的最大压榨杀伤深度", scenario, dd)

            stress_drawdowns[scenario] = dd

        context["stress_drawdowns"] = stress_drawdowns

    except Exception as e:
        # logger.error("[OP] Run Stress Simulator | [SOURCE] Crisis Slicer Pipeline | [RESULT] Failure: %s | [SIGNIFICANCE] Keeps default empty drawdown registry to alert master audit", str(e))
        logger.error("[操作] 运行极端高压模拟器 | [来源] 危机时空切轨管线 | [结果] 算法失败: %s | [意义] 保留默认空回撤字典注册表，向上游主控制器警示潜在的数据断层风险", str(e))
        context["stress_drawdowns"] = {scen: float('nan') for scen in stress_scenarios.keys()}