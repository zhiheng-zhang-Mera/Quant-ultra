# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 9.3: Telemetry Dashboard & Pre-emptive Nested System Alarms
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from Phase_9.config import DEFAULT_MLOPS_CONFIG

logger = logging.getLogger("MLOps.TelemetryAlerts")

def process_nested_risk_telemetry(context: dict) -> dict:
    """
    因子拥挤度度量与低波泡沫分层嵌套主动遥测预警。
    """
    logger.info("[OP] Inspect Systematic Risk Telemetry | [SOURCE] Live Portfolio Performance stream | [RESULT] Assessing nesting alarms | [SIGNIFICANCE] Dynamically detects risk overlaps to preempt liquidity flash crashes")
    logger.info("[操作] 巡检系统性风险主动遥测 | [来源] 组合实盘运行净值流 | [结果] 正在评估分层嵌套预警指标 | [意义] 动态识别极端风险重叠，拦截流动性闪崩陷阱")

    data_bus = context.get('data_bus')
    nav_history = context.get('nav_history', [])
    
    if data_bus is None or not nav_history:
        logger.warning("Data bus or nav history vacant. Skipping nested risk telemetry.")
        return context

    config = context.get('config', {})
    v_window = config.get('volatility_window', DEFAULT_MLOPS_CONFIG['volatility_window'])
    compress_q = config.get('vol_compress_quantile', DEFAULT_MLOPS_CONFIG['vol_compress_quantile'])
    corr_threshold = config.get('crowded_corr_threshold', DEFAULT_MLOPS_CONFIG['crowded_corr_threshold'])
    position_cap = config.get('crowded_risk_position_cap', DEFAULT_MLOPS_CONFIG['enforce_crowded_allocation_cap'])

    condition_b_triggered = False
    
    # 1. 解析条件 B 认知波动率：彻底修复幽灵变量与海象算子错位
    nav_values = list(nav_history.values()) if isinstance(nav_history, dict) else list(nav_history)
    if len(nav_values) >= v_window:
        nav_series = pd.Series(nav_values)
        rets = np.log(nav_series / nav_series.shift(1)).dropna()
        vol_rolling = rets.rolling(v_window, min_periods=5).std().dropna()
        
        if not vol_rolling.empty:
            curr_vol = vol_rolling.iloc[-1]
            hist_vol_quantile = vol_rolling.quantile(compress_q)
            
            if curr_vol < hist_vol_quantile:
                condition_b_triggered = True
                logger.warning("[OP] Monitor Volatility Compression | [SOURCE] Sliding Return Series | [RESULT] Under limits! Vol: %.6f, Quantile Limit: %.6f | [SIGNIFICANCE] Identifies early signals of extreme bubble compression", curr_vol, hist_vol_quantile)
                logger.warning("[操作] 监控净值波动率极端压缩 | [来源] 滚动收益率时间序列 | [结果] 击穿警戒底线！当前波幅: %.6f, 分位数底线: %.6f | [意义] 识别出低波泡沫积聚特征，防止突发剧烈均值回归风险")

    # 2. 解析条件 A 风格拥挤度
    condition_a_triggered = False
    try:
        benchmark_code = data_bus.get_benchmark_code()
        current_date_str = context.get('current_date')
        if isinstance(current_date_str, datetime):
            current_date_str = current_date_str.strftime("%Y-%m-%d")
            
        bench_df = data_bus.load_asset_history(benchmark_code, "2010-01-01", current_date_str)
        if bench_df is not None and not bench_df.empty:
            bench_returns = np.log(bench_df['close'] / bench_df['close'].shift(1)).dropna()
            
            # 提取清洗后的策略收益流
            if 'rets' in locals() and not rets.empty:
                portfolio_returns = pd.Series(rets.values)
                # 统一转为时间戳的中立字符串切片对齐
                bench_returns.index = pd.to_datetime(bench_returns.index).strftime("%Y-%m-%d")
                if isinstance(nav_history, dict):
                    portfolio_returns.index = pd.to_datetime(list(nav_history.keys())[1:]).strftime("%Y-%m-%d")
                
                common_dates = portfolio_returns.index.intersection(bench_returns.index)
                if len(common_dates) >= 15:
                    corr_coeff = float(portfolio_returns.loc[common_dates].corr(bench_returns.loc[common_dates]))
                    context['benchmark_crowding_correlation'] = corr_coeff
                    
                    if corr_coeff > corr_threshold:
                        condition_a_triggered = True
                        logger.warning("[OP] Probe Multi-Factor Crowding | [SOURCE] Cross-Sectional Style Correlation | [RESULT] Coeff: %.4f, Limit Line: %.4f | [SIGNIFICANCE] Flags excessive style clustering across components", corr_coeff, corr_threshold)
                        logger.warning("[操作] 探测风格因子高度拥挤度 | [来源] 横截面风格相关系数分析 | [结果] 当前相关系数: %.4f, 顶层警报红线: %.4f | [意义] 指出因子拥挤度过载风险，严防风格抱团崩溃引发的大额踩踏损失")
    except Exception as e:
        logger.error(f"Failed to calculate crowding correlation: {e}")
        context['benchmark_crowding_correlation'] = 0.0

    context['condition_a_active'] = condition_a_triggered
    context['condition_b_active'] = condition_b_triggered

    # 3. 双重交叉重叠交汇熔断保护
    if condition_a_triggered and condition_b_triggered:
        logger.critical("[OP] Nest System Shock Triggered | [SOURCE] Telemetry Overlapping Joint-Alarm | [RESULT] Enforcing %s%% multi-equity limit cap | [SIGNIFICANCE] Forced de-allocation of spot risk capital into raw cash storage to survive systemic drawdowns", position_cap * 100)
        logger.critical("[操作] 风格拥挤与低波泡沫嵌套风险同时激发 | [来源] 遥测双重重叠交汇大报警 | [结果] 强行限制多头总仓位上限至 %s%% | [意义] 实盘顶层终极物理防御防御机制，将高资产强行置换为无风险纯现金持币，绝对阻断高泡沫期的信用过载", position_cap * 100)
        context['enforce_crowded_allocation_cap'] = position_cap
    else:
        context['enforce_crowded_allocation_cap'] = 1.0 
        
    return context