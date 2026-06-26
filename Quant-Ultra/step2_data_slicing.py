"""
Phase 2: Full-Flow Two-Tier Data Slicing and Isolation Architecture
Strictly implements: [Train-A] -> [Train-B1] -> [Train-B2] -> [Validation] -> [Test]
With Purge & Embargo triple-barrier constraints.
Now using Free Open-Source Data Sources (AkShare/BaoStock) with multi-level fallback.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from statsmodels.tsa.stattools import acf
import logging

logger = logging.getLogger("DataSlicing")


def step_2_1_moving_window_slicing(context: dict):
    """
    核心切分逻辑：考虑 Purge 与 Embargo 的硬性截断。
    规范强制 Embargo >= max(holding_period, significant_autocorrelation_lag)
    """
    logger.info("[Step 2.1] 执行多段式物理窗口切分（含 Purge & Embargo）")

    trading_days_dt = context.get('trading_days_dt', [])
    if not trading_days_dt:
        raise ValueError("交易日历 (trading_days_dt) 未在上下文总线中初始化。")

    config = context.get('config', {})
    slicing_ratios = config.get('slicing', {}).get('ratios', [0.50, 0.60, 0.70, 0.85])
    holding_period = context.get('holding_period', config.get('holding_period', 5))

    data_bus = context.get('data_bus')
    audit_logger = context.get('audit_logger')

    # ----------------------------
    # 1. 计算动态 Embargo（基于市场 ACF）
    # ----------------------------
    max_lag = 1  # 保底
    try:
        # 使用数据总线获取沪深300指数或标池平均收益估算市场记忆性
        # 取前 60% 交易日作为样本
        cutoff_idx = int(len(trading_days_dt) * 0.6)
        if cutoff_idx > 20:
            sample_days = trading_days_dt[:cutoff_idx]
            start_dt = sample_days[0].strftime("%Y-%m-%d")
            end_dt = sample_days[-1].strftime("%Y-%m-%d")

            # 修正：使用 fetch_benchmark_prices（原 query_benchmark_prices 不存在）
            benchmark_prices = data_bus.fetch_benchmark_prices(start_dt, end_dt)
            if benchmark_prices is not None and len(benchmark_prices) > 20:
                # 对齐到交易日（取交集）
                common_dates = set(benchmark_prices.index) & set(sample_days)
                if common_dates:
                    aligned = benchmark_prices.loc[sorted(common_dates)]
                    returns = np.log(aligned / aligned.shift(1)).dropna()
                    if len(returns) > 20:
                        acf_vals, confint = acf(returns, nlags=10, alpha=0.05, fft=False)
                        for i in range(1, len(acf_vals)):
                            if i < len(confint):
                                lower, upper = confint[i]
                                if acf_vals[i] < lower or acf_vals[i] > upper:
                                    max_lag = max(max_lag, i)
                        logger.info(f"ACF 显著最大滞后: {max_lag}")
            else:
                # 如果指数无数据，尝试用标池内第一只股票
                assets = context.get('assets', [])
                if assets:
                    sym = assets[0]
                    df = data_bus.load_asset_history(sym, start_dt, end_dt)
                    if df is not None and len(df) > 20:
                        returns = df['log_return'].dropna()
                        if len(returns) > 20:
                            acf_vals, confint = acf(returns, nlags=10, alpha=0.05, fft=False)
                            for i in range(1, len(acf_vals)):
                                if i < len(confint):
                                    lower, upper = confint[i]
                                    if acf_vals[i] < lower or acf_vals[i] > upper:
                                        max_lag = max(max_lag, i)
                            logger.info(f"从 {sym} 估算 ACF 显著滞后: {max_lag}")
    except Exception as e:
        logger.warning(f"ACF 动态计算失败，使用默认滞后 1。错误: {e}")
        if audit_logger:
            audit_logger.log_event("ACF_CALC_FAILED", {"error": str(e), "fallback": 1})

    # 2. 刚性确定禁运区
    embargo_window = max(holding_period, max_lag)
    context['embargo_window'] = embargo_window
    context['holding_period'] = holding_period
    logger.info(f"持有期: {holding_period}, 显著滞后: {max_lag}, 最终禁运窗口: {embargo_window}")

    # 3. 五段式切分（带 Purge/Embargo 截断）
    n = len(trading_days_dt)
    raw_a_end = int(n * slicing_ratios[0])
    raw_b1_end = int(n * slicing_ratios[1])
    raw_b2_end = int(n * slicing_ratios[2])
    raw_val_end = int(n * slicing_ratios[3])
    test_end = n

    # 应用截断：左侧截掉 embargo_window，右侧截掉 embargo_window
    # Train-A: 0 ~ raw_a_end - embargo
    a_end = max(0, raw_a_end - embargo_window)
    slices = {}

    # Train-B1: raw_a_end + embargo ~ raw_b1_end - embargo
    b1_start = min(raw_a_end + embargo_window, raw_b1_end)
    b1_end = max(0, raw_b1_end - embargo_window)
    slices["Train-A"] = trading_days_dt[0:a_end]
    slices["Train-B1"] = trading_days_dt[b1_start:b1_end] if b1_start < b1_end else []

    b2_start = min(raw_b1_end + embargo_window, raw_b2_end)
    b2_end = max(0, raw_b2_end - embargo_window)
    slices["Train-B2"] = trading_days_dt[b2_start:b2_end] if b2_start < b2_end else []

    val_start = min(raw_b2_end + embargo_window, raw_val_end)
    val_end = max(0, raw_val_end - embargo_window)
    slices["Validation"] = trading_days_dt[val_start:val_end] if val_start < val_end else []

    test_start = min(raw_val_end + embargo_window, test_end)
    slices["Test"] = trading_days_dt[test_start:test_end] if test_start < test_end else []

    context['slices'] = slices
    context['raw_split_indices'] = {
        "Train-A_end": raw_a_end,
        "Train-B1_end": raw_b1_end,
        "Train-B2_end": raw_b2_end,
        "Validation_end": raw_val_end,
        "Test_end": test_end
    }

    logger.info("切分结果:")
    for k, v in slices.items():
        start_str = v[0].strftime("%Y-%m-%d") if v else "N/A"
        end_str = v[-1].strftime("%Y-%m-%d") if v else "N/A"
        logger.info(f"   - {k}: {len(v)} 个交易日 ({start_str} ~ {end_str})")

    # 如果任一切片过短，记录审计并尝试收缩禁运区
    for k in ["Train-A", "Train-B1", "Train-B2", "Validation", "Test"]:
        if len(slices.get(k, [])) < 10:
            audit_logger.log_event("SLICE_TOO_SHORT", {"slice": k, "length": len(slices[k])})
            logger.warning(f"切片 {k} 过短 ({len(slices[k])} 日)，考虑增加数据或调整比例")


def step_2_2_purge_and_embargo_validation(context: dict):
    """
    执行交尾审查：验证相邻区间是否真的物理隔离（使用交易日索引差）
    """
    logger.info("[Step 2.2] 执行 Purge & Embargo 完整性校验")

    slices = context.get('slices', {})
    embargo_window = context.get('embargo_window', 5)
    trading_days_dt = context.get('trading_days_dt', [])
    audit_logger = context.get('audit_logger')

    keys = ["Train-A", "Train-B1", "Train-B2", "Validation", "Test"]
    for i in range(len(keys) - 1):
        left = slices.get(keys[i], [])
        right = slices.get(keys[i + 1], [])
        if not left or not right:
            continue

        last_left = left[-1]
        first_right = right[0]

        # 计算交易日索引差（这才是真正的隔离间距）
        try:
            idx_left = trading_days_dt.index(last_left)
            idx_right = trading_days_dt.index(first_right)
            delta_days = idx_right - idx_left  # 交易日数
        except ValueError:
            # 若日期不在列表中，回退到日历日粗略估算
            delta_days = (first_right - last_left).days // 7 * 5  # 粗略

        if delta_days < embargo_window:
            err_msg = (f"隔离带坍塌！{keys[i]} 与 {keys[i+1]} 交界处间隔仅 {delta_days} 个交易日，"
                       f"低于硬性禁运区下限 {embargo_window} 个交易日。")
            logger.error(err_msg)
            if audit_logger:
                audit_logger.log_event("EMBARGO_VIOLATION", {
                    "left": keys[i],
                    "right": keys[i+1],
                    "delta_days": delta_days,
                    "required": embargo_window
                })
            raise RuntimeError(err_msg)

    # 检查所有切片非空且足够大
    for k in keys:
        if len(slices.get(k, [])) < 5:
            raise RuntimeError(f"切片 {k} 包含交易日过少 ({len(slices[k])})，无法支撑模型训练。")

    logger.info("Purge & Embargo 校验通过 ✅")


def execute(pipeline_context: dict):
    # 确保交易日历存在
    if 'trading_days_dt' not in pipeline_context or not pipeline_context['trading_days_dt']:
        # 尝试从 data_bus 重建（但 data_bus 已持有日历）
        cal = pipeline_context.get('data_bus').manager.fetch_trading_calendar(2010, 2026)
        pipeline_context['trading_days_dt'] = cal.tz_localize(pytz.timezone("Asia/Shanghai")).tolist()

    step_2_1_moving_window_slicing(pipeline_context)
    step_2_2_purge_and_embargo_validation(pipeline_context)
    pipeline_context['slices_isolated'] = True
    return pipeline_context