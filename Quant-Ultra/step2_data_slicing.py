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
    同时支持从配置中读取自定义最小禁运区（embargo_min）。
    """
    logger.info("[Step 2.1] 执行多段式物理窗口切分（含 Purge & Embargo）")

    trading_days_dt = context.get('trading_days_dt', [])
    if not trading_days_dt:
        raise ValueError("交易日历 (trading_days_dt) 未在上下文总线中初始化。")

    config = context.get('config', {})
    slicing_ratios = config.get('slicing', {}).get('ratios', [0.50, 0.60, 0.70, 0.85])
    holding_period = context.get('holding_period', config.get('holding_period', 5))
    embargo_min = config.get('embargo_min', 5)

    data_bus = context.get('data_bus')
    audit_logger = context.get('audit_logger')

    # ----------------------------
    # 1. 计算动态 Embargo（基于资产池中多个股票的 ACF 平均值）
    # ----------------------------
    max_lag = 1  # 保底
    assets = context.get('assets', [])

    if not assets:
        # 情况1：资产池为空
        logger.warning("资产池为空，无法进行 ACF 估算，使用默认滞后 1")
    else:
        # 检查交易日样本是否足够
        cutoff_idx = int(len(trading_days_dt) * 0.6)
        if cutoff_idx <= 100:
            logger.warning(f"交易日样本不足（仅 {cutoff_idx} 天），无法可靠计算 ACF，使用默认滞后 1")
        else:
            sample_days = trading_days_dt[:cutoff_idx]
            start_dt = sample_days[0].strftime("%Y-%m-%d")
            end_dt = sample_days[-1].strftime("%Y-%m-%d")

            # 取前 100 只（或全部）流动性较好的股票
            sample_assets = assets[:100] if len(assets) > 100 else assets
            lags = []
            success_count = 0
            fail_count = 0
            no_sig_count = 0

            for sym in sample_assets:
                try:
                    df = data_bus.load_asset_history(sym, start_dt, end_dt)
                    if df is not None and len(df) > 100:
                        returns = df['log_return'].dropna()
                        if len(returns) > 100:
                            acf_vals, confint = acf(returns, nlags=10, alpha=0.05, fft=False)
                            sig_lags = []
                            for i in range(1, len(acf_vals)):
                                if i < len(confint):
                                    lower, upper = confint[i]
                                    if acf_vals[i] < lower or acf_vals[i] > upper:
                                        sig_lags.append(i)
                            if sig_lags:
                                lags.append(max(sig_lags))
                                success_count += 1
                            else:
                                no_sig_count += 1
                                logger.debug(f"{sym} 无显著自相关滞后")
                        else:
                            fail_count += 1
                            logger.debug(f"{sym} 有效收益率样本不足")
                    else:
                        fail_count += 1
                        logger.debug(f"{sym} 历史数据加载失败或长度不足")
                except Exception as e:
                    fail_count += 1
                    logger.debug(f"{sym} ACF 计算异常: {e}")

            # 统计结果并输出详细警告
            if lags:
                max_lag = int(np.median(lags))
                logger.info(f"基于 {len(lags)} 只股票的有效 ACF（共尝试 {len(sample_assets)} 只），"
                            f"中位数显著滞后: {max_lag}")
                if fail_count > 0:
                    logger.warning(f"其中 {fail_count} 只股票数据加载失败或数据不足，{no_sig_count} 只无显著滞后")
            else:
                # 情况2：所有股票均无法提取有效 ACF
                if fail_count > 0 and no_sig_count == 0:
                    logger.warning(f"所有尝试的股票（{len(sample_assets)} 只）均数据加载失败或长度不足，"
                                   f"使用默认滞后 1")
                elif no_sig_count > 0 and fail_count == 0:
                    logger.warning(f"所有尝试的股票（{len(sample_assets)} 只）均无显著自相关滞后，"
                                   f"使用默认滞后 1")
                else:
                    logger.warning(f"无法提取有效 ACF（数据失败 {fail_count} 只，无显著 {no_sig_count} 只），"
                                   f"使用默认滞后 1")

    # 2. 刚性确定禁运区
    embargo_window = max(holding_period, max_lag, embargo_min)
    context['embargo_window'] = embargo_window
    context['holding_period'] = holding_period
    logger.info(f"持有期: {holding_period}, 显著滞后: {max_lag}, 配置最小禁运区: {embargo_min}, 最终禁运窗口: {embargo_window}")

    # 3. 五段式切分（带 Purge/Embargo 截断）
    n = len(trading_days_dt)
    raw_a_end = int(n * slicing_ratios[0])
    raw_b1_end = int(n * slicing_ratios[1])
    raw_b2_end = int(n * slicing_ratios[2])
    raw_val_end = int(n * slicing_ratios[3])
    test_end = n

    a_end = max(0, raw_a_end - embargo_window)
    slices = {}

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

        try:
            idx_left = trading_days_dt.index(last_left)
            idx_right = trading_days_dt.index(first_right)
            delta_days = idx_right - idx_left
        except ValueError:
            delta_days = (first_right - last_left).days // 7 * 5

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

    for k in keys:
        if len(slices.get(k, [])) < 5:
            raise RuntimeError(f"切片 {k} 包含交易日过少 ({len(slices[k])})，无法支撑模型训练。")

    logger.info("Purge & Embargo 校验通过 ✅")


def execute(pipeline_context: dict):
    # 确保交易日历存在
    if 'trading_days_dt' not in pipeline_context or not pipeline_context['trading_days_dt']:
        cal = pipeline_context.get('data_bus').manager.fetch_trading_calendar(2010, 2026)
        pipeline_context['trading_days_dt'] = cal.tz_localize(pytz.timezone("Asia/Shanghai")).tolist()

    step_2_1_moving_window_slicing(pipeline_context)
    step_2_2_purge_and_embargo_validation(pipeline_context)
    pipeline_context['slices_isolated'] = True
    return pipeline_context