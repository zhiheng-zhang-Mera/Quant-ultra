# step2/step2_1_slicing.py
import logging
from step2.config import DEFAULT_SLICING_RATIOS, DEFAULT_HOLDING_PERIOD, DEFAULT_EMBARGO_MIN, MIN_SLICE_WARN_LEN
from step2.acf_analyzer import compute_dynamic_acf_lag

logger = logging.getLogger("DataSlicing.Slicing")

def run_moving_window_slicing(context: dict):
    """
    核心切分逻辑：依据规范强制满足 Embargo >= max(holding_period, significant_autocorrelation_lag)
    """
    logger.info("[Step 2.1] 执行多段式物理窗口切分（含 Purge & Embargo）")

    trading_days_dt = context.get('trading_days_dt', [])
    if not trading_days_dt:
        raise ValueError("交易日历 (trading_days_dt) 未在上下文总线中初始化。")

    config = context.get('config', {})
    slicing_ratios = config.get('slicing', {}).get('ratios', DEFAULT_SLICING_RATIOS)
    holding_period = context.get('holding_period', config.get('holding_period', DEFAULT_HOLDING_PERIOD))
    embargo_min = config.get('embargo_min', DEFAULT_EMBARGO_MIN)
    audit_logger = context.get('audit_logger')

    # 调用原子分析器获取自相关滞后
    max_lag = compute_dynamic_acf_lag(context)

    # 刚性确定禁运窗口
    embargo_window = max(holding_period, max_lag, embargo_min)
    context['embargo_window'] = embargo_window
    context['holding_period'] = holding_period
    logger.info(f"持有期: {holding_period}, 显著滞后: {max_lag}, 配置最小禁运区: {embargo_min}, 最终禁运窗口: {embargo_window}")

    # 五段式严格物理截断
    n = len(trading_days_dt)
    raw_a_end = int(n * slicing_ratios[0])
    raw_b1_end = int(n * slicing_ratios[1])
    raw_b2_end = int(n * slicing_ratios[2])
    raw_val_end = int(n * slicing_ratios[3])
    test_end = n

    slices = {}

    # Train-A 截断
    a_end = max(0, raw_a_end - embargo_window)
    slices["Train-A"] = trading_days_dt[0:a_end]

    # Train-B1 双端夹击隔离
    b1_start = min(raw_a_end + embargo_window, raw_b1_end)
    b1_end = max(0, raw_b1_end - embargo_window)
    slices["Train-B1"] = trading_days_dt[b1_start:b1_end] if b1_start < b1_end else []

    # Train-B2 双端夹击隔离
    b2_start = min(raw_b1_end + embargo_window, raw_b2_end)
    b2_end = max(0, raw_b2_end - embargo_window)
    slices["Train-B2"] = trading_days_dt[b2_start:b2_end] if b2_start < b2_end else []

    # Validation 隔离
    val_start = min(raw_b2_end + embargo_window, raw_val_end)
    val_end = max(0, raw_val_end - embargo_window)
    slices["Validation"] = trading_days_dt[val_start:val_end] if val_start < val_end else []

    # Test 样本隔离
    test_start = min(raw_val_end + embargo_window, test_end)
    slices["Test"] = trading_days_dt[test_start:test_end] if test_start < test_end else []

    # 上下文写回
    context['slices'] = slices
    context['raw_split_indices'] = {
        "Train-A_end": raw_a_end,
        "Train-B1_end": raw_b1_end,
        "Train-B2_end": raw_b2_end,
        "Validation_end": raw_val_end,
        "Test_end": test_end
    }

    # 日志输出与轻量级审计
    logger.info("物理切分完毕:")
    for k, v in slices.items():
        start_str = v[0].strftime("%Y-%m-%d") if v else "N/A"
        end_str = v[-1].strftime("%Y-%m-%d") if v else "N/A"
        logger.info(f"   - {k}: {len(v)} 个交易日 ({start_str} ~ {end_str})")

    for k in ["Train-A", "Train-B1", "Train-B2", "Validation", "Test"]:
        if len(slices.get(k, [])) < MIN_SLICE_WARN_LEN:
            if audit_logger:
                audit_logger.log_event("SLICE_TOO_SHORT", {"slice": k, "length": len(slices[k])})
            logger.warning(f"切片 {k} 过短 ({len(slices[k])} 日)，考虑增加回测数据源或重调比例")