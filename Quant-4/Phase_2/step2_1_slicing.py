# -*- coding: utf-8 -*-
import logging
from Phase_2.config import DEFAULT_SLICING_RATIOS, DEFAULT_HOLDING_PERIOD, DEFAULT_EMBARGO_MIN, MIN_SLICE_WARN_LEN
from Phase_2.acf_analyzer import compute_dynamic_acf_lag

logger = logging.getLogger("DataSlicing.Slicing")

def run_moving_window_slicing(context: dict):
    logger.info("[OP] Launch Multi-Track Index Alignment Engine | [SOURCE] Dual-Market Standard Datetime Indices | [RESULT] Activating synchronous chronology slice workflow | [SIGNIFICANCE] Guarantees that historical token N in US aligns precisely with token N in A-share")
    logger.info("[操作] 启动多轨索引对齐引擎 | [来源] 双市场标准日期时间索引 | [结果] 激活同步时序切片工作流 | [意义] 保证海外市场的历史第 N 个交易日代数令牌与 A 股主战场精准锚定对齐")

    trading_days_dt_cn = context.get('trading_days_dt_cn', [])
    trading_days_dt_us = context.get('trading_days_dt_us', [])
    if not trading_days_dt_cn or not trading_days_dt_us:
        raise ValueError("Missing essential multi-node core calendars.")

    config = context.get('config', {})
    slicing_ratios = config.get('slicing', {}).get('ratios', DEFAULT_SLICING_RATIOS)
    holding_period = context.get('holding_period', config.get('holding_period', DEFAULT_HOLDING_PERIOD))
    embargo_min = config.get('embargo_min', DEFAULT_EMBARGO_MIN)
    audit_logger = context.get('audit_logger')

    max_lag = compute_dynamic_acf_lag(context)
    embargo_window = max(holding_period, max_lag, embargo_min)
    context['embargo_window'] = embargo_window
    context['holding_period'] = holding_period
    
    logger.info("[OP] Lock Global Embargo Insulation Window | [SOURCE] Combined Max(Holding, ACF Lag, Embargo Min) | [RESULT] Calculated Guard Width: %s Trading Days | [SIGNIFICANCE] Hardens mathematical firewall widths to enforce physical spacing between partitions", embargo_window)
    logger.info("[操作] 锁定全局禁运隔离视窗 | [来源] 归并上限算子值 | [结果] 算得安全垫宽度: %s 个交易日 | [意义] 固化数学防火墙宽度，强制各回测/训练分区在物理上不产生交叉重叠")

    n_cn, n_us = len(trading_days_dt_cn), len(trading_days_dt_us)
    raw_a_end = int(n_cn * slicing_ratios[0])
    raw_b1_end = int(n_cn * slicing_ratios[1])
    raw_b2_end = int(n_cn * slicing_ratios[2])
    raw_val_end = int(n_cn * slicing_ratios[3])
    test_end = n_cn

    slices = {"CN": {}, "US": {}}

    def slice_track(calendar, total_len):
        track_slices = {}
        a_end = max(0, raw_a_end - embargo_window)
        track_slices["Train-A"] = calendar[0:min(a_end, total_len)]

        b1_start = min(raw_a_end + embargo_window, raw_b1_end)
        b1_end = max(0, raw_b1_end - embargo_window)
        track_slices["Train-B1"] = calendar[b1_start:min(b1_end, total_len)] if b1_start < b1_end else []

        b2_start = min(raw_b1_end + embargo_window, raw_b2_end)
        b2_end = max(0, raw_b2_end - embargo_window)
        track_slices["Train-B2"] = calendar[b2_start:min(b2_end, total_len)] if b2_start < b2_end else []

        val_start = min(raw_b2_end + embargo_window, raw_val_end)
        val_end = max(0, raw_val_end - embargo_window)
        track_slices["Validation"] = calendar[val_start:min(val_end, total_len)] if val_start < val_end else []

        test_start = min(raw_val_end + embargo_window, test_end)
        track_slices["Test"] = calendar[test_start:min(test_end, total_len)] if test_start < total_len else []
        return track_slices

    slices["CN"] = slice_track(trading_days_dt_cn, n_cn)
    slices["US"] = slice_track(trading_days_dt_us, n_us)

    context['slices'] = slices
    context['raw_split_indices'] = {
        "Train-A_end": raw_a_end, "Train-B1_end": raw_b1_end, "Train-B2_end": raw_b2_end,
        "Validation_end": raw_val_end, "Test_end": test_end
    }

    for m_key in ["CN", "US"]:
        for k in ["Train-A", "Train-B1", "Train-B2", "Validation", "Test"]:
            v = slices[m_key].get(k, [])
            if 0 < len(v) < MIN_SLICE_WARN_LEN:
                if audit_logger: audit_logger.log_event("SLICE_TOO_SHORT_WARNING", {"market": m_key, "slice": k, "length": len(v)})
                logger.warning(f"[时空审计] ⚠️ {m_key}轨分区 {k} 长度过短({len(v)}天)，请核实回测视区")