# step2/step2_1_slicing.py
import logging
from step2.config import DEFAULT_SLICING_RATIOS, DEFAULT_HOLDING_PERIOD, DEFAULT_EMBARGO_MIN, MIN_SLICE_WARN_LEN
from step2.acf_analyzer import compute_dynamic_acf_lag

logger = logging.getLogger("DataSlicing.Slicing")

def run_moving_window_slicing(context: dict):
    """
    【双轨交易日序号令牌对齐切片引擎】
    严格遵循美股第 N 个交易日对齐 A 股第 N 个交易日硬防线，同步切割跨市场时间序列。
    """
    logger.info("[Step 2.1] 启动跨市场双轨时序序号同步切分管线")

    trading_days_dt_cn = context.get('trading_days_dt_cn', [])
    trading_days_dt_us = context.get('trading_days_dt_us', [])
    
    if not trading_days_dt_cn or not trading_days_dt_us:
        raise ValueError("缺失多节点核心日历底座 (trading_days_dt_cn 或 trading_days_dt_us 未初始化)")

    config = context.get('config', {})
    slicing_ratios = config.get('slicing', {}).get('ratios', DEFAULT_SLICING_RATIOS)
    holding_period = context.get('holding_period', config.get('holding_period', DEFAULT_HOLDING_PERIOD))
    embargo_min = config.get('embargo_min', DEFAULT_EMBARGO_MIN)
    audit_logger = context.get('audit_logger')

    # 调用分布式联合算子提取最高显著滞后
    max_lag = compute_dynamic_acf_lag(context)

    # 刚性确定全局隔离窗口宽度
    embargo_window = max(holding_period, max_lag, embargo_min)
    context['embargo_window'] = embargo_window
    context['holding_period'] = holding_period
    
    # 以A股（Target Domain主战场）序号轴为全流程基准母版
    n_cn = len(trading_days_dt_cn)
    n_us = len(trading_days_dt_us)
    
    raw_a_end = int(n_cn * slicing_ratios[0])
    raw_b1_end = int(n_cn * slicing_ratios[1])
    raw_b2_end = int(n_cn * slicing_ratios[2])
    raw_val_end = int(n_cn * slicing_ratios[3])
    test_end = n_cn

    # 声明双轨分区存储拓扑
    slices = {"CN": {}, "US": {}}

    # --- 核心切分逻辑（交易日序号令牌范围克隆映射） ---
    def slice_track(calendar, total_len):
        track_slices = {}
        # Train-A 物理左截断
        a_end = max(0, raw_a_end - embargo_window)
        track_slices["Train-A"] = calendar[0:min(a_end, total_len)]

        # Train-B1 双端夹击隔离带
        b1_start = min(raw_a_end + embargo_window, raw_b1_end)
        b1_end = max(0, raw_b1_end - embargo_window)
        track_slices["Train-B1"] = calendar[b1_start:min(b1_end, total_len)] if b1_start < b1_end else []

        # Train-B2 双端夹击隔离带
        b2_start = min(raw_b1_end + embargo_window, raw_b2_end)
        b2_end = max(0, raw_b2_end - embargo_window)
        track_slices["Train-B2"] = calendar[b2_start:min(b2_end, total_len)] if b2_start < b2_end else []

        # Validation 隔离带
        val_start = min(raw_b2_end + embargo_window, raw_val_end)
        val_end = max(0, raw_val_end - embargo_window)
        track_slices["Validation"] = calendar[val_start:min(val_end, total_len)] if val_start < val_end else []

        # Test 样本绝对物理区
        test_start = min(raw_val_end + embargo_window, test_end)
        track_slices["Test"] = calendar[test_start:min(test_end, total_len)] if test_start < total_len else []
        return track_slices

    # 顺次填充海内外隔离切片区 (美股 Train-A 为阶段5提供 Source Domain 对抗特征预训练)
    slices["CN"] = slice_track(trading_days_dt_cn, n_cn)
    slices["US"] = slice_track(trading_days_dt_us, n_us)

    # 状态写回主总线
    context['slices'] = slices
    context['raw_split_indices'] = {
        "Train-A_end": raw_a_end,
        "Train-B1_end": raw_b1_end,
        "Train-B2_end": raw_b2_end,
        "Validation_end": raw_val_end,
        "Test_end": test_end
    }

    # 双轨审计与长尾对齐健壮度观测
    logger.info(f"双轨时序同步序号物理切分完成 (Embargo宽: {embargo_window} 交易日):")
    for m_key in ["CN", "US"]:
        logger.info(f" 🌐 [{m_key} 市场轨分区时空镜像]")
        for k in ["Train-A", "Train-B1", "Train-B2", "Validation", "Test"]:
            v = slices[m_key].get(k, [])
            st = v[0].strftime("%Y-%m-%d") if v else "N/A"
            ed = v[-1].strftime("%Y-%m-%d") if v else "N/A"
            logger.info(f"   - {k}: {len(v)} 天 ({st} ~ {ed})")
            
            if len(v) < MIN_SLICE_WARN_LEN and len(v) > 0:
                if audit_logger:
                    audit_logger.log_event("SLICE_TOO_SHORT_WARNING", {"market": m_key, "slice": k, "length": len(v)})
                logger.warning(f"[时空审计] ⚠️ {m_key}轨分区 {k} 长度过短({len(v)}天)，请核实回测视区")