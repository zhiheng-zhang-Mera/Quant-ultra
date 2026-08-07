# -*- coding: utf-8 -*-
import logging
import pandas as pd
from Phase_2.config import MIN_SLICE_CRIT_LEN

logger = logging.getLogger("DataSlicing.Validation")

def run_purge_and_embargo_validation(context: dict):
    logger.info("[OP] Invoke Inter-Segment Embargo Sentinel | [SOURCE] Generated Chronological Slices Structure | [RESULT] Commencing structural distance validation | [SIGNIFICANCE] Rigidly prevents post-window info leakage from polluting prior parameters")
    logger.info("[操作] 唤醒跨区间禁运审查哨兵 | [来源] 已生成的时序切片拓扑结构 | [结果] 开始对区间几何间距执行结构硬验证 | [意义] 刚性拦截后窗口的信息向下游或前置参数发生渗透与溢出污染")

    slices = context.get('slices', {})
    embargo_window = context.get('embargo_window', 5)
    audit_logger = context.get('audit_logger')

    keys = ["Train-A", "Train-B1", "Train-B2", "Validation", "Test"]
    markets = {"CN": 'trading_days_dt_cn', "US": 'trading_days_dt_us'}

    flat_slices = all(k in slices for k in ("Train-A", "Train-B1", "Train-B2", "Validation", "Test"))
    if flat_slices:
        slices = {"CN": slices, "US": {}}

    for m_label, cal_key in markets.items():
        calendar = context.get(cal_key, [])
        m_slices = slices.get(m_label, {})
        if not m_slices: continue
        
        # 1. 严格区间交尾间距硬性索引审查
        for i in range(len(keys) - 1):
            left = m_slices.get(keys[i], [])
            right = m_slices.get(keys[i + 1], [])
            if not left or not right: continue

            last_left, first_right = left[-1], right[0]
            try:
                last_left_ts = pd.Timestamp(last_left)
                first_right_ts = pd.Timestamp(first_right)
                cal_norm = []
                for c in calendar:
                    c_ts = pd.Timestamp(c)
                    if getattr(c_ts, "tzinfo", None) is not None:
                        c_ts = c_ts.tz_localize(None)
                    cal_norm.append(c_ts)
                idx_left = cal_norm.index(last_left_ts)
                idx_right = cal_norm.index(first_right_ts)
                delta_days = idx_right - idx_left
            except ValueError:
                delta_days = (pd.Timestamp(first_right) - pd.Timestamp(last_left)).days // 7 * 5

            if delta_days < embargo_window:
                err_msg = (f"【致命级风控熔断】{m_label}轨数据隔离带发生物理坍塌！"
                           f"{keys[i]} 与 {keys[i+1]} 交界实质序号差仅 {delta_days} 交易日，"
                           f"低于统计合规红线安全边界 {embargo_window} 交易日！")
                logger.error(err_msg)
                if audit_logger:
                    audit_logger.log_event("EMBARGO_VIOLATION_CRITICAL", {
                        "market": m_label, "left": keys[i], "right": keys[i+1], "delta": delta_days, "required": embargo_window
                    })
                raise RuntimeError(err_msg)

        # 2. 深度机器学习基础训练规模下限审查
        for k in keys:
            v_slice = m_slices.get(k, [])
            # 特殊高内聚放行：美股轨主要作为共享特征层预训练的 Source Domain，允许其样本外调试及回测分区为空，但核心 Train-A 必须达标！
            if m_label == "US" and k != "Train-A" and len(v_slice) == 0: continue
                
            if len(v_slice) < MIN_SLICE_CRIT_LEN:
                err_len_msg = f"【硬物理熔断】{m_label}轨切片窗口 {k} 包含有效交易日过少 ({len(v_slice)}天)，无法支撑多层级拓扑模型训练！"
                logger.error(err_len_msg)
                raise RuntimeError(err_len_msg)

    logger.info("[OP] Pass Purge & Embargo Firewall | [SOURCE] Structural Matrix Multi-Point Audit | [RESULT] Status: Verified (No leakage found) | [SIGNIFICANCE] Confirms non-overlapping chronological alignment across all modeling zones")
    logger.info("[操作] 通过清除与禁运物理防火墙 | [来源] 几何矩阵多点联合审计 | [结果] 状态: 复核安全通过（未发现任何渗透漏损） | [意义] 确认所有建模建模/验证分区之间不存在交叉信息重叠")
