# step2/step2_2_validation.py
import logging
from step2.config import MIN_SLICE_CRIT_LEN

logger = logging.getLogger("DataSlicing.Validation")

def run_purge_and_embargo_validation(context: dict):
    """
    【分布式跨域多轨隔离带硬性合规性审计卫兵】
    利用交易日历绝对数组执行物理隔离索引差校验，阻断因任何信息跨窗渗透引起的回测净值虚高。
    """
    logger.info("[Step 2.2] 触发分布式多市场双层切片物理强隔离合规校验哨兵")

    slices = context.get('slices', {})
    embargo_window = context.get('embargo_window', 5)
    audit_logger = context.get('audit_logger')

    keys = ["Train-A", "Train-B1", "Train-B2", "Validation", "Test"]
    markets = {"CN": 'trading_days_dt_cn', "US": 'trading_days_dt_us'}

    for m_label, cal_key in markets.items():
        calendar = context.get(cal_key, [])
        m_slices = slices.get(m_label, {})
        if not m_slices:
            continue
        
        # 1. 严格区间交尾间距硬性索引审查
        for i in range(len(keys) - 1):
            left = m_slices.get(keys[i], [])
            right = m_slices.get(keys[i + 1], [])
            if not left or not right:
                continue

            last_left = left[-1]
            first_right = right[0]

            try:
                idx_left = calendar.index(last_left)
                idx_right = calendar.index(first_right)
                delta_days = idx_right - idx_left
            except ValueError:
                # 容灾兜底退化估计
                delta_days = (first_right - last_left).days // 7 * 5

            if delta_days < embargo_window:
                err_msg = (f"【致命级风控熔断】发现重大逻辑漏洞！{m_label}轨数据隔离带发生物理坍塌！"
                           f"{keys[i]} 与 {keys[i+1]} 交界实质序号差仅 {delta_days} 交易日，"
                           f"低于统计合规红线安全边界 {embargo_window} 交易日！")
                logger.error(err_msg)
                if audit_logger:
                    audit_logger.log_event("EMBARGO_VIOLATION_CRITICAL", {
                        "market": m_label, "left": keys[i], "right": keys[i+1],
                        "delta": delta_days, "required": embargo_window
                    })
                raise RuntimeError(err_msg)

        # 2. 深度机器学习基础训练规模下限审查
        for k in keys:
            v_slice = m_slices.get(k, [])
            # 特殊高内聚放行：由于美股轨在阶段五主要作为共享特征层预训练的 Source Domain，
            # 允许其样本外调试及回测分区（B1, B2, Val, Test）为空，但核心主预训练轨 Train-A 必须达标！
            if m_label == "US" and k != "Train-A" and len(v_slice) == 0:
                continue
                
            if len(v_slice) < MIN_SLICE_CRIT_LEN:
                err_len_msg = f"【硬物理熔断】{m_label}轨切片窗口 {k} 包含有效交易日过少 ({len(v_slice)}天)，无法支撑多层级拓扑模型训练！"
                logger.error(err_len_msg)
                raise RuntimeError(err_len_msg)

    logger.info("分布式双轨 Purge & Embargo 物理防火墙全面复核通过，未见逻辑穿透风险 ✅")