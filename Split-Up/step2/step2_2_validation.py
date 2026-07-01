# step2/step2_2_validation.py
import logging
from step2.config import MIN_SLICE_CRIT_LEN

logger = logging.getLogger("DataSlicing.Validation")

def run_purge_and_embargo_validation(context: dict):
    """
    执行交尾审查：验证相邻区间是否真的物理隔离（使用交易日历绝对物理索引差）
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
            # 容灾备用：退化为自然日折算交易日近似估计
            delta_days = (first_right - last_left).days // 7 * 5

        # 核心检查：实际隔离宽度低于硬性要求
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

    # 深度硬性规模检查
    for k in keys:
        if len(slices.get(k, [])) < MIN_SLICE_CRIT_LEN:
            raise RuntimeError(f"物理切片 {k} 包含交易日过少 ({len(slices[k])})，无法有效训练机器学习模型。")

    logger.info("Purge & Embargo 物理隔离审查通过 ✅")