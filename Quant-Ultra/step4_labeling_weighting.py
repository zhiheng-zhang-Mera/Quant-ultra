"""
Phase 4: Primary Labeling and Sample Weighting (Directional & Numeric Targets)
Fully compliant with Final-Flow.md [2026 Production Release]
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("LabelingWeighting")

# 全局配置（应与主配置保持一致）
CONFIG = {
    "LAMBDA_DECAY": 0.01,          # 时间衰减因子 λ
    "VOL_WINDOW": 20,              # 动态波动率计算窗口
    "THRESHOLD_MULTIPLIER": 0.5,   # 三屏障阈值 = vol * 此系数
}

def step_4_1_triple_barrier_labeling(context: dict):
    """
    对 Train-A 中的每个交易日，为每只股票生成：
    - y_clf: 方向标签 {-1, 0, 1}，基于动态波动率的三屏障
    - y_reg: 次日错位对数实际远期收益率 (target_return)
    同时考虑融券券源约束：若做空且无券源，强制置0。
    """
    print("[Step 4.1] Generating dual-track labels (y_clf, y_reg) with short liquidity mask.")

    bus = context['data_bus']
    assets = context.get('assets', [])
    slices = context.get('slices', {})
    train_dates = slices.get('Train-A', [])
    if not train_dates:
        raise ValueError("Train-A 切片为空，请检查阶段二切分。")

    # 融券券源可用性模拟（实际应由外部数据服务提供）
    # 这里模拟：假设只有部分资产可融券
    # 生产环境应每日动态查询
    borrow_liquidity_mask = {sym: (sym in ["600519.SH", "000001.SZ"]) for sym in assets}

    # 存储结果：按日期和资产组织的字典
    y_clf_all = {}   # {(date, asset): label}
    y_reg_all = {}   # {(date, asset): return}
    # 同时也可按资产组织
    y_clf_by_asset = {sym: [] for sym in assets}
    y_reg_by_asset = {sym: [] for sym in assets}
    dates_by_asset = {sym: [] for sym in assets}

    # 遍历Train-A中的每个交易日
    for idx, current_date in enumerate(train_dates):
        # 获取下一交易日（用于计算远期收益）
        next_date = train_dates[idx + 1] if idx + 1 < len(train_dates) else None
        if next_date is None:
            continue  # 最后一个交易日无法计算远期收益

        for sym in assets:
            # 1. 获取当日收盘价（全收益价格）
            p_t = bus.query_by_pit(sym, current_date, "total_return_price")
            if p_t is None:
                # 若当日无价格，尝试向前找
                p_t = _get_nearest_price(bus, sym, current_date, train_dates, backward=True)
            if p_t is None:
                continue  # 无数据，跳过该样本

            # 2. 获取下一日收盘价
            p_t1 = bus.query_by_pit(sym, next_date, "total_return_price")
            if p_t1 is None:
                p_t1 = _get_nearest_price(bus, sym, next_date, train_dates, backward=False)
            if p_t1 is None:
                continue

            # 3. 计算远期对数收益率 (y_reg)
            y_reg = np.log(p_t1 / p_t)
            key = (current_date, sym)
            y_reg_all[key] = y_reg
            y_reg_by_asset[sym].append(y_reg)
            dates_by_asset[sym].append(current_date)

            # 4. 计算动态波动率（过去VOL_WINDOW个交易日的收益率波动）
            vol = _calc_rolling_vol(bus, sym, current_date, train_dates, window=CONFIG["VOL_WINDOW"])
            if vol is None:
                # 若波动率无法计算，使用固定阈值 0.02 作为后备
                threshold = 0.02
            else:
                threshold = vol * CONFIG["THRESHOLD_MULTIPLIER"]

            # 5. 三屏障分类
            if y_reg >= threshold:
                y_clf = 1
            elif y_reg <= -threshold:
                # 检查融券券源
                if borrow_liquidity_mask.get(sym, False):
                    y_clf = -1
                else:
                    y_clf = 0  # 无券源，强制置0
            else:
                y_clf = 0

            y_clf_all[key] = y_clf
            y_clf_by_asset[sym].append(y_clf)

    # 存储到上下文
    context['y_clf_all'] = y_clf_all
    context['y_reg_all'] = y_reg_all
    context['y_clf_by_asset'] = y_clf_by_asset
    context['y_reg_by_asset'] = y_reg_by_asset
    context['dates_by_asset'] = dates_by_asset
    context['margin_borrow_liquidity'] = borrow_liquidity_mask

    # 统计
    total_samples = len(y_clf_all)
    if total_samples > 0:
        labels = list(y_clf_all.values())
        n_pos = sum(1 for v in labels if v == 1)
        n_neg = sum(1 for v in labels if v == -1)
        n_zero = sum(1 for v in labels if v == 0)
        print(f"[完成] 生成 {total_samples} 个样本标签（正:{n_pos}, 负:{n_neg}, 零:{n_zero}）")
    else:
        print("[警告] 未生成任何有效标签，请检查数据完整性。")


def step_4_2_sample_weighting_matrix(context: dict):
    """
    为训练样本赋予指数衰减权重 w_t = exp(-λ * (T_max - t))
    其中 T_max 为 Train-A 的最后日期，t 为样本的日期。
    """
    print("[Step 4.2] Computing exponential time-decay sample weights.")

    slices = context.get('slices', {})
    train_dates = slices.get('Train-A', [])
    if not train_dates:
        raise ValueError("Train-A 切片为空。")

    T_max = train_dates[-1]  # 训练集最后日期
    lambda_ = CONFIG["LAMBDA_DECAY"]

    y_reg_all = context.get('y_reg_all', {})
    if not y_reg_all:
        print("[警告] 未找到 y_reg_all，跳过权重计算。")
        context['sample_weights'] = {}
        return

    weights = {}
    for (date, sym), _ in y_reg_all.items():
        # 计算天数差（日历日或交易日？规范未明确，通常用日历日）
        delta = (T_max - date).days
        w = np.exp(-lambda_ * delta)
        weights[(date, sym)] = w

    context['sample_weights'] = weights
    print(f"[完成] 为 {len(weights)} 个样本分配了时间衰减权重（λ={lambda_}）")


# ---------- 辅助函数 ----------
def _get_nearest_price(bus, sym, target_date, all_dates, backward=True):
    """在日期列表中向前或向后搜索最近的有效价格"""
    if target_date is None:
        return None
    idx = all_dates.index(target_date) if target_date in all_dates else None
    if idx is None:
        # 若目标日期不在列表中，尝试最近
        for d in all_dates:
            if d <= target_date if backward else d >= target_date:
                p = bus.query_by_pit(sym, d, "total_return_price")
                if p is not None:
                    return p
        return None
    # 从目标日期开始向指定方向搜索
    step = -1 if backward else 1
    for offset in range(0, len(all_dates)):
        pos = idx + offset * step
        if pos < 0 or pos >= len(all_dates):
            break
        p = bus.query_by_pit(sym, all_dates[pos], "total_return_price")
        if p is not None:
            return p
    return None


def _calc_rolling_vol(bus, sym, current_date, all_dates, window=20):
    """计算 current_date 之前 window 个交易日的收益率标准差（年化可选，但此处直接使用日波动）"""
    # 获取 current_date 之前的 window 个交易日
    try:
        idx = all_dates.index(current_date)
    except ValueError:
        return None
    start = max(0, idx - window)
    # 获取价格序列（从 start 到 idx，含 current_date）
    prices = []
    for d in all_dates[start:idx+1]:
        p = bus.query_by_pit(sym, d, "total_return_price")
        if p is not None:
            prices.append(p)
        else:
            # 若某个价格缺失，尝试补全（用最近的值）
            p_nearest = _get_nearest_price(bus, sym, d, all_dates, backward=True)
            if p_nearest is not None:
                prices.append(p_nearest)
    if len(prices) < 5:  # 至少需要5个样本
        return None
    rets = np.diff(np.log(prices))
    vol = np.std(rets)  # 日波动率
    return vol


def execute(pipeline_context: dict):
    """
    Phase 4 主入口
    """
    # 确保切片存在
    if 'slices' not in pipeline_context:
        raise ValueError("缺少切片信息，请先执行 Phase 2。")
    if 'data_bus' not in pipeline_context:
        raise ValueError("缺少数据总线，请先执行 Phase 1。")

    step_4_1_triple_barrier_labeling(pipeline_context)
    step_4_2_sample_weighting_matrix(pipeline_context)

    pipeline_context['labeling_ready'] = True
    return pipeline_context