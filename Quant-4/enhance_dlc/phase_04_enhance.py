# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_04 Enhancement Modules
三重屏障标注 · 元标注 · 并发度去相关权重
设计模式：可插拔增强组件，兼容现有主流程
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, List, Optional, Callable, Any

logger = logging.getLogger("LabelingWeighting.Enhance")

# ====================== 3.1 动态三重屏障标注 ======================
def compute_triple_barrier_labels(
    price_series: pd.Series,
    vol_series: pd.Series,             # 滚动波动率（应与价格索引对齐）
    upper_mult: float = 1.5,
    lower_mult: float = 1.0,
    max_hold_days: int = 5,
    min_ret: float = 0.0               # 最小收益门槛（避免噪声）
) -> Tuple[pd.Series, pd.Series]:
    """
    对单个资产计算三重屏障标签（Triple-Barrier Method）。
    
    :param price_series:      日频价格序列（索引为日期）
    :param vol_series:        日频波动率序列（已对齐）
    :param upper_mult:        上屏障乘数（波动率的倍数）
    :param lower_mult:        下屏障乘数（波动率的倍数，通常取1.0）
    :param max_hold_days:     垂直屏障（最大持有天数）
    :param min_ret:           最小触发收益（避免微幅波动触发）
    :return: (label_series, touch_time_series)
             label: 1=上屏障(止盈), -1=下屏障(止损), 0=垂直屏障到期
             touch_time: 实际触碰屏障的日期（若到期则为 NaN）
    """
    dates = price_series.index
    n = len(dates)
    labels = pd.Series(index=dates, dtype=float)
    touch_dates = pd.Series(index=dates, dtype=object)
    
    for i in range(n - 1):
        entry_price = price_series.iloc[i]
        # 当前交易日的波动率作为屏障基准
        sigma = vol_series.iloc[i]
        if np.isnan(sigma) or sigma <= 0:
            # 若波动率缺失，跳过该样本
            labels.iloc[i] = np.nan
            continue
        
        upper_barrier = entry_price * (1 + upper_mult * sigma)
        lower_barrier = entry_price * (1 - lower_mult * sigma)
        # 垂直屏障到期日
        end_idx = min(i + max_hold_days, n - 1)
        end_date = dates[end_idx]
        
        # 遍历持有期
        touched = False
        for j in range(i + 1, end_idx + 1):
            price = price_series.iloc[j]
            # 检查上屏障
            if price >= upper_barrier:
                # 确保收益超过最小门槛
                if (price / entry_price - 1) >= min_ret:
                    labels.iloc[i] = 1
                    touch_dates.iloc[i] = dates[j]
                    touched = True
                    break
            # 检查下屏障
            if price <= lower_barrier:
                if (price / entry_price - 1) <= -min_ret:
                    labels.iloc[i] = -1
                    touch_dates.iloc[i] = dates[j]
                    touched = True
                    break
        if not touched:
            # 垂直屏障到期
            labels.iloc[i] = 0
            touch_dates.iloc[i] = end_date
    
    return labels, touch_dates


def build_triple_barrier_labels_for_universe(
    assets: List[str],
    train_dates: pd.DatetimeIndex,
    bus: Any,
    vol_window: int = 20,
    min_valid_obs: int = 5,
    upper_mult: float = 1.5,
    lower_mult: float = 1.0,
    max_hold_days: int = 5,
    global_vol_fallback: float = 0.02,
    min_ret: float = 0.001           # 最小收益0.1%
) -> Tuple[Dict[Tuple[pd.Timestamp, str], int], Dict[Tuple[pd.Timestamp, str], float]]:
    """
    对整个股票池批量生成三重屏障标签（替代原有的 build_dual_track_labels）。
    输出格式与 Phase-4 主流程兼容：返回 y_clf (分类标签) 和 y_reg (回归标签，这里使用实际持有期收益率)
    """
    y_clf = {}
    y_reg = {}
    processed = 0
    skipped = 0
    
    for sym in assets:
        try:
            start_dt = train_dates[0] - pd.Timedelta(days=60)
            end_dt = train_dates[-1] + pd.Timedelta(days=max_hold_days + 5)
            df = bus.load_asset_history(sym, start_date=start_dt.strftime("%Y-%m-%d"), end_date=end_dt.strftime("%Y-%m-%d"))
        except RuntimeError:
            skipped += 1
            continue
        
        if df is None or df.empty:
            skipped += 1
            continue
        
        # 标准化索引
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        elif 'Date' in df.columns:
            df.set_index('Date', inplace=True)
        
        # 提取价格
        price_col = None
        for col in ['total_return_price', 'adj_close', 'adjclose', 'Adj Close', 'close', 'Close']:
            if col in df.columns:
                price_col = col
                break
        if not price_col:
            skipped += 1
            continue
        
        price_series = df[price_col]
        price_series = price_series[~price_series.index.duplicated(keep='last')]
        # 对齐训练日期
        prices = price_series.reindex(train_dates, method='ffill')
        if prices.notna().sum() < 2:
            skipped += 1
            continue
        
        # 计算滚动波动率（用日收益率）
        daily_ret = prices.pct_change()
        rolling_vol = daily_ret.rolling(window=vol_window, min_periods=min_valid_obs).std()
        # 填充缺失
        global_median = rolling_vol.median()
        if pd.isna(global_median) or global_median == 0:
            global_median = global_vol_fallback
        rolling_vol_filled = rolling_vol.fillna(global_median)
        
        # 调用单资产三重屏障
        labels, touch_dates = compute_triple_barrier_labels(
            prices, rolling_vol_filled,
            upper_mult=upper_mult,
            lower_mult=lower_mult,
            max_hold_days=max_hold_days,
            min_ret=min_ret
        )
        
        # 将标签填入字典
        for dt in train_dates:
            if dt not in labels.index:
                continue
            label_val = labels.loc[dt]
            if pd.isna(label_val):
                continue
            key = (dt, sym)
            y_clf[key] = int(label_val)
            # 回归标签：实际持有期收益率（从入场到触碰或到期）
            # 计算方式：若触碰时间存在，用触碰价格；否则用到期价格
            touch_dt = touch_dates.loc[dt]
            if pd.isna(touch_dt):
                # 垂直到期，取 max_hold_days 后的价格（注意可能超出索引）
                end_dt = dt + pd.Timedelta(days=max_hold_days)
                if end_dt in prices.index:
                    end_price = prices.loc[end_dt]
                else:
                    # 若超出数据范围，使用最后一个可用价格
                    end_price = prices.asof(end_dt)
                if pd.isna(end_price):
                    continue
            else:
                if touch_dt in prices.index:
                    end_price = prices.loc[touch_dt]
                else:
                    end_price = prices.asof(touch_dt)
                if pd.isna(end_price):
                    continue
            entry_price = prices.loc[dt]
            if entry_price <= 0:
                continue
            ret = end_price / entry_price - 1
            y_reg[key] = float(ret)
        
        processed += 1
        if processed % 100 == 0:
            logger.info("[OP] Progressively Build Triple-Barrier Labels | [SOURCE] Point-In-Time Price Bars | [RESULT] Processed %s assets | [SIGNIFICANCE] Generating path-dependent profit/loss targets", processed)
    
    logger.info("[OP] Terminate Triple-Barrier Labeling | [SOURCE] Universe Loop | [RESULT] Success: %s, Skips: %s | [SIGNIFICANCE] Dynamic barrier targets replace fixed-horizon labels", processed, skipped)
    return y_clf, y_reg


# ====================== 3.2 元标注生成器 ======================
def generate_meta_labels(
    pred_labels: Dict[Tuple[pd.Timestamp, str], int],   # 一阶模型预测的方向 (+1/0/-1)
    actual_returns: Dict[Tuple[pd.Timestamp, str], float],  # 实际持有期收益
    threshold: float = 0.0                              # 正收益定义为正确
) -> Dict[Tuple[pd.Timestamp, str], int]:
    """
    生成元标签：1 表示一阶预测正确（信号方向与实际收益同向且收益 > threshold），0 表示错误。
    """
    meta = {}
    for key, pred in pred_labels.items():
        if key not in actual_returns:
            continue
        ret = actual_returns[key]
        # 若预测为做多（+1）且实际收益 > threshold，则为正确
        # 若预测为中性（0）或做空（-1），可根据策略定义，此处简单处理：预测非0且方向正确
        if pred == 1 and ret > threshold:
            meta[key] = 1
        elif pred == -1 and ret < -threshold:
            meta[key] = 1
        else:
            meta[key] = 0
    logger.info("[OP] Meta-Labeling Generator | [SOURCE] First-order predictions & actual returns | [RESULT] Generated %s meta labels | [SIGNIFICANCE] Provides secondary target for confidence estimation", len(meta))
    return meta


# ====================== 3.3 并发度去相关权重 ======================
def compute_concurrency_weights(
    sample_keys: List[Tuple[pd.Timestamp, str]],
    holding_periods: Dict[Tuple[pd.Timestamp, str], int],   # 每个样本的持有期长度（天数）
    min_holding: int = 1,
    max_holding: int = 30,
    lambda_concurrency: float = 0.5
) -> Dict[Tuple[pd.Timestamp, str], float]:
    """
    基于样本的时间重叠度计算权重（唯一性加权）。
    核心思想：若多个样本的持仓区间大量重叠，则降低其权重，以削弱自相关性。
    采用“时间权重”方法：对每个交易日，统计当天活跃的样本数，权重 = 1 / (活跃数)^alpha。
    """
    # 为每个样本生成其持仓日期范围（从入场日到入场日+持有期-1）
    active_days = {}
    for key in sample_keys:
        date, sym = key
        hold = holding_periods.get(key, min_holding)
        hold = max(min_holding, min(hold, max_holding))
        end_date = date + pd.Timedelta(days=hold)
        # 生成该区间内的所有交易日（需考虑实际交易日，这里简化为日期范围）
        # 更精确应使用交易日历，但为简化，我们用pd.date_range并后续过滤
        # 此处仅记录起止，后续计算每日活跃数时再展开
        active_days[key] = (date, end_date)
    
    # 构建所有日期的索引（从所有样本中最小日期到最大日期）
    all_dates = set()
    for (date, _), (start, end) in active_days.items():
        all_dates.add(start)
        all_dates.add(end)
    all_dates = sorted(list(all_dates))
    # 为了效率，只遍历样本日期
    date_counts = {}
    # 对每个样本，增加其覆盖日期的计数
    for key, (start, end) in active_days.items():
        # 生成日期范围（可能包含非交易日，但后续权重计算只需样本日期）
        # 由于只有样本日期是训练日，我们只计算样本日期的活跃数
        pass  # 我们直接基于样本日期本身计算：对于样本i，其覆盖的样本日期为 [date_i, date_i+hold]
    
    # 更简单的方法：对每个样本，计算与其它样本的重叠度（Jaccard 或重叠天数）
    # 但计算量大，可采用近似方法：对每个样本，权重 = 1 / (该样本所在日期的平均活跃数)
    # 这里实现一个简化版：对每个样本计算其持有期内所有样本的总数
    weights = {}
    n = len(sample_keys)
    for i, key_i in enumerate(sample_keys):
        date_i, sym_i = key_i
        hold_i = holding_periods.get(key_i, min_holding)
        end_i = date_i + pd.Timedelta(days=hold_i)
        overlap_count = 0
        for key_j in sample_keys:
            if key_i == key_j:
                continue
            date_j, sym_j = key_j
            hold_j = holding_periods.get(key_j, min_holding)
            end_j = date_j + pd.Timedelta(days=hold_j)
            # 检查两个区间是否重叠
            if date_i <= end_j and date_j <= end_i:
                overlap_count += 1
        # 权重 = 1 / (1 + overlap_count) 或使用指数衰减
        w = 1.0 / (1 + overlap_count * lambda_concurrency)
        weights[key_i] = w
    
    logger.info("[OP] Concurrency Weighting | [SOURCE] Sample overlapping analysis | [RESULT] Weights range: [%.4f, %.4f] | [SIGNIFICANCE] Reduces autocorrelation from overlapping signals", min(weights.values()), max(weights.values()))
    return weights


# ====================== 3.4 集成增强器：统一封装 ======================
def enhance_phase4_output(
    original_output: Dict,
    use_triple_barrier: bool = False,
    use_meta_labeling: bool = False,
    use_concurrency_weight: bool = False,
    triple_barrier_kwargs: Optional[Dict] = None,
    meta_label_kwargs: Optional[Dict] = None,
    concurrency_kwargs: Optional[Dict] = None,
    bus: Optional[Any] = None,
    train_dates: Optional[pd.DatetimeIndex] = None,
    assets: Optional[List[str]] = None,
    y_reg_original: Optional[Dict] = None   # 若需使用元标注，需提供实际收益
) -> Dict:
    """
    对 Phase-4 主流程的输出进行增强：
    - 替换标签为三重屏障标签 (use_triple_barrier)
    - 生成元标签并加入输出
    - 替换或叠加并发度权重
    返回增强后的输出字典（包含原所有字段，并添加增强字段）
    """
    enhanced = original_output.copy()
    
    # 若启用三重屏障，重新生成标签
    if use_triple_barrier:
        if bus is None or train_dates is None or assets is None:
            raise ValueError("Triple-barrier labeling requires bus, train_dates and assets.")
        kwargs = triple_barrier_kwargs or {}
        y_clf_new, y_reg_new = build_triple_barrier_labels_for_universe(
            assets, train_dates, bus, **kwargs
        )
        enhanced['y_clf_all'] = y_clf_new
        enhanced['y_reg_all'] = y_reg_new
        enhanced['labeling_method'] = 'triple_barrier'
        logger.info("[ENHANCE] Replaced labels with triple-barrier targets.")
    
    # 若启用元标注，需要在输出中添加 meta_labels
    if use_meta_labeling:
        # 需要一阶预测标签（假设来自 original_output 或新生成的 y_clf）
        pred_labels = enhanced.get('y_clf_all')
        if pred_labels is None:
            raise ValueError("Meta-labeling requires y_clf_all in output.")
        # 需要实际收益（可用 y_reg_all 作为实际收益）
        actual_returns = enhanced.get('y_reg_all')
        if actual_returns is None:
            raise ValueError("Meta-labeling requires actual returns (y_reg_all).")
        meta_kwargs = meta_label_kwargs or {}
        meta_labels = generate_meta_labels(pred_labels, actual_returns, **meta_kwargs)
        enhanced['meta_labels'] = meta_labels
        logger.info("[ENHANCE] Added meta-labels based on first-order predictions.")
    
    # 若启用并发度权重
    if use_concurrency_weight:
        # 需要样本 keys 和每个样本的持有期（若三重屏障已计算，可从其返回的持有期得到；否则从原有 y_reg 推断）
        # 为简化，若没有持有期信息，则使用默认持有期（例如 1 天或 5 天）
        sample_keys = list(enhanced.get('y_clf_all', {}).keys())
        if not sample_keys:
            raise ValueError("Concurrency weighting requires sample keys.")
        # 若有 y_reg_original（可能包含实际持有期），可以从中提取
        if y_reg_original is not None:
            # 假设持有期可从 y_reg 的键推断（但未提供，此处演示）
            pass
        # 默认持有期：若有三重屏障，max_hold_days 可获取，否则设为 5
        default_hold = triple_barrier_kwargs.get('max_hold_days', 5) if triple_barrier_kwargs else 5
        holding_periods = {key: default_hold for key in sample_keys}
        conc_kwargs = concurrency_kwargs or {}
        conc_weights = compute_concurrency_weights(sample_keys, holding_periods, **conc_kwargs)
        # 与原有权重相乘（或替换），此处为相乘，保留时间衰减
        orig_weights = enhanced.get('sample_weights', {})
        if orig_weights:
            new_weights = {}
            for key, w in orig_weights.items():
                if key in conc_weights:
                    new_weights[key] = w * conc_weights[key]
                else:
                    new_weights[key] = w
            enhanced['sample_weights'] = new_weights
        else:
            enhanced['sample_weights'] = conc_weights
        enhanced['concurrency_weight_applied'] = True
        logger.info("[ENHANCE] Applied concurrency weighting on top of original weights.")
    
    return enhanced


# ====================== 保留原有装饰器，并扩展 ======================
def dynamic_weight_decay(decay_rate=0.99):
    """
    装饰器：对权重函数的结果施加时序指数衰减（强调近期样本）。
    与主流程中的 compute_exponential_decay_weights 互补，可叠加使用。
    """
    def decorator(weighting_func: Callable):
        def wrapper(*args, **kwargs):
            logger.info(f"[ENHANCE] Injecting time-decay (rate={decay_rate}) into {weighting_func.__name__} ...")
            weights = weighting_func(*args, **kwargs)
            # 假设 weights 是字典或列表，此处按原始逻辑处理
            if isinstance(weights, dict):
                # 按日期排序，赋予衰减
                sorted_keys = sorted(weights.keys(), key=lambda x: x[0])  # 按日期排序
                decay_factors = np.power(decay_rate, np.arange(len(sorted_keys))[::-1])
                adjusted = {}
                for k, factor in zip(sorted_keys, decay_factors):
                    adjusted[k] = weights[k] * factor
                return adjusted
            elif isinstance(weights, (list, np.ndarray)):
                decay_factors = np.power(decay_rate, np.arange(len(weights))[::-1])
                return np.array(weights) * decay_factors
            else:
                return weights
        return wrapper
    return decorator