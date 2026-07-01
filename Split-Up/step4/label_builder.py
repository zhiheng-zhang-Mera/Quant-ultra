# -*- coding: utf-8 -*-
"""
step4/label_builder.py
双轨标签（y_clf 与 y_reg）编译面板内核
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Set, Any

logger = logging.getLogger("LabelingWeighting.LabelBuilder")

def build_dual_track_labels(
    assets: list,
    train_dates: pd.DatetimeIndex,
    bus: Any,
    borrowable_set: Set[str],
    vol_window: int,
    min_valid_obs: int,
    threshold_multiplier: float,
    global_vol_fallback: float
) -> Tuple[Dict[Tuple[pd.Timestamp, str], int], Dict[Tuple[pd.Timestamp, str], float]]:
    """
    批量计算全宇宙资产在 Train-A 区间内的双轨标签
    """
    y_clf_all = {}
    y_reg_all = {}
    processed_count = 0
    skipped_no_data = 0

    for sym in assets:
        try:
            # 严格按照 PITDataBus 时空轴回溯边界拉取数据
            start_dt = train_dates[0] - pd.Timedelta(days=60)
            end_dt = train_dates[-1] + pd.Timedelta(days=5)
            df = bus.load_asset_history(
                sym,
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d")
            )
        except RuntimeError as e:
            logger.warning(f"加载 {sym} 历史失败: {e}，跳过")
            skipped_no_data += 1
            continue

        if df is None or df.empty:
            skipped_no_data += 1
            continue

        price_series = df['total_return_price']
        prices = price_series.reindex(train_dates, method='ffill')

        if prices.notna().sum() < 2:
            skipped_no_data += 1
            continue

        # 1. 远期连续对数收益率标签计算 (y_reg)
        prices_t1 = prices.shift(-1)
        y_reg = np.log(prices_t1 / prices)

        # 2. 自适应滚动波动率阈值判定
        daily_ret = prices.pct_change()
        rolling_vol = daily_ret.rolling(window=vol_window, min_periods=min_valid_obs).std()
        global_vol_median = rolling_vol.median()
        
        if pd.isna(global_vol_median) or global_vol_median == 0:
            global_vol_median = global_vol_fallback
            
        rolling_vol_filled = rolling_vol.fillna(global_vol_median)
        threshold = rolling_vol_filled * threshold_multiplier

        # 3. 三屏障方向过滤标签分配 (y_clf)
        y_clf = np.zeros(len(train_dates), dtype=np.int8)
        
        long_mask = (y_reg >= threshold)
        y_clf[long_mask] = 1
        
        short_mask = (y_reg <= -threshold)
        if sym in borrowable_set:
            y_clf[short_mask] = -1
        else:
            y_clf[short_mask] = 0  # 融券不可用时，做空信号无条件降级为中性
            
        # 隔离任何非有限值与空头断层
        invalid_mask = y_reg.isna() | threshold.isna()
        y_clf[invalid_mask] = 0

        # 4. 时空联合键对齐压入字典总线
        for i, d in enumerate(train_dates):
            y_reg_val = y_reg.iloc[i]
            if pd.isna(y_reg_val):
                continue
            key = (d, sym)
            y_reg_all[key] = float(y_reg_val)
            y_clf_all[key] = int(y_clf[i])

        processed_count += 1
        if processed_count % 500 == 0:
            logger.info(f"已处理 {processed_count} 只资产，跳过 {skipped_no_data} 只无数据")

    logger.info(f"✅ 标签生成完成。有效标的: {processed_count}，无数据跳过: {skipped_no_data}")
    return y_clf_all, y_reg_all