# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 4.1: Adaptive Volatility Barrier Label Builder Engine (Long-Only Cleaned)
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Any

logger = logging.getLogger("LabelingWeighting.LabelBuilder")

def build_dual_track_labels(
    assets: list,
    train_dates: pd.DatetimeIndex,
    bus: Any,
    vol_window: int,
    min_valid_obs: int,
    threshold_multiplier: float,
    global_vol_fallback: float
) -> Tuple[Dict[Tuple[pd.Timestamp, str], int], Dict[Tuple[pd.Timestamp, str], float]]:
    
    y_clf_all, y_reg_all = {}, {}
    processed_count, skipped_no_data = 0, 0

    for sym in assets:
        try:
            start_dt = train_dates[0] - pd.Timedelta(days=60)
            end_dt = train_dates[-1] + pd.Timedelta(days=5)
            df = bus.load_asset_history(sym, start_date=start_dt.strftime("%Y-%m-%d"), end_date=end_dt.strftime("%Y-%m-%d"))
        except RuntimeError:
            skipped_no_data += 1; continue

        if df is None or df.empty:
            skipped_no_data += 1; continue

        # 时序索引升维防线：解决 int64 与 datetime 冲突
        if 'date' in df.columns:
            df.set_index('date', inplace=True)
        elif 'Date' in df.columns:
            df.set_index('Date', inplace=True)

        # price_series = df['total_return_price'] # 固定价格
        
        # 动态价格尝试列表，防止不同数据源列名异构
        price_col = None
        possible_cols = ['total_return_price', 'adj_close', 'adjclose', 'Adj Close', 'close', 'Close']
        for col in possible_cols:
            if col in df.columns:
                price_col = col
                break
                
        if not price_col:
            # 如果极端情况下连收盘价都没有，跳过该标的以防止系统崩溃
            continue 
            
        price_series = df[price_col]
        
        # 剔除同日重复的脏数据，防止 reindex 函数底层崩溃
        price_series = price_series[~price_series.index.duplicated(keep='last')]

        prices = price_series.reindex(train_dates, method='ffill')
        if prices.notna().sum() < 2:
            skipped_no_data += 1; continue

        # 1. 远期连续预期收益标签 (y_reg) -> 次日对数全收益变动率
        prices_t1 = prices.shift(-1)
        y_reg = np.log(prices_t1 / prices)

        # 2. 自适应滚动波动率动态阈值抽离
        daily_ret = prices.pct_change()
        rolling_vol = daily_ret.rolling(window=vol_window, min_periods=min_valid_obs).std()
        global_vol_median = rolling_vol.median()
        
        if pd.isna(global_vol_median) or global_vol_median == 0:
            global_vol_median = global_vol_fallback
            
        rolling_vol_filled = rolling_vol.fillna(global_vol_median)
        threshold = rolling_vol_filled * threshold_multiplier

        # 3. 三屏障过滤标签分配 (y_clf ∈ {0, 1}) -> 移除做空，下行及中性常态全部安全阻断归入 0
        y_clf = np.zeros(len(train_dates), dtype=np.int8)
        long_mask = (y_reg >= threshold)
        y_clf[long_mask] = 1
        
        invalid_mask = y_reg.isna() | threshold.isna()
        y_clf[invalid_mask] = 0

        # 4. 压缩压入复合主内存时空哈希拓扑
        for i, d in enumerate(train_dates):
            y_reg_val = y_reg.iloc[i]
            if pd.isna(y_reg_val): continue
            key = (d, sym)
            y_reg_all[key] = float(y_reg_val)
            y_clf_all[key] = int(y_clf[i])

        processed_count += 1
        if processed_count % 500 == 0:
            logger.info("[OP] Progressively Compute Dual Labels | [SOURCE] Point-In-Time Historical Index Traces | [RESULT] Tracked Accumulation: %s active assets | [SIGNIFICANCE] Feeds clean categorical signals into downstream optimization graphs", processed_count)
            logger.info("[操作] 步进式编译双轨标记面板 | [来源] 时点不可变历史行情索引 | [结果] 累计成功解算资产数: %s 只 | [意义] 提炼无前瞻偏差的非线性分类信号，安全投喂给下游优化器决策网络")

    logger.info("[OP] Terminate Dual Track Compilation | [SOURCE] Integrated Context Assets Loop | [RESULT] Success Assets: %s, Data Miss Skips: %s | [SIGNIFICANCE] Establishes authoritative whitebox objective matrices", processed_count, skipped_no_data)
    logger.info("[操作] 终止双轨标签全局大循环 | [来源] 完整集成上下文标的序列 | [结果] 成功解算标的: %s 只, 无数据跳过: %s 只 | [意义] 建立全历史视区权威的分类与回归客观标签目标矩阵")
    return y_clf_all, y_reg_all