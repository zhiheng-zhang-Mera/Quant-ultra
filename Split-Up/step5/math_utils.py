# -*- coding: utf-8 -*-
"""
step5/math_utils.py
数学计算与特征物理引擎：包含严格因果递推的分数阶微分算法及经典量化物理特征计算。
"""
import numpy as np
import pandas as pd

def fractional_diff_series(series: np.ndarray, d: float) -> np.ndarray:
    """对一维时间序列进行分数阶微分 $(1-L)^d$，使用严格因果递推（仅依赖历史）"""
    n = len(series)
    if n == 0:
        return series
    weights = [1.0]
    for k in range(1, n):
        w = -weights[-1] * (d - k + 1) / k
        weights.append(w)
    diff = np.zeros(n)
    for t in range(n):
        s = 0.0
        for k in range(t + 1):
            s += weights[k] * series[t - k]
        diff[t] = s
    return diff

def compute_whitebox_features(df: pd.DataFrame) -> np.ndarray:
    """根据单个资产的日线 DataFrame 计算 5 维白盒特征面板"""
    df = df.sort_index()
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    amount = df['amount'].values
    T = len(df)
    features = np.zeros((T, 5))

    # 1. 经典对数收益率
    log_ret = np.full(T, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    features[:, 0] = log_ret

    # 2. 短期 5 日动量
    mom5 = np.full(T, np.nan)
    for t in range(5, T):
        mom5[t] = np.sum(log_ret[t-5:t])
    features[:, 1] = mom5

    # 3. 中期 20 日动量
    mom20 = np.full(T, np.nan)
    for t in range(20, T):
        mom20[t] = np.sum(log_ret[t-20:t])
    features[:, 2] = mom20

    # 4. 非线性高速 Garman-Klass 波动率
    gk = np.full(T, np.nan)
    for t in range(T):
        if high[t] > 0 and low[t] > 0 and open_[t] > 0 and close[t] > 0:
            hl = np.log(high[t] / low[t])
            co = np.log(close[t] / open_[t])
            gk[t] = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    features[:, 3] = gk

    # 5. 自适应流动性换手冲击
    shock = np.full(T, np.nan)
    for t in range(20, T):
        avg_amt = np.mean(amount[t-20:t])
        if avg_amt > 0:
            shock[t] = amount[t] / avg_amt
    features[:, 4] = shock

    return features