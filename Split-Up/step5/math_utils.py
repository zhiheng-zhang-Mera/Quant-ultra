# -*- coding: utf-8 -*-
"""
step5/math_utils.py
数学计算与特征物理引擎：包含由 FFT 傅里叶变换加速的分数阶微分算法及经典量化物理特征计算。
"""
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

def fractional_diff_series(series: np.ndarray, d: float) -> np.ndarray:
    """
    对一维时间序列进行分数阶微分 $(1-L)^d$
    🚀 升级方案：使用快速傅里叶变换卷积 (FFT Convolution) 将时空开销从 O(n²) 彻底降维至 O(n log n)
    🟩 严格因果：通过截取 mode='full' 的前 n 项，完美确保递推链条中无任何前瞻信息泄露。
    """
    n = len(series)
    if n == 0:
        return series
        
    # 1. 预计算高维长记忆自回归权重向量 (长度对齐 n)
    weights = np.zeros(n)
    weights[0] = 1.0
    for k in range(1, n):
        weights[k] = -weights[k-1] * (d - k + 1) / k
        
    # 2. 执行一维高性能 FFT 信号因果卷积
    # 卷积结果的前 n 个元素物理契合：\sum_{k=0}^t weights[k] * series[t-k]
    res = fftconvolve(series, weights, mode='full')
    return res[:n]

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
    valid = (close[1:] > 0) & (close[:-1] > 0)
    log_ret[1:][valid] = np.log(close[1:][valid] / close[:-1][valid])
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