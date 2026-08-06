# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_5 High-Performance Signal Convolution & Feature Physical Engine
"""
import logging
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve
from Main.fast_math import garman_klass_volatility, rolling_sum

logger = logging.getLogger("ModelTraining.Math")

def fractional_diff_series(series: np.ndarray, d: float) -> np.ndarray:
    """
    使用快速傅里叶变换卷积 (FFT Convolution) 将分数阶微分长记忆算子递推时间开销由 O(n²) 彻底压缩至 O(n log n)
    严格截取 mode='full' 的前 n 项，完美确保时间序列级联因果链条中无任何前瞻信息渗透。
    """
    n = len(series)
    if n == 0: return series
        
    weights = np.zeros(n)
    weights[0] = 1.0
    for k in range(1, n):
        weights[k] = -weights[k-1] * (d - k + 1) / k
        
    res = fftconvolve(series, weights, mode='full')
    logger.debug("[OP] Run FFT Signal Convolution | [SOURCE] One-Dimensional Asset Price Array | [RESULT] Convolved matrix length: %s | [SIGNIFICANCE] Accelerates fractional memory transformation under strict causality", n)
    logger.debug("[操作] 执行高性能 FFT 信号因果卷积 | [来源] 一维单资产价格收盘变动序列 | [结果] 卷积输出长度: %s | [意义] 在严格因果链约束下极大加速非平稳时间序列的长记忆特征转化")
    return res[:n]

def compute_whitebox_features(df: pd.DataFrame) -> np.ndarray:
    df = df.sort_index()
    close, open_, high, low = df['close'].values, df['open'].values, df['high'].values, df['low'].values
    T = len(df)
    features = np.zeros((T, 5))

    # 1. 经典对数收益率
    log_ret = np.full(T, np.nan)
    valid = (close[1:] > 0) & (close[:-1] > 0)
    log_ret[1:][valid] = np.log(close[1:][valid] / close[:-1][valid])
    features[:, 0] = log_ret

    # 2. 短期 5 日动量
    mom5 = np.full(T, np.nan)
    if T > 5: mom5[5:] = rolling_sum(log_ret, 5)[4:-1]
    features[:, 1] = mom5

    # 3. 中期 20 日动量
    mom20 = np.full(T, np.nan)
    if T > 20: mom20[20:] = rolling_sum(log_ret, 20)[19:-1]
    features[:, 2] = mom20

    # 4. Garman-Klass 白盒无偏波动率
    gk = np.full(T, np.nan)
    v = (high > 0) & (low > 0) & (close > 0) & (open_ > 0)
    gk[v] = garman_klass_volatility(open_, high, low, close)[v]
    features[:, 3] = gk

    # 5. 横截面成交额偏离度度量
    if 'amount' in df.columns and 'adv_ma20' in df.columns:
        features[:, 4] = (df['amount'].values - df['adv_ma20'].values) / (df['adv_ma20'].values + 1e-8)
        
    return features
