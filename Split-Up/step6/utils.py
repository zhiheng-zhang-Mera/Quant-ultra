import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from sklearn.covariance import LedoitWolf
from typing import Optional

logger = logging.getLogger("PositionSizing.Utils")

def _fractional_diff_series(series: np.ndarray, d: float) -> np.ndarray:
    n = len(series)
    if n == 0 or d == 0:
        return series.copy()
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

def _compute_whitebox_features(df: pd.DataFrame) -> np.ndarray:
    df = df.sort_index()
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    amount = df['amount'].values
    T = len(df)
    features = np.full((T, 5), np.nan)
    log_ret = np.full(T, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    features[:, 0] = log_ret
    for t in range(5, T):
        features[t, 1] = np.sum(log_ret[t-5:t])
    for t in range(20, T):
        features[t, 2] = np.sum(log_ret[t-20:t])
    for t in range(T):
        if high[t] > 0 and low[t] > 0 and open_[t] > 0 and close[t] > 0:
            hl = np.log(high[t] / low[t])
            co = np.log(close[t] / open_[t])
            gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
            features[t, 3] = np.sqrt(max(gk, 0.0))
    for t in range(20, T):
        avg_amt = np.mean(amount[t-20:t])
        if avg_amt > 0:
            features[t, 4] = amount[t] / avg_amt
    return features

def _get_features_for_date(asset: str, date: datetime, context: dict) -> Optional[np.ndarray]:
    bus = context['data_bus']
    d = context.get('best_d', 0.0)
    selected = context.get('selected_features', list(range(5)))
    scaler = context.get('feature_scaler')
    if scaler is None:
        logger.error(f"[特征转换失败] {asset} 缺失 feature_scaler 算子，无法对齐矩阵特征。")
        return None
    end_date = date.strftime("%Y-%m-%d")
    start_date = (date - timedelta(days=180)).strftime("%Y-%m-%d")
    df = bus.load_asset_history(asset, start_date, end_date)
    if df is None or df.empty or len(df) < 30:
        return None
    raw_feat = _compute_whitebox_features(df)
    if np.isnan(raw_feat).any():
        raw_feat = pd.DataFrame(raw_feat).fillna(method='ffill').values
        if np.isnan(raw_feat).any():
            raw_feat = np.nan_to_num(raw_feat, nan=0.0)
    diff_feat = np.zeros_like(raw_feat)
    for f in range(raw_feat.shape[1]):
        diff_feat[:, f] = _fractional_diff_series(raw_feat[:, f], d)
    current_feat = diff_feat[-1, :]
    current_feat = current_feat[selected]
    return scaler.transform(current_feat.reshape(1, -1)).flatten()

def _compute_robust_covariance(bus, assets, date, lookback=252):
    returns = []
    end_date_str = date.strftime('%Y-%m-%d')
    for asset in assets:
        hist = bus.load_asset_history(asset, end_date=end_date_str)
        if hist is None or hist.empty:
            ret = np.zeros(lookback)
        else:
            ret_series = hist['log_return'].tail(lookback)
            if len(ret_series) < 50:
                ret_series = hist['log_return']
            ret = ret_series.values
            if len(ret) < 2:
                ret = np.zeros(lookback)
        if len(ret) > lookback:
            ret = ret[-lookback:]
        elif len(ret) < lookback:
            pad = lookback - len(ret)
            ret = np.pad(ret, (pad, 0), constant_values=np.nan)
        returns.append(ret)
    X = np.vstack(returns).T
    X_clean = X[~np.isnan(X).any(axis=1)]
    if X_clean.shape[0] < 10:
        n = len(assets)
        logger.warning(f"[协方差降级估计] 日期 {end_date_str} 有效横截面数据行不足 ({X_clean.shape[0]}行)，降级启用单位阵缩放。")
        return np.eye(n) * 0.01
    try:
        lw = LedoitWolf().fit(X_clean)
        return lw.covariance_
    except Exception as e:
        logger.error(f"[LedoitWolf 估计崩溃] 日期 {end_date_str} 矩阵拟合失败: {e}，触发安全防御。")
        return np.eye(len(assets)) * 0.01

def _compute_individual_shares_upper(bus, asset: str, date: datetime, nav: float, config: dict) -> float:
    """
    针对个人账户流动性进行资金量双重硬约束防线（废除原大股东举牌4.5%条款）
    单票权重上限 = min(10%, 10% * ADV_20 / 账户总权益)
    """
    lookback_adv = config.get('lookback_adv', 20)
    end_date_str = date.strftime('%Y-%m-%d')
    start_date_str = (date - timedelta(days=45)).strftime('%Y-%m-%d')
    
    try:
        hist = bus.load_asset_history(asset, start_date=start_date_str, end_date=end_date_str)
        if hist is not None and not hist.empty:
            adv_20 = hist['amount'].tail(lookback_adv).mean()
        else:
            adv_20 = 0.0
    except Exception as e:
        logger.debug(f"[{asset}] 无法抽取流式交易额数据: {e}，安全降级至固定10%红线。")
        return 0.10

    if adv_20 <= 0 or nav <= 0:
        return 0.10
        
    liquidity_weight_cap = 0.10 * (adv_20 / nav)
    return min(0.10, liquidity_weight_cap)

def _compute_market_weights(bus, assets, date):
    caps = []
    for asset in assets:
        try:
            cap = bus.get_free_float_market_cap(asset, date)
            caps.append(max(cap, 0.0))
        except:
            caps.append(0.0)
    total = sum(caps)
    if total <= 0:
        return np.ones(len(assets)) / len(assets)
    return np.array(caps) / total