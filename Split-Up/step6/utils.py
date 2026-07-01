import numpy as np
import pandas as pd
import cvxpy as cp
import akshare as ak
import logging
from datetime import datetime, timedelta
from sklearn.covariance import LedoitWolf
from typing import Optional, Set

logger = logging.getLogger("PositionSizing.Utils")

# 模块级物理缓存，防止在每日循环内发生几十万次完全重复的个股总股本网络抓取请求
_TOTAL_SHARES_CACHE = {}

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

def _fetch_borrowable_stocks(date: datetime) -> Set[str]:
    try:
        date_str = date.strftime("%Y%m%d")
        df = ak.stock_borrow_analysis(date=date_str)
        if df.empty:
            logger.debug(f"[融券列表审计] 日期 {date.strftime('%Y-%m-%d')} AkShare 融券分析接口返回空数据。")
            return set()
        df = df[df['融券余量'] > 0]
        codes = df['代码'].astype(str).tolist()
        result = set()
        for c in codes:
            if len(c) != 6:
                continue
            if c.startswith('6'):
                result.add(f"{c}.SH")
            elif c.startswith('0') or c.startswith('3'):
                result.add(f"{c}.SZ")
        return result
    except Exception as e:
        logger.warning(f"[网络级融券获取失败] 日期 {date.strftime('%Y-%m-%d')} 接口异常: {e}，今日默认清空券池。")
        return set()

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

def _compute_shares_upper(bus, asset: str, date: datetime, nav: float, config: dict) -> float:
    max_shares_pct = config.get('max_shares_pct', 0.045)
    if asset in _TOTAL_SHARES_CACHE:
        total_shares = _TOTAL_SHARES_CACHE[asset]
    else:
        try:
            pure_code = asset.split('.')[0]
            info = ak.stock_individual_info_em(symbol=pure_code)
            total_shares = info[info['item']=='总股本']['value'].values[0]
            _TOTAL_SHARES_CACHE[asset] = total_shares
        except Exception as e:
            logger.debug(f"[{asset}] 无法抓取总股本元数据: {e}，启用刚性默认上限。")
            return max_shares_pct
    try:
        price = bus.query_by_pit(asset, date, "total_return_price")
        if price is None or price <= 0:
            return max_shares_pct
        market_value = total_shares * price
        max_weight = (max_shares_pct * market_value) / nav if nav > 0 else max_shares_pct
        return min(max_weight, max_shares_pct)
    except Exception as e:
        logger.warning(f"[{asset}] 股本比例换算异常: {e}，锁死刚性上限。")
        return max_shares_pct

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