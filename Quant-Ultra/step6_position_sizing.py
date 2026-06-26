"""
Phase 6: Multi-directional Uncertainty Position Sizing and Convex Optimization
Full implementation with Black-Litterman fusion, CQR intervals, and MVO.
Now precomputes daily weights and intervals for the entire Test set.
Fully compliant with Final-Flow.md [2026 Production Release]
"""
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.covariance import LedoitWolf
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import pytz

logger = logging.getLogger("PositionSizing")

# ---------- 辅助函数（保持不变） ----------
def _fractional_diff_series(series: np.ndarray, d: float) -> np.ndarray:
    """因果递推分数阶微分 (1-L)^d，仅依赖历史。"""
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
    """计算五个白盒特征：Mom_1D, Mom_5D, Mom_20D, GK_Vol, Turnover_Shock。"""
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
    """为指定资产在给定日期提取特征向量（经分数阶微分、特征选择、标准化）。"""
    bus = context['data_bus']
    d = context.get('best_d', 0.0)
    selected = context.get('selected_features', list(range(5)))
    scaler = context.get('feature_scaler')
    if scaler is None:
        logger.error("Missing feature_scaler in context, cannot transform.")
        return None
    end_date = date.strftime("%Y-%m-%d")
    start_date = (date - timedelta(days=180)).strftime("%Y-%m-%d")
    df = bus.load_asset_history(asset, start_date, end_date)
    if df is None or df.empty or len(df) < 30:
        return None
    raw_feat = _compute_whitebox_features(df)
    if np.isnan(raw_feat).any():
        raw_feat = pd.DataFrame(raw_feat).fillna(method='ffill').values
    diff_feat = np.zeros_like(raw_feat)
    for f in range(raw_feat.shape[1]):
        diff_feat[:, f] = _fractional_diff_series(raw_feat[:, f], d)
    current_feat = diff_feat[-1, :]
    current_feat = current_feat[selected]
    current_feat_scaled = scaler.transform(current_feat.reshape(1, -1)).flatten()
    return current_feat_scaled

def _fetch_borrowable_stocks(date: datetime) -> set:
    """从 AkShare 获取指定日期的融券可用标的集合。"""
    try:
        import akshare as ak
        date_str = date.strftime("%Y%m%d")
        df = ak.stock_borrow_analysis(date=date_str)
        if df.empty:
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
        logger.warning(f"获取融券列表失败 ({date}): {e}")
        return set()

def _compute_robust_covariance(bus, assets, date, lookback=252):
    """Ledoit-Wolf 协方差估计，使用真实历史收益率。"""
    returns = []
    for asset in assets:
        hist = bus.load_asset_history(asset, end_date=date.strftime('%Y-%m-%d'))
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
    X = X[~np.isnan(X).any(axis=1)]
    if X.shape[0] < 2:
        n = len(assets)
        return np.eye(n) * 0.01
    lw = LedoitWolf().fit(X)
    return lw.covariance_

def _compute_market_weights(bus, assets, date):
    """自由流通市值权重"""
    caps = []
    for asset in assets:
        cap = bus.get_free_float_market_cap(asset, date)
        caps.append(max(cap, 0.0))
    total = sum(caps)
    if total <= 0:
        return np.ones(len(assets)) / len(assets)
    return np.array(caps) / total

# ---------- 核心步骤（保持不变，但需要返回更多信息） ----------
def step_m_1_directional_mask(context: dict, date: datetime) -> dict:
    """返回方向掩码字典。"""
    assets = context['assets']
    clf = context.get('direction_classifier')
    gamma = context.get('gamma_star', 0.5)
    if clf is None:
        raise RuntimeError("direction_classifier 未找到。")
    borrowable = _fetch_borrowable_stocks(date)
    masks = {}
    for sym in assets:
        feat = _get_features_for_date(sym, date, context)
        if feat is None:
            masks[sym] = 0
            continue
        prob = clf.predict_proba(feat.reshape(1, -1))[0]
        prob_neg, prob_zero, prob_pos = prob[0], prob[1], prob[2]
        if prob_pos >= gamma and prob_pos > prob_neg:
            masks[sym] = 1
        elif prob_neg >= gamma and prob_neg > prob_pos:
            if sym in borrowable:
                masks[sym] = -1
            else:
                masks[sym] = 0
        else:
            masks[sym] = 0
    context['directional_symbol_masks'] = masks
    return masks

def step_m_2_black_litterman_fusion(context: dict, date: datetime, prev_weights: Optional[np.ndarray] = None):
    """
    返回 (R_BL, Sigma_robust, q_low, q_mid, q_high, Omega_diag, Q_view)
    同时更新上下文中的历史宽度等。
    """
    bus = context['data_bus']
    assets = context['assets']
    n = len(assets)
    masks = context.get('directional_symbol_masks', {})
    quant_models = context.get('quantile_models')
    if quant_models is None:
        raise RuntimeError("quantile_models 未找到。")
    scaler = context.get('feature_scaler')
    config = context.get('config', {})
    tau = config.get('tau_BL', 0.02)
    omega_min = config.get('omega_min', 1e-8)
    omega_max = config.get('omega_max', 0.01)
    halflife = 21
    if 'width_history' not in context:
        context['width_history'] = {sym: [] for sym in assets}
    if 'smoothed_width' not in context:
        context['smoothed_width'] = {}

    q_low = np.zeros(n)
    q_mid = np.zeros(n)
    q_high = np.zeros(n)
    Q_view = np.zeros(n)
    Omega_diag = np.zeros(n)

    for i, sym in enumerate(assets):
        feat = _get_features_for_date(sym, date, context)
        if feat is None:
            q_low[i] = q_mid[i] = q_high[i] = 0.0
            Q_view[i] = 0.0
            Omega_diag[i] = omega_max
            continue
        X = feat.reshape(1, -1)
        pred_low = quant_models[0.025].predict(X)[0]
        pred_mid = quant_models[0.5].predict(X)[0]
        pred_high = quant_models[0.975].predict(X)[0]
        pred_low = min(pred_low, pred_mid)
        pred_high = max(pred_high, pred_mid)
        err_thresh = context.get('q_error_threshold_dict', {}).get(sym, 0.0)
        q_low[i] = pred_low - err_thresh
        q_mid[i] = pred_mid
        q_high[i] = pred_high + err_thresh

        width = q_high[i] - q_low[i]
        context['width_history'][sym].append(width)
        if len(context['width_history'][sym]) > 252 * 2:
            context['width_history'][sym] = context['width_history'][sym][-252:]
        alpha = 1 - 0.5 ** (1 / halflife)
        prev_smooth = context['smoothed_width'].get(sym, width)
        smoothed = alpha * width + (1 - alpha) * prev_smooth
        context['smoothed_width'][sym] = smoothed
        omega_ii = np.clip((smoothed ** 2) * tau, omega_min, omega_max)
        Omega_diag[i] = omega_ii

        short_rate = bus.get_short_rate(sym) if masks.get(sym, 0) == -1 else 0.0
        Q_view[i] = q_mid[i] - short_rate

    Sigma_robust = _compute_robust_covariance(bus, assets, date, lookback=252)
    w_mkt = _compute_market_weights(bus, assets, date)
    lambda_mkt = bus.compute_market_risk_aversion(date.strftime('%Y-%m-%d'))
    Pi = lambda_mkt * (Sigma_robust @ w_mkt)
    Omega = np.diag(Omega_diag)
    inv_prior = np.linalg.inv(tau * Sigma_robust)
    inv_omega = np.linalg.inv(Omega)
    P = np.eye(n)
    R_BL = np.linalg.inv(inv_prior + P.T @ inv_omega @ P) @ (inv_prior @ Pi + P.T @ inv_omega @ Q_view)

    # 将计算结果存入上下文（供优化使用）
    context['Sigma_robust'] = Sigma_robust
    context['Pi'] = Pi
    context['R_BL'] = R_BL
    context['Omega_diag'] = Omega_diag
    context['Q_view'] = Q_view

    return R_BL, Sigma_robust, q_low, q_high

def step_m_3_convex_optimization(context: dict, date: datetime, prev_weights: Optional[np.ndarray] = None):
    """返回目标权重向量。"""
    n = len(context['assets'])
    R_BL = context.get('R_BL')
    Sigma = context.get('Sigma_robust')
    masks = context.get('directional_symbol_masks', {})
    config = context.get('config', {})
    borrow_liquidity = context.get('borrowable_today', set())
    if R_BL is None or Sigma is None:
        raise RuntimeError("缺少 R_BL 或 Sigma。")
    gamma_risk = config.get('gamma_risk_initial', 2.5)
    max_leverage = config.get('max_leverage', 1.0)
    sector_limit = config.get('sector_limit', 0.3)
    stock_cap_long = config.get('stock_cap_pct', 0.045)
    stock_cap_short = config.get('stock_cap_short', 0.25)
    eps = config.get('epsilon', 0.001)
    trans_cost = config.get('transaction_cost_coeff', 0.0003)
    if prev_weights is None:
        w_prev = np.zeros(n)
    else:
        w_prev = np.array(prev_weights)

    w = cp.Variable(n)
    utility = w.T @ R_BL - (gamma_risk / 2) * cp.quad_form(w, Sigma) - trans_cost * cp.sum(cp.abs(w - w_prev))
    constraints = []
    constraints.append(cp.sum(cp.abs(w)) <= max_leverage)
    for i, sym in enumerate(context['assets']):
        constraints.append(w[i] <= stock_cap_long)
        if sym in borrow_liquidity:
            constraints.append(w[i] >= -stock_cap_short)
        else:
            constraints.append(w[i] >= 0.0)
    sector_map = context.get('sector_map')
    if sector_map is None:
        bus = context['data_bus']
        sector_map = {}
        for sym in context['assets']:
            sector_map[sym] = bus.get_sector(sym)
        context['sector_map'] = sector_map
    sectors = set(sector_map.values())
    for sec in sectors:
        idx = [i for i, sym in enumerate(context['assets']) if sector_map.get(sym) == sec]
        if idx:
            constraints.append(cp.sum(w[idx]) <= sector_limit)

    prob = cp.Problem(cp.Maximize(utility), constraints)
    prob.solve(solver=cp.ECOS, verbose=False)
    if w.value is None:
        logger.error("凸优化求解失败，使用上一期权重。")
        final_w = w_prev.copy()
    else:
        final_w = w.value.flatten()
        final_w[np.abs(final_w) < eps] = 0.0
    return final_w

# ---------- 新的 execute：预计算整个 Test 集 ----------
def execute(pipeline_context: dict) -> dict:
    logger.info("=" * 60)
    logger.info("Phase 6: 多空双向不确定性头寸分配与凸优化（预计算全 Test 集）")
    logger.info("=" * 60)

    # 确保必要字段存在
    if 'assets' not in pipeline_context or not pipeline_context['assets']:
        raise ValueError("上下文中缺少 assets。")
    slices = pipeline_context.get('slices', {})
    test_dates = slices.get('Test', [])
    if not test_dates:
        raise ValueError("Test 集为空，无法计算权重。")

    assets = pipeline_context['assets']
    n_assets = len(assets)
    # 初始化收集列表
    weight_records = []
    interval_records = []
    prev_weights = None

    for date in test_dates:
        # 更新当前日期
        pipeline_context['current_date'] = date
        # 步骤 M.1-M.3
        masks = step_m_1_directional_mask(pipeline_context, date)
        pipeline_context['directional_symbol_masks'] = masks
        pipeline_context['borrowable_today'] = _fetch_borrowable_stocks(date)
        R_BL, Sigma, q_low, q_high = step_m_2_black_litterman_fusion(pipeline_context, date, prev_weights)
        weights = step_m_3_convex_optimization(pipeline_context, date, prev_weights)

        # 记录权重和区间
        weight_records.append(weights)
        interval_records.append((q_low, q_high))
        prev_weights = weights

    # 构建 DataFrame
    weight_df = pd.DataFrame(weight_records, index=test_dates, columns=assets)
    # 构建区间 DataFrame，存储 q_low 和 q_high 两列，每列是 (日期, 资产) 的多级索引？但为了方便，我们存为两个 DataFrame 或一个 MultiIndex。
    # 为简化，存两个 DataFrame: q_low_df, q_high_df
    q_low_list = []
    q_high_list = []
    for low, high in interval_records:
        q_low_list.append(low)
        q_high_list.append(high)
    q_low_df = pd.DataFrame(q_low_list, index=test_dates, columns=assets)
    q_high_df = pd.DataFrame(q_high_list, index=test_dates, columns=assets)

    # 存入上下文
    pipeline_context['daily_weights'] = weight_df
    pipeline_context['daily_intervals'] = {'q_low': q_low_df, 'q_high': q_high_df}

    # 保留最后一天的目标权重（供后续使用）
    pipeline_context['target_weights'] = {assets[i]: weights[i] for i in range(n_assets)}
    pipeline_context['allocation_weights_ready'] = True
    logger.info(f"Step 6 预计算完成，共 {len(test_dates)} 个交易日。")
    return pipeline_context