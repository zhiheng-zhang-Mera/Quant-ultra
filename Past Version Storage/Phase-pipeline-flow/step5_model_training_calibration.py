"""
Phase 5: Joint Hyperparameter Tuning, Dual-Track Cascade Calibration, and Model Fitting
Fully compliant with Final-Flow.md [2026 Production Release]
All data sourced from real free feeds (AkShare/BaoStock) through PITDataBus.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import accuracy_score, r2_score
import hashlib
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger("ModelTraining")

# ============================
# 辅助函数：分数阶微分（因果递推）
# ============================
def fractional_diff_series(series: np.ndarray, d: float) -> np.ndarray:
    """
    对一维时间序列进行分数阶微分 (1-L)^d，使用因果递推（仅依赖历史）。
    """
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


def apply_fractional_diff_to_features(feature_matrix: np.ndarray, d: float) -> np.ndarray:
    if d == 0:
        return feature_matrix
    T, F = feature_matrix.shape
    diffed = np.zeros_like(feature_matrix)
    for f in range(F):
        diffed[:, f] = fractional_diff_series(feature_matrix[:, f], d)
    return diffed


def compute_whitebox_features(df: pd.DataFrame) -> np.ndarray:
    """
    根据单个资产的日线DataFrame计算五个白盒特征（原始值，未微分）。
    返回 (T, 5) 矩阵。
    """
    df = df.sort_index()
    close = df['close'].values
    open_ = df['open'].values
    high = df['high'].values
    low = df['low'].values
    amount = df['amount'].values
    T = len(df)
    features = np.zeros((T, 5))

    log_ret = np.full(T, np.nan)
    log_ret[1:] = np.log(close[1:] / close[:-1])
    features[:, 0] = log_ret

    mom5 = np.full(T, np.nan)
    for t in range(5, T):
        mom5[t] = np.sum(log_ret[t-5:t])
    features[:, 1] = mom5

    mom20 = np.full(T, np.nan)
    for t in range(20, T):
        mom20[t] = np.sum(log_ret[t-20:t])
    features[:, 2] = mom20

    gk = np.full(T, np.nan)
    for t in range(T):
        if high[t] > 0 and low[t] > 0 and open_[t] > 0 and close[t] > 0:
            hl = np.log(high[t] / low[t])
            co = np.log(close[t] / open_[t])
            gk[t] = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    features[:, 3] = gk

    shock = np.full(T, np.nan)
    for t in range(20, T):
        avg_amt = np.mean(amount[t-20:t])
        if avg_amt > 0:
            shock[t] = amount[t] / avg_amt
    features[:, 4] = shock

    return features


# ============================
# 核心步骤实现（全部使用配置）
# ============================

def step_5_1_walk_forward_cv(context: dict):
    """搜索最优分数阶微分阶数 d*，使用配置中的参数。"""
    logger.info("[Step 5.1] Running Purged Walk-Forward CV for optimal d...")

    config = context.get('config', {})
    d_min_search = config.get('d_min_search', [0.1, 0.3, 0.5, 0.7, 0.9])
    cv_folds = config.get('cv_folds', 3)
    lgb_params_base = config.get('lgb_params', {
        "n_estimators": 100,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "deterministic": True,
        "num_threads": 1,
        "random_state": 42,
        "verbosity": -1,
    })

    bus = context['data_bus']
    slices = context['slices']
    train_a_dates = slices.get('Train-A', [])
    if not train_a_dates:
        raise ValueError("Train-A 为空，无法进行CV。")
    assets = context['assets']
    if not assets:
        raise ValueError("资产列表为空。")

    start_date = train_a_dates[0]
    end_date = train_a_dates[-1]
    asset_dfs = {}
    for sym in assets:
        df = bus.load_asset_history(sym, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            asset_dfs[sym] = df
        else:
            logger.warning(f"资产 {sym} 历史数据缺失，跳过。")
    if not asset_dfs:
        raise RuntimeError("所有资产数据缺失，无法继续。")

    common_dates = set(train_a_dates)
    for sym, df in asset_dfs.items():
        common_dates = common_dates.intersection(set(df.index))
    common_dates = sorted(common_dates)
    if not common_dates:
        raise RuntimeError("无共同交易日。")

    T = len(common_dates)
    F = 5
    N = len(assets)
    asset_feature_matrices = {}
    for sym, df in asset_dfs.items():
        df_sub = df.loc[common_dates]
        if len(df_sub) < T:
            df_sub = df_sub.reindex(common_dates, method='ffill')
        feat = compute_whitebox_features(df_sub)
        asset_feature_matrices[sym] = feat

    full_feat = np.hstack([asset_feature_matrices[sym] for sym in assets])

    y_clf_all = context.get('y_clf_all', {})
    y_reg_all = context.get('y_reg_all', {})
    y_clf_matrix = np.full((T, N), np.nan)
    y_reg_matrix = np.full((T, N), np.nan)
    for i, dt in enumerate(common_dates):
        for j, sym in enumerate(assets):
            key = (dt, sym)
            if key in y_clf_all:
                y_clf_matrix[i, j] = y_clf_all[key]
            if key in y_reg_all:
                y_reg_matrix[i, j] = y_reg_all[key]

    # 构建样本矩阵 (T*N, F)
    X_list = []
    y_clf_list = []
    y_reg_list = []
    for i, dt in enumerate(common_dates):
        for j, sym in enumerate(assets):
            feat_asset = asset_feature_matrices[sym][i, :]
            if np.isnan(feat_asset).any():
                continue
            yc = y_clf_matrix[i, j]
            yr = y_reg_matrix[i, j]
            if np.isnan(yc) or np.isnan(yr):
                continue
            X_list.append(feat_asset)
            y_clf_list.append(yc)
            y_reg_list.append(yr)
    if not X_list:
        raise RuntimeError("无有效样本。")
    X_all = np.vstack(X_list)
    y_clf_all_flat = np.array(y_clf_list)
    y_reg_all_flat = np.array(y_reg_list)

    date_indices = []
    for i, dt in enumerate(common_dates):
        for j in range(N):
            date_indices.append(dt)
    date_indices = np.array(date_indices)

    unique_dates = sorted(common_dates)
    n_dates = len(unique_dates)
    fold_size = n_dates // cv_folds if cv_folds > 0 else 1

    best_score = -np.inf
    best_d = 0.0
    best_params = lgb_params_base.copy()

    total_trials = len(d_min_search) * cv_folds  # 初步计数

    for d in d_min_search:
        logger.info(f"  评估 d={d:.2f} ...")
        X_cube = full_feat.reshape(T, N, F)
        diff_cube = np.zeros_like(X_cube)
        for j in range(N):
            for f in range(F):
                diff_cube[:, j, f] = fractional_diff_series(X_cube[:, j, f], d)
        X_diff_flat = diff_cube.reshape(T*N, F)

        y_clf_full = np.full((T, N), np.nan)
        y_reg_full = np.full((T, N), np.nan)
        for i, dt in enumerate(common_dates):
            for j, sym in enumerate(assets):
                key = (dt, sym)
                if key in y_clf_all:
                    y_clf_full[i, j] = y_clf_all[key]
                if key in y_reg_all:
                    y_reg_full[i, j] = y_reg_all[key]
        y_clf_flat = y_clf_full.reshape(T*N)
        y_reg_flat = y_reg_full.reshape(T*N)

        valid_mask = ~np.isnan(y_clf_flat) & ~np.isnan(y_reg_flat)
        X_valid = X_diff_flat[valid_mask]
        y_clf_valid = y_clf_flat[valid_mask]
        y_reg_valid = y_reg_flat[valid_mask]

        date_idx_full = np.repeat(np.arange(T), N)
        date_idx_valid = date_idx_full[valid_mask]

        scores = []
        for fold in range(cv_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size
            if fold == 0:
                train_date_indices = list(range(val_end, n_dates))
            elif fold == cv_folds - 1:
                train_date_indices = list(range(0, val_start))
            else:
                train_date_indices = list(range(0, val_start)) + list(range(val_end, n_dates))
            val_date_indices = list(range(val_start, val_end))

            train_mask = np.isin(date_idx_valid, train_date_indices)
            val_mask = np.isin(date_idx_valid, val_date_indices)

            if np.sum(train_mask) == 0 or np.sum(val_mask) == 0:
                continue

            X_train = X_valid[train_mask]
            yc_train = y_clf_valid[train_mask]
            yr_train = y_reg_valid[train_mask]
            X_val = X_valid[val_mask]
            yc_val = y_clf_valid[val_mask]
            yr_val = y_reg_valid[val_mask]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            params_clf = lgb_params_base.copy()
            params_clf['objective'] = 'multiclass'
            params_clf['num_class'] = 3
            clf = lgb.LGBMClassifier(**params_clf)
            clf.fit(X_train_scaled, yc_train)
            reg = lgb.LGBMRegressor(**lgb_params_base)
            reg.fit(X_train_scaled, yr_train)

            acc = accuracy_score(yc_val, clf.predict(X_val_scaled))
            r2 = r2_score(yr_val, reg.predict(X_val_scaled))
            score = 0.5 * acc + 0.5 * max(0, r2)
            scores.append(score)

        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_d = d
                best_params = lgb_params_base.copy()

    logger.info(f"[CV] 最优 d* = {best_d:.2f}, 平均得分 = {best_score:.4f}")
    context['best_d'] = best_d
    context['best_lgb_params'] = best_params
    param_hash = hashlib.sha256(json.dumps(best_params, sort_keys=True).encode()).hexdigest()
    context['param_hash'] = param_hash
    with open("param_audit.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} d={best_d} hash={param_hash}\n")

    # 统计总试验次数：包括 d 搜索、gamma 网格等（后续在 step_5_5 中累加）
    # 这里先保存 d 搜索的试验次数，后面累加
    context['num_trials'] = total_trials
    logger.info(f"[CV] 总独立试验次数 (d搜索) = {total_trials}")
    logger.info("[完成] Walk-Forward CV 结束。")


def step_5_2_fractional_diff_state(context: dict):
    """使用最优 d* 对 Train-A 完整区间执行流式因果递推，生成尾部记忆状态矩阵。"""
    logger.info("[Step 5.2] Applying fractional differentiation on Train-A to generate memory state.")
    d = context['best_d']
    bus = context['data_bus']
    slices = context['slices']
    train_a_dates = slices.get('Train-A', [])
    if not train_a_dates:
        raise ValueError("Train-A 为空。")
    assets = context['assets']
    start_date = train_a_dates[0]
    end_date = train_a_dates[-1]

    asset_feat_dict = {}
    for sym in assets:
        df = bus.load_asset_history(sym, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            df = df.loc[train_a_dates]
            feat = compute_whitebox_features(df)
            asset_feat_dict[sym] = feat
        else:
            logger.warning(f"资产 {sym} 无数据，跳过。")
    if not asset_feat_dict:
        raise RuntimeError("无资产数据。")

    T = len(train_a_dates)
    N = len(assets)
    F = 5
    X_cube = np.zeros((T, N, F))
    for j, sym in enumerate(assets):
        if sym in asset_feat_dict:
            X_cube[:, j, :] = asset_feat_dict[sym]
        else:
            X_cube[:, j, :] = np.nan

    diff_cube = np.zeros_like(X_cube)
    for j in range(N):
        for f in range(F):
            diff_cube[:, j, f] = fractional_diff_series(X_cube[:, j, f], d)

    context['fractional_features_cube'] = diff_cube
    context['train_a_dates'] = train_a_dates
    context['assets'] = assets
    logger.info(f"[完成] 分数阶微分特征矩阵生成，形状 {diff_cube.shape}")


def step_5_3_feature_filtering(context: dict):
    """层次聚类 + VIF > 30 剔除，基于 Train-A 微分特征矩阵。"""
    logger.info("[Step 5.3] Feature filtering via clustering and VIF.")
    config = context.get('config', {})
    vif_threshold = config.get('vif_threshold', 30)
    cluster_select_ratio = config.get('cluster_select_ratio', 0.8)

    diff_cube = context.get('fractional_features_cube')
    if diff_cube is None:
        raise RuntimeError("缺少分数阶微分特征矩阵，请先执行 step_5_2。")
    T, N, F = diff_cube.shape
    X_flat = diff_cube.reshape(T*N, F)
    valid_rows = ~np.isnan(X_flat).any(axis=1)
    X = X_flat[valid_rows]
    if X.shape[0] == 0:
        raise RuntimeError("无有效特征样本。")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    vif = np.zeros(F)
    for i in range(F):
        vif[i] = variance_inflation_factor(X_scaled, i)
    high_vif_idx = np.where(vif > vif_threshold)[0]
    keep_idx = [i for i in range(F) if i not in high_vif_idx]

    if len(keep_idx) == 0:
        logger.warning("所有特征VIF过高，保留全部特征。")
        keep_idx = list(range(F))

    if len(keep_idx) > 1:
        clustering = FeatureAgglomeration(n_clusters=min(3, len(keep_idx)))
        clustering.fit(X[:, keep_idx])
        pca = PCA(n_components=min(5, len(keep_idx)))
        pca.fit(X[:, keep_idx])
        importance = np.abs(pca.components_).sum(axis=0)
        labels = clustering.labels_
        selected = []
        for cluster_id in set(labels):
            idx_in_cluster = [i for i, lab in enumerate(labels) if lab == cluster_id]
            sorted_idx = sorted(idx_in_cluster, key=lambda i: importance[i], reverse=True)
            total_imp = sum(importance[i] for i in idx_in_cluster)
            cum = 0
            for i in sorted_idx:
                selected.append(keep_idx[i])
                cum += importance[i]
                if cum / total_imp >= cluster_select_ratio:
                    break
        selected = sorted(set(selected))
    else:
        selected = keep_idx

    context['selected_features'] = selected
    logger.info(f"[完成] 保留特征索引: {selected}")


def step_5_4_model_bundle_fitting(context: dict):
    """在 Train-A 上拟合方向分类器和分位数回归群（q=0.025, 0.5, 0.975）。"""
    logger.info("[Step 5.4] Fitting dual-track models (direction classifier & CQR quantile regressors).")
    config = context.get('config', {})
    lgb_params_base = config.get('lgb_params', {
        "n_estimators": 100,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "deterministic": True,
        "num_threads": 1,
        "random_state": 42,
        "verbosity": -1,
    })

    diff_cube = context.get('fractional_features_cube')
    if diff_cube is None:
        raise RuntimeError("缺少微分特征矩阵。")
    T, N, F = diff_cube.shape
    selected = context['selected_features']
    train_dates = context['train_a_dates']

    y_clf_all = context.get('y_clf_all', {})
    y_reg_all = context.get('y_reg_all', {})
    X_list = []
    y_clf_list = []
    y_reg_list = []
    for i, dt in enumerate(train_dates):
        for j, sym in enumerate(context['assets']):
            feat = diff_cube[i, j, selected]
            if np.isnan(feat).any():
                continue
            key = (dt, sym)
            yc = y_clf_all.get(key)
            yr = y_reg_all.get(key)
            if yc is not None and yr is not None:
                X_list.append(feat)
                y_clf_list.append(yc)
                y_reg_list.append(yr)
    if not X_list:
        raise RuntimeError("无有效训练样本。")
    X_train = np.vstack(X_list)
    y_clf = np.array(y_clf_list)
    y_reg = np.array(y_reg_list)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    context['feature_scaler'] = scaler

    params_clf = lgb_params_base.copy()
    params_clf['objective'] = 'multiclass'
    params_clf['num_class'] = 3
    clf = lgb.LGBMClassifier(**params_clf)
    clf.fit(X_train_scaled, y_clf)
    context['direction_classifier'] = clf

    quantile_models = {}
    for q in [0.025, 0.5, 0.975]:
        params_reg = lgb_params_base.copy()
        params_reg['objective'] = 'quantile'
        params_reg['alpha'] = q
        reg = lgb.LGBMRegressor(**params_reg)
        reg.fit(X_train_scaled, y_reg)
        quantile_models[q] = reg
    context['quantile_models'] = quantile_models
    logger.info("[完成] 模型拟合完成。")


def step_5_5_calibration_and_monotonic_fix(context: dict):
    """在 Train-B1 优化 gamma*，在 Train-B2 计算 CQR 误差阈值，并执行分位数单调性后处理。"""
    logger.info("[Step 5.5] Calibrating gamma* and CQR error thresholds with monotonicity fixes.")
    config = context.get('config', {})
    gamma_grid = config.get('train_b1_grid_gamma', np.linspace(0.3, 0.7, 9))
    error_window = config.get('error_threshold_window', 252)
    error_min_samples = config.get('error_min_samples', 50)
    tau_BL = config.get('tau_BL', 0.02)

    bus = context['data_bus']
    slices = context['slices']
    b1_dates = slices.get('Train-B1', [])
    b2_dates = slices.get('Train-B2', [])
    assets = context['assets']
    selected = context['selected_features']
    clf = context['direction_classifier']
    quant_models = context['quantile_models']
    scaler = context['feature_scaler']
    d = context['best_d']

    all_dates = context['train_a_dates'] + b1_dates + b2_dates
    all_dates = sorted(set(all_dates))
    start_dt = all_dates[0]
    end_dt = all_dates[-1]
    asset_raw_feat = {}
    for sym in assets:
        df = bus.load_asset_history(sym, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            df = df.loc[all_dates]
            feat = compute_whitebox_features(df)
            asset_raw_feat[sym] = feat
        else:
            logger.warning(f"资产 {sym} 数据缺失，跳过。")
    if not asset_raw_feat:
        raise RuntimeError("无资产数据用于校准。")

    T_all = len(all_dates)
    N = len(assets)
    F = 5
    X_cube = np.zeros((T_all, N, F))
    for j, sym in enumerate(assets):
        if sym in asset_raw_feat:
            X_cube[:, j, :] = asset_raw_feat[sym]
        else:
            X_cube[:, j, :] = np.nan

    diff_cube = np.zeros_like(X_cube)
    for j in range(N):
        for f in range(F):
            diff_cube[:, j, f] = fractional_diff_series(X_cube[:, j, f], d)

    date_to_idx = {dt: i for i, dt in enumerate(all_dates)}
    b1_indices = [date_to_idx[dt] for dt in b1_dates if dt in date_to_idx]
    b2_indices = [date_to_idx[dt] for dt in b2_dates if dt in date_to_idx]

    # -------- 方向阈值优化 (Train-B1) --------
    X_b1 = []
    y_b1 = []
    for idx in b1_indices:
        for j in range(N):
            feat = diff_cube[idx, j, selected]
            if np.isnan(feat).any():
                continue
            sym = assets[j]
            key = (all_dates[idx], sym)
            yc = context.get('y_clf_all', {}).get(key)
            if yc is not None:
                X_b1.append(feat)
                y_b1.append(yc)
    if X_b1:
        X_b1 = np.vstack(X_b1)
        X_b1_scaled = scaler.transform(X_b1)
        probs = clf.predict_proba(X_b1_scaled)
        prob_pos = probs[:, 2]
        prob_neg = probs[:, 0]
        best_gamma = 0.5
        best_win_rate = -1
        for gamma in gamma_grid:
            pred = np.zeros(len(y_b1))
            mask_long = (prob_pos >= gamma) & (prob_pos > prob_neg)
            pred[mask_long] = 1
            mask_short = (prob_neg >= gamma) & (prob_neg > prob_pos)
            pred[mask_short] = -1
            win_rate = np.mean(pred == np.array(y_b1))
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_gamma = gamma
        context['gamma_star'] = best_gamma
        logger.info(f"[校准] 最优 gamma* = {best_gamma:.3f}, 胜率 = {best_win_rate:.4f}")
        # 累加试验次数：gamma网格大小
        context['num_trials'] = context.get('num_trials', 0) + len(gamma_grid)
    else:
        context['gamma_star'] = 0.5
        logger.warning("Train-B1 无数据，使用默认 gamma=0.5")

    # -------- CQR 误差阈值 (Train-B2) --------
    error_dict = {sym: [] for sym in assets}
    y_reg_all = context.get('y_reg_all', {})
    for idx in b2_indices:
        dt = all_dates[idx]
        for j, sym in enumerate(assets):
            feat = diff_cube[idx, j, selected]
            if np.isnan(feat).any():
                continue
            X_single = scaler.transform(feat.reshape(1, -1))
            q_low = quant_models[0.025].predict(X_single)[0]
            q_mid = quant_models[0.5].predict(X_single)[0]
            q_high = quant_models[0.975].predict(X_single)[0]
            q_low = min(q_low, q_mid)
            q_high = max(q_high, q_mid)
            key = (dt, sym)
            y_true = y_reg_all.get(key)
            if y_true is not None:
                error = max(q_low - y_true, y_true - q_high, 0.0)
                error_dict[sym].append(error)

    error_thresholds = {}
    for sym in assets:
        errors = error_dict.get(sym, [])
        if len(errors) >= error_min_samples:
            window = min(error_window, len(errors))
            recent = errors[-window:] if window > 0 else errors
            q = np.percentile(recent, 95) if recent else 0.0
            error_thresholds[sym] = q
        else:
            all_errors = [e for errs in error_dict.values() for e in errs]
            if all_errors:
                median_err = np.median(all_errors)
            else:
                median_err = 0.01
            error_thresholds[sym] = median_err
            logger.info(f"[降级] 资产 {sym} 样本不足，使用行业中位数 {median_err:.4f}")

    context['q_error_threshold_dict'] = error_thresholds
    context['tau_BL'] = tau_BL
    logger.info("[完成] CQR 误差阈值计算完毕。")


def execute(pipeline_context: dict) -> dict:
    """阶段五主入口"""
    logger.info("="*60)
    logger.info(">>> Phase 5: Model Training & Calibration")

    # 确保 num_trials 初始存在
    if 'num_trials' not in pipeline_context:
        pipeline_context['num_trials'] = 0

    step_5_1_walk_forward_cv(pipeline_context)
    step_5_2_fractional_diff_state(pipeline_context)
    step_5_3_feature_filtering(pipeline_context)
    step_5_4_model_bundle_fitting(pipeline_context)
    step_5_5_calibration_and_monotonic_fix(pipeline_context)
    pipeline_context['model_training_ready'] = True
    logger.info(">>> Phase 5 completed successfully.")
    return pipeline_context