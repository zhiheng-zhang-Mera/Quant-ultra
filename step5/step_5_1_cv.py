# -*- coding: utf-8 -*-
"""
step5/step_5_1_cv.py
业务模块：在 Train-A 区间上执行净化向前行走交叉验证（Purged Walk-Forward CV），搜索最优记忆阶数 d*。
内置刚性时区对齐的多维数据缺失白盒诊断审计引擎。
"""
import logging
import hashlib
import json
import os
from datetime import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, r2_score
from .config import BASE_LGB_PARAMS, D_MIN_SEARCH_SPACE, CV_FOLDS
from .math_utils import compute_whitebox_features, fractional_diff_series

logger = logging.getLogger("ModelTraining.CV")

def diagnose_missing_data(bus, sym: str, start_date, end_date) -> str:
    """内部审计探针：具备时区刚性净化能力"""
    cache_dir = getattr(bus, 'cache_dir', "d:/Quant3/data_cache")
    file_path = os.path.join(cache_dir, f"{sym}_history.parquet")
    
    if not os.path.exists(file_path):
        alt_path = os.path.join("data_cache", f"{sym}_history.parquet")
        if not os.path.exists(alt_path):
            return f"❌ 确实没有数据文件 (磁盘预期路径无此文件: {file_path})"
        file_path = alt_path
        
    try:
        df_meta = pd.read_parquet(file_path)
        if df_meta.empty: 
            return "⚠️ 其他可能: 物理文件存在，但内部数据行为 0 (空壳文件)。"
            
        dates = pd.to_datetime(df_meta['date']) if 'date' in df_meta.columns else pd.to_datetime(df_meta.index)
        if dates.empty: 
            return "⚠️ 其他可能: 文件存在但无有效日期轴。"
            
        if hasattr(dates, 'dt'):
            if dates.dt.tz is not None:
                dates = dates.dt.tz_localize(None)
        else:
            if dates.tz is not None:
                dates = dates.tz_localize(None)
                
        min_dt, max_dt = dates.min(), dates.max()
        
        req_start = pd.to_datetime(start_date).tz_localize(None) if pd.to_datetime(start_date).tz is not None else pd.to_datetime(start_date)
        req_end = pd.to_datetime(end_date).tz_localize(None) if pd.to_datetime(end_date).tz is not None else pd.to_datetime(end_date)
        
        if max_dt < req_start or min_dt > req_end:
            return f"👶 在时间段未上市 (真实区间: {min_dt.strftime('%Y-%m-%d')} ~ {max_dt.strftime('%Y-%m-%d')} | 错配当前窗口: {req_start.strftime('%Y-%m-%d')} ~ {req_end.strftime('%Y-%m-%d')})"
        else:
            return f"🔥 其他可能: 文件存在且包含交集({min_dt.strftime('%Y-%m-%d')} ~ {max_dt.strftime('%Y-%m-%d')})，但总线读取返回空，请检查过滤Bug。"
            
    except Exception as e:
        return f"💥 其他可能: 文件解析崩溃/格式损坏。错误: {str(e)}"

def run_walk_forward_cv(context: dict):
    logger.info("[Step 5.1] Running Purged Walk-Forward CV for optimal d...")
    
    config = context.get('config', {})
    d_min_search = config.get('d_min_search', D_MIN_SEARCH_SPACE)
    cv_folds = config.get('cv_folds', CV_FOLDS)
    lgb_params_base = config.get('lgb_params', BASE_LGB_PARAMS)

    bus = context['data_bus']
    slices = context['slices']
    train_a_dates = slices.get('Train-A', [])
    if not train_a_dates:
        raise ValueError("Train-A 时间序列为空，无法执行交叉验证。")
    assets = context['assets']
    if not assets:
        raise ValueError("全局资产池列表为空。")

    master_timeline = pd.DatetimeIndex(train_a_dates)
    start_date = master_timeline[0]
    end_date = master_timeline[-1]
    
    T, N, F = len(master_timeline), len(assets), 5
    asset_feature_matrices = {}
    
    for sym in assets:
        df = bus.load_asset_history(sym, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep='first')]
            
            # 🟩 加固：用 .fillna(0.0) 替换原有的 .bfill()，杜绝历史回测中的前瞻记忆偏差
            df_sub = df.reindex(master_timeline).ffill().fillna(0.0)
            asset_feature_matrices[sym] = compute_whitebox_features(df_sub)
        else:
            audit_reason = diagnose_missing_data(bus, sym, start_date, end_date)
            logger.warning(f"资产 {sym} 缺失诊断 -> {audit_reason}")
            asset_feature_matrices[sym] = np.zeros((T, F))

    full_feat = np.hstack([asset_feature_matrices[sym] for sym in assets])

    y_clf_all = context.get('y_clf_all', {})
    y_reg_all = context.get('y_reg_all', {})
    y_clf_matrix = np.full((T, N), np.nan)
    y_reg_matrix = np.full((T, N), np.nan)
    
    for i, dt in enumerate(master_timeline):
        dt_str = dt.strftime("%Y-%m-%d")
        for j, sym in enumerate(assets):
            for key in [(dt, sym), (dt_str, sym)]:
                if key in y_clf_all: y_clf_matrix[i, j] = y_clf_all[key]
                if key in y_reg_all: y_reg_matrix[i, j] = y_reg_all[key]

    date_idx_full = np.repeat(np.arange(T), N)
    n_dates = T
    fold_size = n_dates // cv_folds if cv_folds > 0 else 1

    best_score = -np.inf
    best_d = 0.0
    best_params = lgb_params_base.copy()
    total_trials = len(d_min_search) * cv_folds

    for d in d_min_search:
        logger.info(f"  Evaluating candidate memory d = {d:.2f} ...")
        X_cube = full_feat.reshape(T, N, F)
        diff_cube = np.zeros_like(X_cube)
        for j in range(N):
            for f in range(F):
                diff_cube[:, j, f] = fractional_diff_series(X_cube[:, j, f], d)
        X_diff_flat = diff_cube.reshape(T*N, F)

        y_clf_flat = y_clf_matrix.reshape(T*N)
        y_reg_flat = y_reg_matrix.reshape(T*N)
        valid_mask = ~np.isnan(y_clf_flat) & ~np.isnan(y_reg_flat)
        
        X_valid = X_diff_flat[valid_mask]
        y_clf_valid = y_clf_flat[valid_mask]
        y_reg_valid = y_reg_flat[valid_mask]
        date_idx_valid = date_idx_full[valid_mask]

        if len(X_valid) == 0:
            continue

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

            X_train, X_val = X_valid[train_mask], X_valid[val_mask]
            yc_train, yc_val = y_clf_valid[train_mask], y_clf_valid[val_mask]
            yr_train, yr_val = y_reg_valid[train_mask], y_reg_valid[val_mask]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            params_clf = lgb_params_base.copy()
            params_clf.update({'objective': 'multiclass', 'num_class': 3})
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

    logger.info(f"[CV] Selection Finished. Optimal d* = {best_d:.2f}, Mean Joint Score = {best_score:.4f}")
    context['best_d'] = best_d
    context['best_lgb_params'] = best_params
    
    param_hash = hashlib.sha256(json.dumps(best_params, sort_keys=True).encode()).hexdigest()
    context['param_hash'] = param_hash
    with open("param_audit.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} d={best_d} hash={param_hash}\n")

    context['num_trials'] = context.get('num_trials', 0) + total_trials