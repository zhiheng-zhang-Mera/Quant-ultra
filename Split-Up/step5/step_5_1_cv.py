# -*- coding: utf-8 -*-
"""
step5/step_5_1_cv.py
业务模块：在 Train-A 区间上执行时间同步联邦时序交叉验证（Purged Walk-Forward CV），搜索最优记忆阶数 d*，并嵌入负迁移拦截与熔断防御机制。
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
from sklearn.metrics import accuracy_score, mean_squared_error
from .config import BASE_LGB_PARAMS, D_MIN_SEARCH_SPACE, CV_FOLDS, NEGATIVE_TRANSFER_PATIENCE
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
    logger.info("[Step 5.1] Running Federated Time-Synchronized Walk-Forward CV for optimal d...")
    
    config = context.get('config', {})
    d_min_search = config.get('d_min_search', D_MIN_SEARCH_SPACE)
    cv_folds = config.get('cv_folds', CV_FOLDS)
    lgb_params_base = config.get('lgb_params', BASE_LGB_PARAMS)
    patience = config.get('negative_transfer_patience', NEGATIVE_TRANSFER_PATIENCE)

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
    
    # 根据分布式边缘分区拆分资产池
    a_share_assets = [sym for sym in assets if bus.get_node_by_asset(sym) == "A_share_node"]
    us_share_assets = [sym for sym in assets if bus.get_node_by_asset(sym) == "US_share_node"]
    
    for sym in assets:
        df = bus.load_asset_history(sym, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep='first')]
            
            df_sub = df.reindex(master_timeline).ffill().fillna(0.0)
            asset_feature_matrices[sym] = compute_whitebox_features(df_sub)
        else:
            audit_reason = diagnose_missing_data(bus, sym, start_date, end_date)
            logger.warning(f"资产 {sym} 缺失诊断 -> {audit_reason}")
            asset_feature_matrices[sym] = np.zeros((T, F))

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

    fold_size = T // cv_folds if cv_folds > 0 else 1
    best_score = -np.inf
    best_d = 0.0
    best_params = lgb_params_base.copy()
    total_trials = len(d_min_search) * cv_folds
    
    monitor = context.get("negative_transfer_monitor", {"consecutive_violation_count": 0, "triggered_melt": False})

    for d in d_min_search:
        logger.info(f"  Federated evaluating candidate memory d = {d:.2f} ...")
        X_cube = np.zeros((T, N, F))
        for j, sym in enumerate(assets):
            X_cube[:, j, :] = asset_feature_matrices[sym]
            
        diff_cube = np.zeros_like(X_cube)
        for j in range(N):
            for f in range(F):
                diff_cube[:, j, f] = fractional_diff_series(X_cube[:, j, f], d)

        scores = []
        violation_count = 0
        
        for fold in range(cv_folds):
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size
            if fold == 0:
                train_idx = list(range(val_end, T))
            elif fold == cv_folds - 1:
                train_idx = list(range(0, val_start))
            else:
                train_idx = list(range(0, val_start)) + list(range(val_end, T))
            val_idx = list(range(val_start, val_end))

            a_indices = [j for j, sym in enumerate(assets) if sym in a_share_assets]
            us_indices = [j for j, sym in enumerate(assets) if sym in us_share_assets]

            def extract_samples(t_indices, asset_indices):
                X_list, yc_list = [], []
                for t in t_indices:
                    for j in asset_indices:
                        feat = diff_cube[t, j, :]
                        yc = y_clf_matrix[t, j]
                        if not np.isnan(feat).any() and not np.isnan(yc):
                            X_list.append(feat)
                            yc_list.append(yc)
                if X_list:
                    return np.vstack(X_list), np.array(yc_list)
                return None, None

            X_tr_us, yc_tr_us = extract_samples(train_idx, us_indices)
            X_tr_a, yc_tr_a = extract_samples(train_idx, a_indices)
            X_val_a, yc_val_a = extract_samples(val_idx, a_indices)

            if X_tr_a is None or X_val_a is None:
                continue

            scaler = StandardScaler()
            X_tr_a_scaled = scaler.fit_transform(X_tr_a)
            X_val_a_scaled = scaler.transform(X_val_a)

            # 轨 1：纯 A 股本地模型基线独立拟合轨
            params_base = lgb_params_base.copy()
            params_base.update({'objective': 'multiclass', 'num_class': 3})
            clf_base = lgb.LGBMClassifier(**params_base)
            clf_base.fit(X_tr_a_scaled, yc_tr_a)
            base_loss = mean_squared_error(yc_val_a, clf_base.predict(X_val_a_scaled))

            # 轨 2：跨市场联邦迁移微调轨
            if X_tr_us is not None:
                X_tr_us_scaled = scaler.transform(X_tr_us)
                clf_us = lgb.LGBMClassifier(**params_base)
                clf_us.fit(X_tr_us_scaled, yc_tr_us)
                
                # 级联微调特征结构对齐
                us_pred_tr = clf_us.predict_proba(X_tr_a_scaled)
                X_tr_tuned = np.hstack([X_tr_a_scaled, us_pred_tr])
                X_val_tuned = np.hstack([X_val_a_scaled, clf_us.predict_proba(X_val_a_scaled)])
                
                clf_transfer = lgb.LGBMClassifier(**lgb_params_base)
                clf_transfer.set_params(objective='multiclass', num_class=3)
                clf_transfer.fit(X_tr_tuned, yc_tr_a)
                transfer_loss = mean_squared_error(yc_val_a, clf_transfer.predict(X_val_tuned))
            else:
                transfer_loss = base_loss

            # 负迁移检测红线审计
            if transfer_loss > base_loss:
                violation_count += 1
                
            acc = accuracy_score(yc_val_a, clf_base.predict(X_val_a_scaled))
            scores.append(acc)

        if scores:
            avg_score = np.mean(scores)
            if avg_score > best_score:
                best_score = avg_score
                best_d = d
                
        if violation_count >= patience:
            monitor["consecutive_violation_count"] += 1
            if monitor["consecutive_violation_count"] >= patience:
                monitor["triggered_melt"] = True
                logger.warning("🚨 [负迁移硬红线熔断] 迁移学习验证集损失连续超标，一键物理安全回退激活！")

    logger.info(f"[CV] Selection Finished. Optimal d* = {best_d:.2f}, Mean Joint Score = {best_score:.4f}")
    context['best_d'] = best_d
    context['best_lgb_params'] = best_params
    context['negative_transfer_monitor'] = monitor
    
    param_hash = hashlib.sha256(json.dumps(best_params, sort_keys=True).encode()).hexdigest()
    context['param_hash'] = param_hash
    with open("param_audit.log", "a") as f:
        f.write(f"{datetime.now().isoformat()} d={best_d} hash={param_hash}\n")

    context['num_trials'] = context.get('num_trials', 0) + total_trials