# -*- coding: utf-8 -*-
"""
step5/step_5_5_calibration.py
业务模块：在 Train-B1 上优化方向行权门槛 gamma*，并在 Train-B2 上 100% 独立基于A股节点校准 CQR 误差分位数。
"""
import logging
import numpy as np
import pandas as pd
from .config import GAMMA_GRID, ERROR_THRESHOLD_WINDOW, ERROR_MIN_SAMPLES, TAU_BL

logger = logging.getLogger("ModelTraining.Calibration")

def run_cascade_calibration(context: dict):
    logger.info("[Step 5.5] Calibrating gamma* and CQR error thresholds 100% independently for target market...")
    config = context.get('config', {})
    gamma_grid = config.get('train_b1_grid_gamma', GAMMA_GRID)
    error_window = config.get('error_threshold_window', ERROR_THRESHOLD_WINDOW)
    error_min_samples = config.get('error_min_samples', ERROR_MIN_SAMPLES)
    tau_BL = config.get('tau_BL', TAU_BL)

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

    from .math_utils import compute_whitebox_features, fractional_diff_series

    raw_all_dates = sorted(set(context['train_a_dates'] + b1_dates + b2_dates))
    master_timeline = pd.DatetimeIndex(raw_all_dates)
    start_dt = master_timeline[0]
    end_dt = master_timeline[-1]
    
    asset_raw_feat = {}
    T_all, N, F = len(master_timeline), len(assets), 5
    
    for sym in assets:
        df = bus.load_asset_history(sym, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            df = df[~df.index.duplicated(keep='first')]
            
            df_sub = df.reindex(master_timeline).ffill().fillna(0.0)
            asset_raw_feat[sym] = compute_whitebox_features(df_sub)
        else:
            from .step_5_1_cv import diagnose_missing_data
            audit_reason = diagnose_missing_data(bus, sym, start_dt, end_dt)
            logger.warning(f"资产 {sym} 校准阶段特征生成中断 -> {audit_reason}")
            asset_raw_feat[sym] = np.zeros((T_all, F))

    X_cube = np.zeros((T_all, N, F))
    for j, sym in enumerate(assets):
        X_cube[:, j, :] = asset_raw_feat[sym]

    diff_cube = np.zeros_like(X_cube)
    for j in range(N):
        for f in range(F):
            diff_cube[:, j, f] = fractional_diff_series(X_cube[:, j, f], d)

    date_to_idx = {dt: i for i, dt in enumerate(master_timeline)}
    
    b1_indices = [date_to_idx[pd.Timestamp(dt)] for dt in b1_dates if pd.Timestamp(dt) in date_to_idx]
    b2_indices = [date_to_idx[pd.Timestamp(dt)] for dt in b2_dates if pd.Timestamp(dt) in date_to_idx]

    # -------- 1. 方向多分类决策概率阈值优化 (Train-B1) --------
    X_b1, y_b1 = [], []
    y_clf_all = context.get('y_clf_all', {})
    
    for idx in b1_indices:
        dt = master_timeline[idx]
        dt_str = dt.strftime("%Y-%m-%d")
        for j in range(N):
            feat = diff_cube[idx, j, selected]
            if np.isnan(feat).any(): continue
            sym = assets[j]
            
            # 刚性约束：买入决策特征必须 100% 局限在 A 股节点资产
            if bus.get_node_by_asset(sym) != "A_share_node":
                continue
                
            yc = None
            for key in [(dt, sym), (dt_str, sym)]:
                if key in y_clf_all:
                    yc = y_clf_all[key]
                    break
            if yc is not None:
                X_b1.append(feat)
                y_b1.append(yc)
                
    if X_b1:
        X_b1_scaled = scaler.transform(np.vstack(X_b1))
        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(X_b1_scaled)
            prob_pos, prob_neg = probs[:, 2], probs[:, 0]
            best_gamma, best_win_rate = 0.5, -1
            
            for gamma in gamma_grid:
                pred = np.zeros(len(y_b1))
                pred[(prob_pos >= gamma) & (prob_pos > prob_neg)] = 1
                pred[(prob_neg >= gamma) & (prob_neg > prob_pos)] = -1
                win_rate = np.mean(pred == np.array(y_b1))
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_gamma = gamma
            context['gamma_star'] = best_gamma
        else:
            context['gamma_star'] = 0.5
        logger.info(f"[Calib] Optimized gamma* = {context.get('gamma_star', 0.5):.3f}")
        context['num_trials'] = context.get('num_trials', 0) + len(gamma_grid)
    else:
        context['gamma_star'] = 0.5
        logger.warning("Train-B1 validation data missing. Fallback gamma* = 0.5")

    # -------- 2. CQR Conformal 误差经验非对称分位数分布核算 (Train-B2) --------
    error_dict = {sym: [] for sym in assets}
    y_reg_all = context.get('y_reg_all', {})
    
    for idx in b2_indices:
        dt = master_timeline[idx]
        dt_str = dt.strftime("%Y-%m-%d")
        for j, sym in enumerate(assets):
            feat = diff_cube[idx, j, selected]
            if np.isnan(feat).any(): continue
            X_single = scaler.transform(feat.reshape(1, -1))
            
            q_low = quant_models[0.025].predict(X_single)[0]
            q_mid = quant_models[0.5].predict(X_single)[0]
            q_high = quant_models[0.975].predict(X_single)[0]
            
            # 刚性单调性修复
            q_low, q_high = min(q_low, q_mid), max(q_high, q_mid)
            
            y_true = None
            for key in [(dt, sym), (dt_str, sym)]:
                if key in y_reg_all:
                    y_true = y_reg_all[key]
                    break
                    
            if y_true is not None:
                error = max(q_low - y_true, y_true - q_high, 0.0)
                error_dict[sym].append(error)

    error_thresholds = {}
    for sym in assets:
        errors = error_dict.get(sym, [])
        if len(errors) >= error_min_samples:
            window = min(error_window, len(errors))
            recent = errors[-window:]
            error_thresholds[sym] = np.percentile(recent, 95) if recent else 0.0
        else:
            # 【Flow-Pro 5.5 硬红线】全量提取纯净目标域（A股节点）外生误差中位数作为灾备兜底
            all_errors = [e for s, errs in error_dict.items() if bus.get_node_by_asset(s) == "A_share_node" for e in errs]
            median_err = np.median(all_errors) if all_errors else 0.01
            error_thresholds[sym] = median_err
            logger.info(f"[Degrade] Asset {sym} sample size deficient. Injected field median: {median_err:.4f}")

    context['q_error_threshold_dict'] = error_thresholds
    context['tau_BL'] = tau_BL
    logger.info("[Step 5.5] Cascade calibration block executed 100% compliant with target domain outer loop rules.")
    