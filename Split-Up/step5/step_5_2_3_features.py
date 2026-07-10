# -*- coding: utf-8 -*-
"""
step5/step_5_2_3_features.py
业务模块：基于最优阶数 d* 生成多源时序因果递推特征矩阵，并通过层次聚类和 VIF 算法完成分布式隐私保护特征空间净化。
"""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from .config import VIF_THRESHOLD, CLUSTER_SELECT_RATIO

logger = logging.getLogger("ModelTraining.Features")

def generate_fractional_features(context: dict):
    logger.info("[Step 5.2] Applying fractional differentiation on Train-A to generate memory state.")
    d = context['best_d']
    bus = context['data_bus']
    slices = context['slices']
    train_a_dates = slices.get('Train-A', [])
    assets = context['assets']
    
    master_timeline = pd.DatetimeIndex(train_a_dates)
    start_date = master_timeline[0]
    end_date = master_timeline[-1]

    T, N, F = len(master_timeline), len(assets), 5
    
    diff_cube = np.zeros((T, N, F))
    alive_mask_matrix = np.zeros((T, N), dtype=bool)
    
    from .math_utils import compute_whitebox_features, fractional_diff_series

    for idx, sym in enumerate(assets):
        df = bus.load_asset_history(sym, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
        if df is not None and not df.empty:
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
                
            # 时区安全对齐栅栏
            if master_timeline.tz is not None and df.index.tz is None:
                df.index = df.index.tz_localize(master_timeline.tz)
            elif master_timeline.tz is None and df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            elif master_timeline.tz is not None and df.index.tz is not None:
                df.index = df.index.tz_convert(master_timeline.tz)
                
            df = df[~df.index.duplicated(keep='first')]
            
            sym_alive = (master_timeline >= df.index.min()) & (master_timeline <= df.index.max())
            alive_mask_matrix[:, idx] = sym_alive
            
            df_sub = df.reindex(master_timeline).ffill().fillna(0.0)
            raw_feats = compute_whitebox_features(df_sub)
            
            for f_idx in range(F):
                diff_cube[:, idx, f_idx] = fractional_diff_series(raw_feats[:, f_idx], d)
        else:
            from .step_5_1_cv import diagnose_missing_data  
            audit_reason = diagnose_missing_data(bus, sym, start_date, end_date)
            logger.warning(f"Asset {sym} Train-A特征生成中断 -> {audit_reason}")
            alive_mask_matrix[:, idx] = False
            diff_cube[:, idx, :] = 0.0

    context['fractional_features_cube'] = diff_cube
    context['alive_mask_matrix'] = alive_mask_matrix
    logger.info(f"[Step 5.2] Feature Cube {diff_cube.shape} and Alive Mask successfully injected into context.")
    
def run_feature_filtering(context: dict):
    logger.info("[Step 5.3] Feature filtering via agglomerative clustering and VIF with privacy compression rules.")
    config = context.get('config', {})
    vif_th = config.get('vif_threshold', VIF_THRESHOLD)
    cluster_ratio = config.get('cluster_select_ratio', CLUSTER_SELECT_RATIO)

    diff_cube = context.get('fractional_features_cube')
    alive_mask = context.get('alive_mask_matrix')
    if diff_cube is None:
        raise RuntimeError("Fractional feature cube missing. Run step_5_2 first.")
        
    T, N, F = diff_cube.shape
    X_flat = diff_cube.reshape(T*N, F)
    
    if alive_mask is not None:
        mask_flat = alive_mask.reshape(T*N)
        valid_rows = (~np.isnan(X_flat).any(axis=1)) & mask_flat
    else:
        valid_rows = ~np.isnan(X_flat).any(axis=1)
        
    X = X_flat[valid_rows]
    if X.shape[0] == 0:
        raise RuntimeError("No finite samples available for feature covariance filtering.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    vif = np.zeros(F)
    for i in range(F):
        vif[i] = variance_inflation_factor(X_scaled, i)
    high_vif_idx = np.where(vif > vif_th)[0]
    keep_idx = [i for i in range(F) if i not in high_vif_idx]

    if len(keep_idx) == 0:
        logger.warning("All features exceed VIF ceiling. Rollback to keep all.")
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
                if cum / total_imp >= cluster_ratio:
                    break
        selected = sorted(set(selected))
    else:
        selected = keep_idx

    context['selected_features'] = selected
    logger.info(f"[Step 5.3] Feature subspace locked under Federated distillation constraints. Retained Indices: {selected}")
}