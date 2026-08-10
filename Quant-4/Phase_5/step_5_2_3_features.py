# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 5.2 & 5.3: Fractional Cube Generation & Multi-Collinearity Distiller Engine
"""
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import FeatureAgglomeration
from sklearn.decomposition import PCA
from statsmodels.stats.outliers_influence import variance_inflation_factor
from Phase_5.config import VIF_THRESHOLD, CLUSTER_SELECT_RATIO

logger = logging.getLogger("ModelTraining.Features")


def apply_rolling_zscore(cube: np.ndarray, lookback: int = 60, min_periods: int = 20) -> np.ndarray:
    """As-of rolling z-score per (asset, feature) along the time axis.

    8-9 计划任务四：在特征面板生成阶段引入自适应滚动标准差标准化。
    仅使用截至当日的历史窗口统计量（rolling window ending at t），不读取未来数据，
    因此相比全样本 StandardScaler 既降低跨周期分布漂移（PSI），又杜绝未来函数。
    """
    T, N, F = cube.shape
    out = np.empty_like(cube, dtype=float)
    for n in range(N):
        for f in range(F):
            series = pd.Series(cube[:, n, f])
            mean = series.rolling(lookback, min_periods=min_periods).mean()
            std = series.rolling(lookback, min_periods=min_periods).std(ddof=0)
            z = (series - mean) / std.replace(0.0, np.nan)
            out[:, n, f] = z.fillna(0.0).to_numpy()
    return out


def generate_fractional_features(context: dict):
    d = context['best_d']
    bus = context['data_bus']
    slices = context['slices']
    
    all_dates = []
    for p in ['Train-A', 'Train-B1', 'Train-B2', 'Validation', 'Test']:
        if p in slices: all_dates.extend(slices[p])
    master_timeline = pd.DatetimeIndex(sorted(set(all_dates)))
    
    assets = context['assets']
    T, N, F = len(master_timeline), len(assets), 5
    
    diff_cube = np.zeros((T, N, F))
    alive_mask_matrix = np.zeros((T, N), dtype=bool)
    
    from Phase_5.maths_utils import compute_whitebox_features, fractional_diff_series

    for idx, sym in enumerate(assets):
        df = bus.load_asset_history(sym, start_date="2010-01-01", end_date=master_timeline[-1].strftime("%Y-%m-%d"))
        if df is None or df.empty: continue
        
        feats_raw = compute_whitebox_features(df)
        feats_diff = np.zeros_like(feats_raw)
        for f_col in range(F):
            feats_diff[:, f_col] = fractional_diff_series(feats_raw[:, f_col], d)
            
        df_diff = pd.DataFrame(feats_diff, index=df.index)
        df_diff_aligned = df_diff.reindex(master_timeline)
        
        diff_cube[:, idx, :] = np.nan_to_num(df_diff_aligned.values, nan=0.0)
        
        # 同步注入真实上市存活状态标记，杜绝空指针
        if 'alive_mask' in context and sym in context['alive_mask'].columns:
            res_mask = context['alive_mask'][sym].reindex(master_timeline).fillna(False)
            if isinstance(res_mask, pd.DataFrame):
                alive_mask_matrix[:, idx] = res_mask.any(axis=1).values
            else:
                alive_mask_matrix[:, idx] = res_mask.values
        else:
            alive_mask_matrix[:, idx] = df_diff_aligned.notna().any(axis=1).values

    context['fractional_features_cube'] = diff_cube
    context['alive_mask_matrix'] = alive_mask_matrix
    context['trading_days_dt'] = master_timeline

    # ---- 8-9 计划任务四：可选 Rolling Z-Score 标准化（默认由主流程配置开启） ----
    config = context.get('config', {}) or {}
    if config.get('rolling_zscore_features', False):
        diff_cube = apply_rolling_zscore(
            diff_cube,
            lookback=int(config.get('rolling_zscore_lookback', 60)),
            min_periods=int(config.get('rolling_zscore_min_periods', 20)),
        )
        context['fractional_features_cube'] = diff_cube
        context['feature_normalization'] = 'rolling_zscore'
        context['feature_zscore_warmup'] = int(config.get('rolling_zscore_min_periods', 20))
    else:
        context['feature_normalization'] = 'none'
        context['feature_zscore_warmup'] = 0
    
    # logger.info("[OP] Construct Fractional Dimension Cube | [SOURCE] Parallel Multi-Market Parser | [RESULT] High-Dim Cube Dimension: %s | [SIGNIFICANCE] Solidifies spatial input graphs for multi-domain adapters", diff_cube.shape)
    logger.info("[操作] 构建三维分数阶立体特征矩阵 | [来源] 并行多市场数据解析器 | [结果] 高维数据立方体维度 footprint: %s | [意义] 为跨域对齐适配器准备饱满、对齐的立体时空截面数据底座", diff_cube.shape)

def run_feature_filtering(context: dict):
    from Phase_5.dataset_utils import build_partition_dataset
    X, _, _ = build_partition_dataset(context, 'Train-A')
    if X is None or X.shape[0] < 50:
        context['selected_features'] = [0, 1, 2, 3, 4]; return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    F = X_scaled.shape[1]
    
    # 1. 运行横截面方差膨胀因子 (VIF) 过滤共线黑洞
    keep_idx = []
    try:
        vif = [variance_inflation_factor(X_scaled, i) for i in range(F)]
        for i, val in enumerate(vif):
            if val < context['config'].get("vif_threshold", VIF_THRESHOLD): keep_idx.append(i)
    except Exception:
        keep_idx = list(range(F))

    if not keep_idx: keep_idx = list(range(F))
    
    # 2. 通过特征层次聚类 (Hierarchical Agglomeration) 实现特征子空间强力纯组净化
    cluster_ratio = context['config'].get("cluster_select_ratio", CLUSTER_SELECT_RATIO)
    if len(keep_idx) > 1:
        clustering = FeatureAgglomeration(n_clusters=min(3, len(keep_idx)))
        clustering.fit(X_scaled[:, keep_idx])
        pca = PCA(n_components=min(5, len(keep_idx))).fit(X_scaled[:, keep_idx])
        importance = np.abs(pca.components_).sum(axis=0)
        
        selected = []
        for cluster_id in set(clustering.labels_):
            idx_in_c = [i for i, lab in enumerate(clustering.labels_) if lab == cluster_id]
            sorted_idx = sorted(idx_in_c, key=lambda i: importance[i], reverse=True)
            total_imp = sum(importance[i] for i in idx_in_c)
            cum = 0
            for i in sorted_idx:
                selected.append(keep_idx[i]); cum += importance[i]
                if cum / (total_imp + 1e-9) >= cluster_ratio: break
        selected = sorted(set(selected))
    else: selected = keep_idx

    context['selected_features'] = selected
    # logger.info("[OP] Filter Orthogonal Feature Subspaces | [SOURCE] Conformal Multi-Collinearity Evaluator | [RESULT] Retained Vector Indices: %s | [SIGNIFICANCE] Purges linear dependency redundancy to guard optimization kernels", selected)
    logger.info("[操作] 过滤正交特征子空间 | [来源] 多重共线性诊断网格 | [结果] 最终留存有效特征轴序号: %s | [意义] 物理消除高度共线性引发的协方差黑洞，保护下游布莱克-利特曼逆矩阵计算的数值稳定性", selected)
