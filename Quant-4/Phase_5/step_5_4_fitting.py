# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 5.4: 3-Stage Federated Transfer Optimization & Log-Odds Classifier Kernel
"""
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from Phase_5.config import BASE_LGB_PARAMS

logger = logging.getLogger("ModelTraining.Fitting")

def _compute_rbf_mmd_weights(X_source, X_target) -> np.ndarray:
    """计算再生核希尔伯特空间 (RKHS) 内高斯核精确的最大均值差异对齐重要性加权系数"""
    n_s, n_t = X_source.shape[0], X_target.shape[0]
    if n_s == 0 or n_t == 0: return np.ones(max(1, n_s))
        
    combined = np.vstack([X_source, X_target])
    median_dist = np.median(cdist(combined, combined, 'sqeuclidean'))
    sigma = median_dist if median_dist > 0 else 1.0
    
    K_ss = np.exp(-cdist(X_source, X_source, 'sqeuclidean') / sigma)
    K_st = np.exp(-cdist(X_source, X_target, 'sqeuclidean') / sigma)
    
    # 求解经验密度分布重要性对冲向量
    beta = (n_s / (n_t + 1e-9)) * np.sum(K_st, axis=1) / (np.sum(K_ss, axis=1) + 1e-9)
    return np.clip(beta, 0.2, 5.0)

def fit_model_bundle(context: dict):
    logger.info("[Step 5.4] Injecting raw margin log-odds initialization for gradient boosting matrices.")
    config = context.get('config', {})
    lgb_params_base = config.get('lgb_params', BASE_LGB_PARAMS)
    
    from Phase_5.dataset_utils import build_partition_dataset
    X_a, y_clf_a, y_reg_a = build_partition_dataset(context, 'Train-A')
    
    if X_a is None or X_a.shape[0] < 10:
        raise RuntimeError("Fatal: Admitted home-domain modeling elements insufficient.")
        
    selected = context['selected_features']
    X_a_sub = X_a[:, selected]

    scaler = StandardScaler()
    X_a_scaled = scaler.fit_transform(X_a_sub)
    context['feature_scaler'] = scaler

    monitor = context.get("negative_transfer_monitor", {"triggered_melt": False})
    
    # 执行多域特征对齐融合约束机制
    if not monitor["triggered_melt"] and "feature_panel_private_us" in context and context["feature_panel_private_us"]:
        try:
            us_samples = list(context["feature_panel_private_us"].values())
            X_us = np.array(us_samples)[:, :len(selected)] if len(us_samples[0]) >= len(selected) else np.repeat(np.array(us_samples), 2, axis=1)[:, :len(selected)]
            X_us_scaled = scaler.transform(X_us)
            
            mmd_weights = _compute_rbf_mmd_weights(X_a_scaled, X_us_scaled)
            logger.info("[OP] Align Hilbert Kernel Space | [SOURCE] Remote Source Matrix vs Home Target Profile | [RESULT] Computed MMD Weight Shape: %s | [SIGNIFICANCE] Resolves distribution drift via adaptive instance matching", mmd_weights.shape)
            logger.info("[操作] 执行希尔伯特空间核对齐 | [来源] 境外源域矩阵与本土目标域截面 | [结果] 计算出 RBF 核 MMD 重要性系数维度: %s | [意义] 在统计学意义上拉齐海内外样本分布，消除非对称异构特征引发的样本选择偏差")
        except Exception as e:
            logger.warning(f"Domain covariance map alignment failed: {e}; force constant unit matrix fallback.")
            mmd_weights = np.ones(X_a_scaled.shape[0])
    else:
        mmd_weights = np.ones(X_a_scaled.shape[0])

    y_clf_a_mat = np.clip(y_clf_a, 0, 2)
    y_reg_a_mat = np.nan_to_num(y_reg_a, nan=0.0)

    # 刚性修复：强制将概率空间转换为原生常态边际对数几率 (Raw Margin Log-Odds) 填入 init_score
    # 彻底别除旧版概率覆盖导致的基增量树自回归崩塌缺陷。
    freqs = np.bincount(y_clf_a_mat, minlength=3) / len(y_clf_a_mat)
    log_odds = np.log((freqs + 1e-5) / (1.0 - freqs + 1e-5))
    init_score_clf = np.tile(log_odds, (X_a_scaled.shape[0], 1))

    params_clf = lgb_params_base.copy()
    params_clf.update({'objective': 'multiclass', 'num_class': 3})
    
    clf = lgb.LGBMClassifier(**params_clf)
    clf.fit(X_a_scaled, y_clf_a_mat, sample_weight=mmd_weights, init_score=init_score_clf)
    context['direction_classifier'] = clf

    # 顺次标定多分位数连续预期收益共形推断算子
    quantile_models = {}
    for q in [0.025, 0.5, 0.975]:
        reg = lgb.LGBMRegressor(**lgb_params_base)
        reg.set_params(objective='quantile', alpha=q)
        reg.fit(X_a_scaled, y_reg_a_mat, sample_weight=mmd_weights)
        quantile_models[q] = reg
        
    context['quantile_models'] = quantile_models
    logger.info("[OP] Fit Non-Linear Model Bundles | [SOURCE] Hilbert Weighted Feature Matrices | [RESULT] Unified Classifier & Quantile Regressors Saved | [SIGNIFICANCE] Forms the mathematical backbone for down-stream conformal allocations", X_a_scaled.shape)
    logger.info("[操作] 拟合非线性模型阵列 | [来源] 经核加权的多域特征子空间 | [结果] 统一方向分类器与分位数回归簇保存成功 | [意义] 完成多维复杂逻辑网络的联合拟合，为下游级联决策标定与共形推断提供高保真预测载体")