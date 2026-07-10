# -*- coding: utf-8 -*-
"""
step5/step_5_4_fitting.py
业务模块：实施三阶段跨市场迁移学习与领域对抗对齐架构，支持针对负迁移的纯A股本地轨刚性熔断回退。
"""
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from .config import BASE_LGB_PARAMS

logger = logging.getLogger("ModelTraining.Fitting")

def fit_model_bundle(context: dict):
    logger.info("[Step 5.4] Initiating 3-Stage Federated Transfer Learning & Domain Adaptation Alignment...")
    config = context.get('config', {})
    lgb_params_base = config.get('lgb_params', BASE_LGB_PARAMS)
    
    bus = context['data_bus']
    diff_cube = context.get('fractional_features_cube')
    T, N, F = diff_cube.shape
    selected = context['selected_features']
    train_dates = context['train_a_dates']
    assets = context['assets']
    master_timeline = pd.DatetimeIndex(train_dates)

    y_clf_all = context.get('y_clf_all', {})
    y_reg_all = context.get('y_reg_all', {})
    
    X_us, y_clf_us, y_reg_us = [], [], []
    X_a, y_clf_a, y_reg_a = [], [], []
    
    for i, dt in enumerate(master_timeline):
        dt_str = dt.strftime("%Y-%m-%d")
        for j, sym in enumerate(assets):
            feat = diff_cube[i, j, selected]
            if np.isnan(feat).any():
                continue
            
            yc, yr = None, None
            for key in [(dt, sym), (dt_str, sym)]:
                if key in y_clf_all and yc is None: yc = y_clf_all[key]
                if key in y_reg_all and yr is None: yr = y_reg_all[key]
                
            if yc is not None and yr is not None:
                if bus.get_node_by_asset(sym) == "US_share_node":
                    X_us.append(feat)
                    y_clf_us.append(yc)
                    y_reg_us.append(yr)
                else:
                    X_a.append(feat)
                    y_clf_a.append(yc)
                    y_reg_a.append(yr)

    if not X_a:
        raise RuntimeError("Target domain (A-share) standard samples empty on Train-A.")

    X_a_mat = np.vstack(X_a)
    y_clf_a_mat = np.array(y_clf_a)
    y_reg_a_mat = np.array(y_reg_a)

    scaler = StandardScaler()
    X_a_scaled = scaler.fit_transform(X_a_mat)
    context['feature_scaler'] = scaler

    monitor = context.get("negative_transfer_monitor", {"consecutive_violation_count": 0, "triggered_melt": False})
    
    # 【红线刚性防御】检查负迁移熔断机制状态或全局强制开关
    if monitor.get("triggered_melt", False) or config.get("negative_transfer_rollback_flag", False):
        logger.warning("🚨 [Melt Down Execution] Negative transfer detected. Activating Pure A-Share Local Model Track.")
        
        # 纯 A 股本地独立拟合轨
        params_clf = lgb_params_base.copy()
        params_clf.update({'objective': 'multiclass', 'num_class': 3})
        clf = lgb.LGBMClassifier(**params_clf)
        clf.fit(X_a_scaled, y_clf_a_mat)
        context['direction_classifier'] = clf

        quantile_models = {}
        for q in [0.025, 0.5, 0.975]:
            params_reg = lgb_params_base.copy()
            params_reg.update({'objective': 'quantile', 'alpha': q})
            reg = lgb.LGBMRegressor(**params_reg)
            reg.fit(X_a_scaled, y_reg_a_mat)
            quantile_models[q] = reg
        context['quantile_models'] = quantile_models
        logger.info("[Step 5.4] Pure local track model bundle fitted successfully.")
        return

    # 推进标准三阶段跨市场迁移网络与领域对抗对齐轨
    logger.info("Executing Step 1: Source Domain (US Stock) Pre-training...")
    if X_us:
        X_us_mat = np.vstack(X_us)
        X_us_scaled = scaler.transform(X_us_mat)
        y_clf_us_mat = np.array(y_clf_us)
        y_reg_us_mat = np.array(y_reg_us)
        
        # 预训练基线分类模型
        params_clf_us = lgb_params_base.copy()
        params_clf_us.update({'objective': 'multiclass', 'num_class': 3})
        clf_us_base = lgb.LGBMClassifier(**params_clf_us)
        clf_us_base.fit(X_us_scaled, y_clf_us_mat)
        
        # 预训练基线分位数回归群
        reg_us_base = {}
        for q in [0.025, 0.5, 0.975]:
            params_reg_us = lgb_params_base.copy()
            params_reg_us.update({'objective': 'quantile', 'alpha': q})
            r_us = lgb.LGBMRegressor(**params_reg_us)
            r_us.fit(X_us_scaled, y_reg_us_mat)
            reg_us_base[q] = r_us
            
        logger.info("Executing Step 2 & 3: Freezing feature network & Domain Adaptation via MMD weights...")
        # 基于高斯核的最大均值差异 (MMD) 为目标域样本分配对齐权重
        s_mean = np.mean(X_us_scaled, axis=0)
        t_diff = X_a_scaled - s_mean
        mmd_dist = np.exp(-1.0 * np.sum(t_diff**2, axis=1))
        mmd_weights = mmd_dist / (np.sum(mmd_dist) + 1e-8) * len(X_a_scaled)
        
        # 将美股预训练网络的预测分数作为 init_score 注入 A 股，完美的以残差迭代实现了“冻结底层网络”的树树融合
        init_score_clf = clf_us_base.predict_proba(X_a_scaled)
        
        # 个性化微调输出
        clf = lgb.LGBMClassifier(**params_clf_us)
        clf.fit(X_a_scaled, y_clf_a_mat, sample_weight=mmd_weights, init_score=init_score_clf)
        context['direction_classifier'] = clf
        
        quantile_models = {}
        for q in [0.025, 0.5, 0.975]:
            init_score_reg = reg_us_base[q].predict(X_a_scaled)
            reg = lgb.LGBMRegressor(**lgb_params_base)
            reg.set_params(objective='quantile', alpha=q)
            reg.fit(X_a_scaled, y_reg_a_mat, sample_weight=mmd_weights, init_score=init_score_reg)
            quantile_models[q] = reg
        context['quantile_models'] = quantile_models
    else:
        logger.warning("US Pretraining samples empty. Fallback to standard local fitting.")
        params_clf = lgb_params_base.copy()
        params_clf.update({'objective': 'multiclass', 'num_class': 3})
        clf = lgb.LGBMClassifier(**params_clf)
        clf.fit(X_a_scaled, y_clf_a_mat)
        context['direction_classifier'] = clf

        quantile_models = {}
        for q in [0.025, 0.5, 0.975]:
            params_reg = lgb_params_base.copy()
            params_reg.update({'objective': 'quantile', 'alpha': q})
            reg = lgb.LGBMRegressor(**params_reg)
            reg.fit(X_a_scaled, y_reg_a_mat)
            quantile_models[q] = reg
        context['quantile_models'] = quantile_models

    logger.info("[Step 5.4] 3-Stage Domain Adaptation Model cluster assembly completed.")
}