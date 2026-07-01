# -*- coding: utf-8 -*-
"""
step5/step_5_4_fitting.py
业务模块：在完整 Train-A 数据集上拟合双轨级联模型束（方向多分类 LightGBM + 三目标分位数回归群）。
"""
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from .config import BASE_LGB_PARAMS

logger = logging.getLogger("ModelTraining.Fitting")

def fit_model_bundle(context: dict):
    logger.info("[Step 5.4] Fitting dual-track models (direction classifier & CQR quantile regressors).")
    config = context.get('config', {})
    lgb_params_base = config.get('lgb_params', BASE_LGB_PARAMS)

    diff_cube = context.get('fractional_features_cube')
    T, N, F = diff_cube.shape
    selected = context['selected_features']
    train_dates = context['train_a_dates']

    master_timeline = pd.DatetimeIndex(train_dates)

    y_clf_all = context.get('y_clf_all', {})
    y_reg_all = context.get('y_reg_all', {})
    X_list, y_clf_list, y_reg_list = [], [], []
    
    for i, dt in enumerate(master_timeline):
        dt_str = dt.strftime("%Y-%m-%d")
        for j, sym in enumerate(context['assets']):
            feat = diff_cube[i, j, selected]
            if np.isnan(feat).any():
                continue
            
            # 双重兼容强匹配检索
            yc, yr = None, None
            for key in [(dt, sym), (dt_str, sym)]:
                if key in y_clf_all and yc is None: yc = y_clf_all[key]
                if key in y_reg_all and yr is None: yr = y_reg_all[key]
                
            if yc is not None and yr is not None:
                X_list.append(feat)
                y_clf_list.append(yc)
                y_reg_list.append(yr)

    if not X_list:
        raise RuntimeError("No aligned standard samples found on Train-A matrix for fitting.")
        
    X_train = np.vstack(X_list)
    y_clf = np.array(y_clf_list)
    y_reg = np.array(y_reg_list)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    context['feature_scaler'] = scaler

    # 1. 拟合三向收益率过滤分类器
    params_clf = lgb_params_base.copy()
    params_clf.update({'objective': 'multiclass', 'num_class': 3})
    clf = lgb.LGBMClassifier(**params_clf)
    clf.fit(X_train_scaled, y_clf)
    context['direction_classifier'] = clf

    # 2. 拟合多目标 CQR 分位数回归群 (q = 0.025, 0.5, 0.975)
    quantile_models = {}
    for q in [0.025, 0.5, 0.975]:
        params_reg = lgb_params_base.copy()
        params_reg.update({'objective': 'quantile', 'alpha': q})
        reg = lgb.LGBMRegressor(**params_reg)
        reg.fit(X_train_scaled, y_reg)
        quantile_models[q] = reg
        
    context['quantile_models'] = quantile_models
    logger.info("[Step 5.4] Dual-track model cluster assembly completed.")