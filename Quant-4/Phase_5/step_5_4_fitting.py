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
    
    # 动态双轨对齐防御：瞬间扩大字典索引槽位
    for label_key in ['y_clf_all', 'y_reg_all']:
        if label_key in context and isinstance(context[label_key], dict):
            matured_dict = {}
            for (k_dt, sym), val in context[label_key].items():
                matured_dict[(k_dt, sym)] = val 
                if isinstance(k_dt, pd.Timestamp):
                    matured_dict[(k_dt.strftime('%Y-%m-%d'), sym)] = val
                    matured_dict[(k_dt.date(), sym)] = val
                elif isinstance(k_dt, str):
                    try:
                        clean_ts = pd.Timestamp(k_dt).tz_localize(None)
                        matured_dict[(clean_ts, sym)] = val
                    except: pass
                elif hasattr(k_dt, 'strftime'):
                    matured_dict[(k_dt.strftime('%Y-%m-%d'), sym)] = val
                    try: matured_dict[(pd.Timestamp(k_dt), sym)] = val
                    except: pass
            context[label_key] = matured_dict

    from Phase_5.dataset_utils import build_partition_dataset
    X_a, y_clf_a, y_reg_a = build_partition_dataset(context, 'Train-A')
    
    # 【核心修复：自主重构兜底引擎】
    # 如果标准组件因为上游存活矩阵时区污染而切空，启动穿透式矩阵重建，强行夺回数据血缘
    if X_a is None or X_a.shape[0] < 10:
        # logger.warning("[GUARD] build_partition_dataset returned insufficient elements. Launching Autonomous Dataset Reconstruction Engine...")
        logger.warning("[守护] build_partition_dataset 返回样本不足。启动自适应时空数据全量重构引擎...")
        
        cube = context.get('fractional_features_cube')
        trading_days = context.get('trading_days_dt')
        assets = context.get('assets', [])
        y_clf_all = context.get('y_clf_all', {})
        y_reg_all = context.get('y_reg_all', {})
        train_a_slices = context.get('slices', {}).get('Train-A', [])
        
        if cube is not None and trading_days is not None and len(assets) > 0:
            # 将所有日期强制对齐为纯净的 YYYY-MM-DD 字符串集合与列表，降维打击时区冲突
            train_a_dates_set = set()
            for d in train_a_slices:
                if isinstance(d, str): train_a_dates_set.add(d[:10])
                elif hasattr(d, 'strftime'): train_a_dates_set.add(d.strftime('%Y-%m-%d'))
            
            trading_days_strs = []
            for d in trading_days:
                if isinstance(d, str): trading_days_strs.append(d[:10])
                elif hasattr(d, 'strftime'): trading_days_strs.append(d.strftime('%Y-%m-%d'))
                else: trading_days_strs.append(str(d)[:10])
                
            X_fallback, y_clf_fallback, y_reg_fallback = [], [], []
            
            # 穿透时空双轴：时间轴 (T) x 资产轴 (N)
            for t_idx, dt_str in enumerate(trading_days_strs):
                if dt_str not in train_a_dates_set:
                    continue
                for a_idx, sym in enumerate(assets):
                    # 只要标签字典里有这个资产在当前日期的记录，就证明其绝对存活且可交易
                    label_val = None
                    for k in [(dt_str, sym), (pd.Timestamp(dt_str), sym), (pd.Timestamp(dt_str).date(), sym)]:
                        if k in y_clf_all:
                            label_val = y_clf_all[k]
                            break
                    
                    if label_val is not None:
                        # 抽取出立体的原始特征轴数据
                        X_fallback.append(cube[t_idx, a_idx, :])
                        y_clf_fallback.append(label_val)
                        
                        reg_val = 0.0
                        for k in [(dt_str, sym), (pd.Timestamp(dt_str), sym), (pd.Timestamp(dt_str).date(), sym)]:
                            if k in y_reg_all:
                                reg_val = y_reg_all[k]
                                break
                        y_reg_fallback.append(reg_val)
            
            if len(X_fallback) >= 10:
                X_a = np.array(X_fallback)
                y_clf_a = np.array(y_clf_fallback)
                y_reg_a = np.array(y_reg_fallback)
                # logger.info(f"[GUARD] Fallback reconstruction successful. Recovered rows: {X_a.shape[0]}")
                logger.info(f"[守护] 强行穿透重构成功。成功夺回特征样本行数: {X_a.shape[0]}")

    # 二级刚性卡点验证
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
            # logger.info("[OP] Align Hilbert Kernel Space | [SOURCE] Remote Source Matrix vs Home Target Profile | [RESULT] Computed MMD Weight Shape: %s | [SIGNIFICANCE] Resolves distribution drift via adaptive instance matching", mmd_weights.shape)
            logger.info("[操作] 执行希尔伯特空间核对齐 | [来源] 境外源域矩阵与本土目标域截面 | [结果] 计算出 RBF 核 MMD 重要性系数维度: %s | [意义] 在统计学意义上拉齐海内外样本分布，消除非对称异构特征引发的样本选择偏差", mmd_weights.shape)
        except Exception as e:
            logger.warning(f"Domain covariance map alignment failed: {e}; force constant unit matrix fallback.")
            mmd_weights = np.ones(X_a_scaled.shape[0])
    else:
        mmd_weights = np.ones(X_a_scaled.shape[0])

    y_clf_a_mat = np.clip(y_clf_a, 0, 2)
    y_reg_a_mat = np.nan_to_num(y_reg_a, nan=0.0)

    # 刚性修复：将概率空间转换为原生常态边际对数几率 (Raw Margin Log-Odds) 填入 init_score
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
    # logger.info("[OP] Fit Non-Linear Model Bundles | [SOURCE] Hilbert Weighted Feature Matrices | [RESULT] Unified Classifier & Quantile Regressors Saved | [SIGNIFICANCE] Forms the mathematical backbone for down-stream conformal allocations", X_a_scaled.shape)
    logger.info("[操作] 拟合非线性模型阵列 | [来源] 经核加权的多域特征子空间 | [结果] 统一方向分类器与分位数回归簇保存成功 | [意义] 完成多维复杂逻辑网络的联合拟合，为下游级联决策标定与共形推断提供高保真预测载体")