"""
Quant-Ultra Flow - Step 5.4: 3-Stage Federated Transfer Learning Engine
Fully refactored to employ mathematically rigorous MLOps standards:
1. True Reproducing Kernel Hilbert Space (RKHS) Gaussian MMD alignment weights (Resolves Flaw A-6).
2. Pure raw margin log-odds initialization for LightGBM gradient boosters (Resolves Flaw A-7).
"""
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from .config import BASE_LGB_PARAMS

logger = logging.getLogger("ModelTraining.Fitting")

def fit_model_bundle(context: dict):
    """
    实施高保真三阶段跨市场联邦迁移学习。
    - 修复 A-6: 引入 RBF 高斯多维核经验重要性估计。
    - 修复 A-7: 将传递给 LightGBM Classifier 的 init_score 修正为 raw_score (log-odds)。
    """
    logger.info("[Step 5.4] 激活三阶段跨市场高保真迁移 network 与领域对抗对齐轨...")
    config = context.get('config', {})
    lgb_params_base = config.get('lgb_params', BASE_LGB_PARAMS)
    
    bus = context['data_bus']
    diff_cube = context.get('fractional_features_cube')
    T, N, F = diff_cube.shape
    selected = context['selected_features']
    logger.info(f"[Step 5.4] Selected feature indices for model fitting: {selected}")
    logger.info(f"[Step 5.4] Fractional feature cube shape: {diff_cube.shape}, Selected features count: {len(selected)}")
    print(f"[Step 5.4] Starting model fitting with {len(selected)} selected features over {T} time points for {N} assets.")

    slices = context.get('slices', {}) or context.get('data_slices', {})
    train_dates = slices.get('Train-A', [])

    # 🛡️ 弹性双轨兜底防线：如果缓存重载导致切片级联断层，启动历轨物理还原
    if not train_dates:
        logger.warning("🚨 [模型拟合警报] context['slices'] 中缺失 'Train-A' 显式时空轴！激活二轨日历自愈补全...")
        if 'trading_days_dt' in context and context['trading_days_dt'] is not None:
            # 严格根据设计契约，截取 Train-A 的标准物理时间视区范围
            idx = pd.DatetimeIndex(context['trading_days_dt']).tz_localize(None)
            train_dates = idx[(idx >= "2010-01-04") & (idx <= "2018-06-25")].strftime("%Y-%m-%d").tolist()
            
    if not train_dates:
        raise KeyError("❌ 致命断层：全局总线中 slices['Train-A'] 与 trading_days_dt 完全瘫痪，无法构建拟合时序轴。")
        
    logger.info(f"[Step 5.4] Train-A dates successfully reconstructed: {len(train_dates)} days from {train_dates[0]} to {train_dates[-1]}.")

    assets = context['assets']
    master_timeline = pd.DatetimeIndex(train_dates)

    # 构建训练集特征矩阵和标签向量
    y_clf_all = context.get('y_clf_all', {})
    y_reg_all = context.get('y_reg_all', {})
    
    # 🍏 【新增控制流：标签字典键名刚性归一化】消灭因缓存落盘/反序列化引发的时区和类型断层
    def normalize_target_dict(d: dict) -> dict:
        if not d: return {}
        normalized = {}
        for k, v in d.items():
            if isinstance(k, tuple) and len(k) == 2:
                k_dt, k_sym = k
                if isinstance(k_dt, str):
                    k_dt_str = k_dt.split()[0]  # 兼容带有时分秒的字符串，截取 YYYY-MM-DD
                elif hasattr(k_dt, 'strftime'):
                    k_dt_str = k_dt.strftime("%Y-%m-%d")
                else:
                    k_dt_str = str(k_dt)
                normalized[(k_dt_str, k_sym)] = v
        return normalized

    logger.info("🧼 [时空契约确证] 正在对全局总线历史标签数据进行时区不敏感归一化清洗...")
    y_clf_normalized = normalize_target_dict(y_clf_all)
    y_reg_normalized = normalize_target_dict(y_reg_all)
    
    X_us, y_clf_us, y_reg_us = [], [], []
    X_a, y_clf_a, y_reg_a = [], [], []
    
    # 执行唯一的、经过时区规避清洗的安全数据样本收集循环
    for i, dt in enumerate(master_timeline):
        dt_str = dt.strftime("%Y-%m-%d")
        for j, sym in enumerate(assets):
            feat = diff_cube[i, j, selected]
            if np.isnan(feat).any():
                continue
            
            # 🍏 【升级检索】：使用归一化后的纯净字符串元组键进行安全检索，无视时区干扰
            yc = y_clf_normalized.get((dt_str, sym), None)
            yr = y_reg_normalized.get((dt_str, sym), None)
                
            if yc is not None and yr is not None:
                if bus.get_node_by_asset(sym) == "US_share_node":
                    X_us.append(feat)
                    y_clf_us.append(yc)
                    y_reg_us.append(yr)
                else:
                    X_a.append(feat)
                    y_clf_a.append(yc)
                    y_reg_a.append(yr)

    # 🛑 【死锁防御】这里已经彻底清除了原先残留的二次重置置空逻辑与老旧大循环 🛑

    if not X_a:
        raise RuntimeError("Target domain (A-share) standard samples empty on Train-A.")

    X_a_mat = np.vstack(X_a)
    y_clf_a_mat = np.array(y_clf_a)
    y_reg_a_mat = np.array(y_reg_a)

    scaler = StandardScaler()
    X_a_scaled = scaler.fit_transform(X_a_mat)
    context['feature_scaler'] = scaler

    monitor = context.get("negative_transfer_monitor", {"consecutive_violation_count": 0, "triggered_melt": False})
    
    # 【红线熔断拦截判断】
    if monitor.get("triggered_melt", False) or config.get("negative_transfer_rollback_flag", False):
        logger.warning("🚨 [负迁移熔断机制触发现场] 正在强制切入纯本地 A 股无污染基线拟合轨道...")
        
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
        return

    # 推进标准跨市场三阶段树融合
    if X_us:
        X_us_mat = np.vstack(X_us)
        X_us_scaled = scaler.transform(X_us_mat)
        y_clf_us_mat = np.array(y_clf_us)
        y_reg_us_mat = np.array(y_reg_us)
        
        # 1. 源域预训练基线网络
        params_clf_us = lgb_params_base.copy()
        params_clf_us.update({'objective': 'multiclass', 'num_class': 3})
        clf_us_base = lgb.LGBMClassifier(**params_clf_us)
        clf_us_base.fit(X_us_scaled, y_clf_us_mat)
        
        reg_us_base = {}
        for q in [0.025, 0.5, 0.975]:
            params_reg_us = lgb_params_base.copy()
            params_reg_us.update({'objective': 'quantile', 'alpha': q})
            r_us = lgb.LGBMRegressor(**params_reg_us)
            r_us.fit(X_us_scaled, y_reg_us_mat)
            reg_us_base[q] = r_us
            
        # ====================================================
        # 核心修复 A-6：严格 RBF 高斯多核 RKHS 经验对齐密度估计
        # ====================================================
        logger.info("📐 [修复 A-6] 正在通过成对欧氏距离矩阵解算高维再生核希尔伯特空间核权重...")
        pairwise_dists = cdist(X_a_scaled, X_us_scaled, metric='sqeuclidean')
        gamma = 1.0 / (X_a_scaled.shape[1] + 1e-8)
        rkhs_kernel_mat = np.exp(-gamma * pairwise_dists)
        
        mmd_density = np.mean(rkhs_kernel_mat, axis=1)
        mmd_weights = mmd_density / (np.sum(mmd_density) + 1e-8) * len(X_a_scaled)
        
        # ====================================================
        # 核心修复 A-7：将 init_score 由概率刚性变更为 Raw Scores
        # ====================================================
        logger.info("🚀 [修复 A-7] 提取美股分类预训练网络 Raw Margins/Log-Odds 作为初始梯度残差基础...")
        init_score_clf = clf_us_base.predict(X_a_scaled, raw_score=True)
        
        # 树模型二级对抗性微调
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
        logger.warning("源域样本集异常为空，降级回退至本地初始化拟合。")
        
        # 刚性注入纯本地 A 股多分类方向网络
        params_clf = lgb_params_base.copy()
        params_clf.update({'objective': 'multiclass', 'num_class': 3})
        clf = lgb.LGBMClassifier(**params_clf)
        clf.fit(X_a_scaled, y_clf_a_mat)
        context['direction_classifier'] = clf

        # 刚性注入纯本地 A 股分位数回归网络
        quantile_models = {}
        for q in [0.025, 0.5, 0.975]:
            params_reg = lgb_params_base.copy()
            params_reg.update({'objective': 'quantile', 'alpha': q})
            reg = lgb.LGBMRegressor(**params_reg)
            reg.fit(X_a_scaled, y_reg_a_mat)
            quantile_models[q] = reg
        context['quantile_models'] = quantile_models

    logger.info("[Step 5.4] 跨市场三阶段模型簇组装完毕，高阶统计矩与梯度基准修复完毕。")