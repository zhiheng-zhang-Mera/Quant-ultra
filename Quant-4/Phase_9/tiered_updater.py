# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 9.2: Tiered MLOps Updating Protocols & Drift Management
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from Phase_9.config import DEFAULT_MLOPS_CONFIG

logger = logging.getLogger("MLOps.TieredUpdater")

def _compute_conformal_psi(base_array: np.ndarray, target_array: np.ndarray, bins: int = 10) -> float:
    """计算单个高维特征轴的群体稳定性指标 (PSI)，加入极小正数防止对数溢出"""
    if len(base_array) == 0 or len(target_array) == 0: return 0.0
    
    base_clean = base_array[~np.isnan(base_array)]
    target_clean = target_array[~np.isnan(target_array)]
    if len(base_clean) == 0 or len(target_clean) == 0: return 0.0

    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(base_clean, percentiles)
    bin_edges[0] -= 1e-5; bin_edges[-1] += 1e-5

    base_counts, _ = np.histogram(base_clean, bins=bin_edges)
    target_counts, _ = np.histogram(target_clean, bins=bin_edges)

    base_pct = base_counts / len(base_clean)
    target_pct = target_counts / len(target_clean)

    psi_val = 0.0
    for i in range(bins):
        actual, expected = target_pct[i], base_pct[i]
        actual = max(1e-4, actual)
        expected = max(1e-4, expected)
        psi_val += (actual - expected) * np.log(actual / expected)
        
    return float(psi_val)

def evaluate_distribution_drift(context: dict) -> dict:
    """
    固定分箱特征分布漂移追踪审计。
    核心修复 A-3：拒绝在 MLOps 层重新手写低阶因子。直接穿透抽取 Phase_5 固化生成并留存的
    三维高维分数阶立体特征矩阵 (fractional_features_cube)，确保漂移监控与模型特征域的完美质地对齐。
    """
    # logger.info("[OP] Fetch Conformal Cube Slice | [SOURCE] Phase_5 Fractional Matrix Cube | [RESULT] Extracting aligned feature matrix | [SIGNIFICANCE] Eliminates mathematical representation discrepancy to secure drift audit consistency")
    logger.info("[操作] 提取共形立体特征切片 | [来源] 阶段5分数阶立体数据立方体 | [结果] 成功导入对齐的多维特征矩阵 | [意义] 彻底消灭监控层与训练层之间的特征定义不对称，确保漂移审计的逻辑一致性")

    feature_cube = context.get('fractional_features_cube')
    selected_features = context.get('selected_features')
    trading_days = context.get('trading_days_dt')
    
    if feature_cube is None or selected_features is None or trading_days is None:
        logger.warning("Feature cube absent in pipeline context. Skip PSI calculation.")
        return context

    config = context.get('config', {})
    lookback = config.get('lookback_psi_window', DEFAULT_MLOPS_CONFIG['lookback_psi_window'])
    psi_limit = config.get('psi_drift_crit_threshold', DEFAULT_MLOPS_CONFIG['psi_drift_crit_threshold'])
    consecutive_trigger = config.get('psi_consecutive_days_trigger', DEFAULT_MLOPS_CONFIG['psi_consecutive_days_trigger'])
    smoothing_period = config.get('model_smoothing_period', DEFAULT_MLOPS_CONFIG['model_smoothing_period'])

    T, N, F = feature_cube.shape
    if T < lookback + 10: return context

    # 提取基准时段与目标现时窗口的特征切面
    base_sub = feature_cube[:lookback, :, :]
    target_sub = feature_cube[-10:, :, :] # 近10个交易日特征变异观测
    
    psi_list = []
    for f_idx in selected_features:
        if f_idx < F:
            base_vec = base_sub[:, :, f_idx].flatten()
            target_vec = target_sub[:, :, f_idx].flatten()
            psi_list.append(_compute_conformal_psi(base_vec, target_vec))

    mean_psi = float(np.mean(psi_list)) if psi_list else 0.0
    context['current_mean_psi'] = mean_psi

    consecutive_breaches = context.get('psi_consecutive_breaches', 0)
    if mean_psi >= psi_limit:
        consecutive_breaches += 1
    else:
        consecutive_breaches = 0
    context['psi_consecutive_breaches'] = consecutive_breaches

    trigger_retrain = False
    if consecutive_breaches >= consecutive_trigger:
        # logger.critical("[OP] Detect Severe Feature Distortion | [SOURCE] PSI Audit Watchdog | [RESULT] Consecutive breaches: %s/%s | [SIGNIFICANCE] Activates Tier 3 full federated retraining pipeline", consecutive_breaches, consecutive_trigger)
        logger.critical("[操作] 检测到严重的特征分布失真 | [来源] PSI 漂移监控看门狗 | [结果] 连续超限天数达标: %s/%s | [意义] 物理激活 Tier 3 全量联邦重训网络，彻底消灭模型长期服役过拟合失效", consecutive_breaches, consecutive_trigger)
        trigger_retrain = True

    # 新老模型双轨渐进线性步进切换 (Smoothing stair-case transitions)
    transition_day = context.get('transition_day', 0)
    if trigger_retrain:
        context['tier3_retrain_active'] = True
        context['transition_day'] = 1
        context['alpha_new_model'] = 1.0 / smoothing_period
    elif transition_day > 0:
        if transition_day >= smoothing_period:
            context['transition_day'] = 0
            context['alpha_new_model'] = 1.0
            context['tier3_retrain_active'] = False
            # logger.info("[OP] Finalize Staircase Transition | [SOURCE] Temporal Model Blender | [RESULT] Smooth handover accomplished | [SIGNIFICANCE] Fully destroys the legacy model instance safely")
            logger.info("[操作] 终结模型线性步进平滑过渡 | [来源] 时域模型混合器 | [结果] 新老模型无缝接交接完毕 | [意义] 彻底物理销毁老模型实体，完成生产集群在线更新交底")
        else:
            context['transition_day'] = transition_day + 1
            context['alpha_new_model'] = (transition_day + 1) / smoothing_period
    else:
        context['alpha_new_model'] = 1.0
        context['tier3_retrain_active'] = False

    return context