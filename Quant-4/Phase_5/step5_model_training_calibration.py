# -*- coding: utf-8 -*-
"""
Phase 5: Joint Hyperparameter Tuning, Dual-Track Cascade Calibration, and Model Fitting
"""
import logging
from datetime import datetime
import pandas as pd
from Phase_5.step_5_1_cv import run_walk_forward_cv
from Phase_5.step_5_2_3_features import generate_fractional_features, run_feature_filtering
from Phase_5.step_5_4_fitting import fit_model_bundle
from Phase_5.step_5_5_calibration import run_cascade_calibration

logger = logging.getLogger("ModelTraining")

def execute(pipeline_context: dict) -> dict:
    logger.info("=" * 60)
    # logger.info("[OP] Enter Phase_5 Master Lifecycle | [SOURCE] Central Workflow Orchestrator Loop | [RESULT] Commencing Joint Federated Machine Learning | [SIGNIFICANCE] Solves structural cross-market distribution drift under rigid security contract guards")
    logger.info("[操作] 锁锁 Phase_5 生命周期主控大关口 | [来源] 中央流水线编排控制循环 | [结果] 开始联合联邦机器学习与双轨级联标定工作流 | [意义] 在防渗透安全契约栅栏下，彻底贯通多源长记忆解算与不稳态时序共形推断")
    logger.info("=" * 60)

    slices = pipeline_context.get('slices', {}) or pipeline_context.get('data_slices', {})
    
    # 时空视窗缺失弹性智能重构与自愈自适应对齐
    if not slices or 'Train-A' not in slices:
        logger.warning("[核心警报] 拦截到热启动阶段变量真空！紧急启动时空拓扑自愈补偿程序...")
        if 'calendar_alignment' in pipeline_context and 'alignment_table' in pipeline_context['calendar_alignment']:
            full_timeline = pipeline_context['calendar_alignment']['alignment_table']["ashare_date"].tolist()
            idx = pd.DatetimeIndex(full_timeline).tz_localize(None)
            slices = {
                "Train-A": idx[(idx >= "2010-01-04") & (idx <= "2018-06-25")].strftime("%Y-%m-%d").tolist(),
                "Train-B1": idx[(idx >= "2018-07-10") & (idx <= "2020-03-05")].strftime("%Y-%m-%d").tolist(),
                "Train-B2": idx[(idx >= "2020-03-20") & (idx <= "2021-11-16")].strftime("%Y-%m-%d").tolist(),
                "Validation": idx[(idx >= "2021-12-01") & (idx <= "2024-06-06")].strftime("%Y-%m-%d").tolist(),
                "Test": idx[(idx >= "2024-06-24") & (idx <= datetime.now().strftime("%Y-%m-%d"))].strftime("%Y-%m-%d").tolist()
            }
            pipeline_context['slices'] = slices
        else:
            raise RuntimeError("Fatal Timeline Interruption: alignment tables completely absent.")

    all_dates = []
    for partition in ['Train-A', 'Train-B1', 'Train-B2', 'Validation', 'Test']:
        if partition in slices: all_dates.extend(slices[partition])
    if all_dates:
        pipeline_context['trading_days_dt'] = pd.DatetimeIndex(sorted(set(all_dates)))
    else:
        raise RuntimeError("Chronological slices registry corrupt.")

    # 顺次拉起解耦重构后的分布式原子计算逻辑链
    run_walk_forward_cv(pipeline_context)
    generate_fractional_features(pipeline_context)
    run_feature_filtering(pipeline_context)
    fit_model_bundle(pipeline_context)
    run_cascade_calibration(pipeline_context)
    if pipeline_context.get('num_trials', 0) < 1:
        raise RuntimeError("Phase 5 cannot prove num_trials from calibration evidence.")
    
    # 构造并向主总线安全递交持久化资产令牌
    return {
        "best_d": pipeline_context['best_d'],
        "selected_features": pipeline_context['selected_features'],
        "direction_classifier": pipeline_context['direction_classifier'],
        "quantile_models": pipeline_context['quantile_models'],
        "gamma_star": pipeline_context['gamma_star'],
        "num_trials": pipeline_context['num_trials'],
        "trial_evidence": pipeline_context['trial_evidence'],
        "q_error_threshold_dict": pipeline_context['q_error_threshold_dict'],
        "fractional_features_cube": pipeline_context['fractional_features_cube'],
        "feature_scaler": pipeline_context['feature_scaler'],
        "model_training_ready": True
    }
