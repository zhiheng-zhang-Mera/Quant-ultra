# -*- coding: utf-8 -*-
"""
Phase 5: Joint Hyperparameter Tuning, Dual-Track Cascade Calibration, and Model Fitting
Fully compliant with Final-Flow.md [2026 Production Release]
面向主管道总线的主入口文件，负责子组件级联调用与监控。
"""
import logging

from .step_5_1_cv import run_walk_forward_cv
from .step_5_2_3_features import generate_fractional_features, run_feature_filtering
from .step_5_4_fitting import fit_model_bundle
from .step_5_5_calibration import run_cascade_calibration

logger = logging.getLogger("ModelTraining")

def execute(pipeline_context: dict) -> dict:
    """阶段五原子化重构总入口函数"""
    logger.info("="*60)
    logger.info(">>> Phase 5: Model Training & Calibration [Modular Engine]")

    if 'num_trials' not in pipeline_context:
        pipeline_context['num_trials'] = 0

    # 5.1 执行净化向前行走交叉验证，探寻最优记忆参数 $d^*$
    run_walk_forward_cv(pipeline_context)
    
    # 5.2 对 Train-A 全序列应用流式因果递推微分
    generate_fractional_features(pipeline_context)
    
    # 5.3 统计特征共线性空间凝聚与 VIF 超限净化
    run_feature_filtering(pipeline_context)
    
    # 5.4 模型多目标双轨拟合（多分类与 conformal 分位数估计）
    fit_model_bundle(pipeline_context)
    
    # 5.5 标定单调性过滤门槛并提取 CQR 误差分位数
    run_cascade_calibration(pipeline_context)
    
    pipeline_context['model_training_ready'] = True
    logger.info(">>> Phase 5 completed successfully.")
    return pipeline_context