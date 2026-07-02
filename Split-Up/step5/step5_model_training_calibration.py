# -*- coding: utf-8 -*-
"""
Phase 5: Joint Hyperparameter Tuning, Dual-Track Cascade Calibration, and Model Fitting
Fully compliant with Final-Flow.md [2026 Production Release]
面向主管道总线的主入口文件，负责子组件级联调用与监控。
🟩 增强安全栅栏：防范由于热启动变量真空引发的下游级联断层。
"""
import logging
import pandas as pd

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

    # 获取时间划分切片边界（兼容多级 Key）
    slices = pipeline_context.get('slices', {}) or pipeline_context.get('data_slices', {})
    
    # 🛡️ 智能时空自愈栅栏：如果由于上游全量缓存击中导致 slices 丢失或为空，通过全局历轨轴强行物理还原
    if not slices or not any(slices.values()):
        logger.warning("🚨 [时空断层自愈] 检测到上游全部缓存击中（热启动）导致 slices 视区为空！启动硬核时空分区恢复程序...")
        
        full_timeline = []
        # 优先从跨市场对齐底座提取
        if "calendar_alignment" in pipeline_context:
            align_table = pipeline_context["calendar_alignment"].get("alignment_table", pd.DataFrame())
            if not align_table.empty and "ashare_date" in align_table.columns:
                full_timeline = align_table["ashare_date"].tolist()
        # 备用：从基础原生交易日历提取
        if not full_timeline and "trading_days_dt_cn" in pipeline_context:
            full_timeline = pipeline_context["trading_days_dt_cn"]
            
        if full_timeline:
            # 强行过滤并剔除隐式时区，还原为纯净物理轴
            idx = pd.DatetimeIndex(full_timeline).tz_localize(None)
            
            # 严格按照 Step 2 设计的物理边界时间指纹一键还原切片
            slices = {
                "Train-A": idx[(idx >= "2010-01-04") & (idx <= "2018-06-25")].strftime("%Y-%m-%d").tolist(),
                "Train-B1": idx[(idx >= "2018-07-10") & (idx <= "2020-03-05")].strftime("%Y-%m-%d").tolist(),
                "Train-B2": idx[(idx >= "2020-03-20") & (idx <= "2021-11-16")].strftime("%Y-%m-%d").tolist(),
                "Validation": idx[(idx >= "2021-12-01") & (idx <= "2024-06-06")].strftime("%Y-%m-%d").tolist(),
                "Test": idx[(idx >= "2024-06-24") & (idx <= "2026-12-31")].strftime("%Y-%m-%d").tolist()
            }
            pipeline_context['slices'] = slices
            logger.info(f"✅ [时空自愈] 物理分区成功无损复原：Train-A({len(slices['Train-A'])}天), Train-B1({len(slices['Train-B1'])}天), Train-B2({len(slices['Train-B2'])}天), Validation({len(slices['Validation'])}天), Test({len(slices['Test'])}天)")
        else:
            raise RuntimeError("❌ 致命断层：全局总线历轨完全丢失，无法执行时空恢复。")

    # 构建全局交易日期日历（用于数据集切分索引）
    all_dates = []
    for partition in ['Train-A', 'Train-B1', 'Train-B2', 'Validation', 'Test']:
        if partition in slices:
            all_dates.extend(slices[partition])
            
    if all_dates:
        all_dates = sorted(set(all_dates))
        pipeline_context['trading_days_dt'] = pd.DatetimeIndex(all_dates)
        logger.info(f"📅 全局日历构建完成，共 {len(all_dates)} 个交易日。")
    else:
        raise RuntimeError("slices 中未找到任何分区日期，无法构建全局日历。")

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