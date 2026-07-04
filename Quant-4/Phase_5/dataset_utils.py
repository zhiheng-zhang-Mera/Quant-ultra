# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_5 Adaptive Cross-Sectional Panel Ingestion Subsystem
"""
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger("ModelTraining.Dataset")

def build_partition_dataset(context: dict, partition_name: str):
    """
    根据标的池资产在历史特定视窗切片内的上市常态以及存活率矩阵，
    动态决定横截面样本准入性。训练期未上市资产物理别除，验证期可用资产安全合并。
    """
    slices = context['slices']
    target_dates = slices.get(partition_name)
    if target_dates is None or len(target_dates) == 0:
        logger.warning("[OP] Evaluate Ingestion Window | [SOURCE] Context Slices Config | [RESULT] Partition Empty: %s | [SIGNIFICANCE] Skip dataset construction loop", partition_name)
        logger.warning("[操作] 评估划分载入窗口 | [来源] 上下文时空分区配置 | [结果] 切片区间为空: %s | [意义] 自动跳过空样本集的数据集构建循环")
        return None, None, None

    trading_days = context['trading_days_dt']
    feat_cube = context['fractional_features_cube']
    alive_matrix = context['alive_mask_matrix']
    assets = context['assets']
    y_clf_all = context['y_clf_all']
    y_reg_all = context['y_reg_all']

    date_indices = [trading_days.get_loc(pd.Timestamp(d)) for d in target_dates if pd.Timestamp(d) in trading_days]
    if not date_indices: return None, None, None

    sub_cube = feat_cube[date_indices, :, :]
    sub_alive = alive_matrix[date_indices, :]
    target_dt = trading_days[date_indices]

    X_list, y_clf_list, y_reg_list = [], [], []
    admitted_count = 0

    for idx, sym in enumerate(assets):
        feat_series = sub_cube[:, idx, :]
        y_c_list, y_r_list = [], []
        valid_indices = []

        for i, dt in enumerate(target_dt):
            dt_str = dt.strftime("%Y-%m-%d")
            yc, yr = None, None
            for key in [(dt, sym), (dt_str, sym), (pd.Timestamp(dt), sym)]:
                if key in y_clf_all: yc = y_clf_all[key]
                if key in y_reg_all: yr = y_reg_all[key]
                if yc is not None and yr is not None: break
                    
            if yc is not None and yr is not None and sub_alive[i, idx]:
                y_c_list.append(yc); y_r_list.append(yr); valid_indices.append(i)

        if len(y_c_list) >= 10:
            X_list.append(feat_series[valid_indices])
            y_clf_list.append(np.array(y_c_list))
            y_reg_list.append(np.array(y_r_list))
            admitted_count += 1

    if X_list:
        X = np.vstack(X_list)
        y_clf = np.concatenate(y_clf_list)
        y_reg = np.concatenate(y_reg_list)
        logger.info("[OP] Ingest Cross-Sectional Panel | [SOURCE] Multi-Track Matrix Trunk | [RESULT] Zone: %s, Admitted Assets: %s, Total Flat Shapes: %s | [SIGNIFICANCE] Locks non-overlapping alignment data inputs", partition_name, admitted_count, X.shape)
        logger.info("[操作] 录入横截面面板数据集 | [来源] 复合时空长记忆特征主干 | [结果] 分区: %s, 接纳可用资产数: %s, 展平总样本维度: %s | [意义] 为主干网络拟合锁定纯净的、无交叉重叠的对齐训练标本输入")
        return X, y_clf, y_reg
    return None, None, None