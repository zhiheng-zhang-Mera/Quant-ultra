# -*- coding: utf-8 -*-
"""
step5/dataset_utils.py
自适应数据集构建工具：根据资产在不同切片内的真实上市存活率，动态决定是否接纳该资产进入训练或验证。
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger("ModelTraining.Dataset")

def build_partition_dataset(context: dict, partition_name: str):
    """
    自适应分段准入数据集构建器
    实现：训练期时长不足则剔除，验证/回测期数据可用则接纳的闭环。
    
    参数:
        context: 管道全局上下文，必须包含 'trading_days_dt'、'fractional_features_cube'、'alive_mask_matrix'、
                 'assets'、'slices'、'y_clf_all'、'y_reg_all'
        partition_name: 切片名称，如 'Train-A', 'Train-B1', 'Train-B2', 'Validation'
    
    返回:
        X: 特征矩阵 (n_samples, F)
        y_clf: 分类标签 (n_samples,)
        y_reg: 回归标签 (n_samples,)
        若无可接纳样本，则返回 (None, None, None)
    """
    slices = context['slices']
    target_dates = slices.get(partition_name)
    if target_dates is None or len(target_dates) == 0:
        logger.warning(f"切片 {partition_name} 在 slices 中不存在或为空。")
        return None, None, None

    assets = context['assets']
    diff_cube = context.get('fractional_features_cube')
    alive_mask_matrix = context.get('alive_mask_matrix')
    if diff_cube is None or alive_mask_matrix is None:
        raise RuntimeError("缺少 fraction_features_cube 或 alive_mask_matrix，请先执行 Step 5.2。")

    # 获取全局日历
    global_calendar = context.get('trading_days_dt')
    if global_calendar is None:
        raise RuntimeError("context 中缺少 trading_days_dt，无法进行日期索引映射。")

    # 将目标日期转为 DatetimeIndex 并查找索引
    target_dt = pd.DatetimeIndex(target_dates)
    slice_idx = np.where(global_calendar.isin(target_dt))[0]
    if len(slice_idx) == 0:
        raise ValueError(f"切片 {partition_name} 在全局日历中未找到匹配的日期。")

    sub_cube = diff_cube[slice_idx, :, :]          # (t_slice, N, F)
    sub_alive = alive_mask_matrix[slice_idx, :]    # (t_slice, N)

    # 计算每只股票在该切片内的存活率
    alive_ratios = sub_alive.mean(axis=0)          # (N,)

    # 动态设定准入阈值：训练集严格（80%），验证/测试集宽松（10%）
    if "Train" in partition_name:
        MIN_ALIVE_RATIO = 0.80
    else:
        MIN_ALIVE_RATIO = 0.10

    X_list, y_clf_list, y_reg_list = [], [], []
    admitted_count = 0

    y_clf_all = context.get('y_clf_all', {})
    y_reg_all = context.get('y_reg_all', {})

    # 遍历资产，过滤存活率不足的标的
    for idx, sym in enumerate(assets):
        if alive_ratios[idx] < MIN_ALIVE_RATIO:
            continue

        # 提取特征 (t_slice, F)
        feat_series = sub_cube[:, idx, :]

        # 提取该标的在切片内每个日期的标签
        y_c_list, y_r_list = [], []
        valid_indices = []  # 存储有效日期在子切片内的索引

        for i, dt in enumerate(target_dt):
            dt_str = dt.strftime("%Y-%m-%d")
            # 尝试多种 key 匹配
            yc, yr = None, None
            for key in [(dt, sym), (dt_str, sym), (pd.Timestamp(dt), sym)]:
                if key in y_clf_all:
                    yc = y_clf_all[key]
                if key in y_reg_all:
                    yr = y_reg_all[key]
                if yc is not None and yr is not None:
                    break
            # 同时需要该日活体标记为 True
            if yc is not None and yr is not None and sub_alive[i, idx]:
                y_c_list.append(yc)
                y_r_list.append(yr)
                valid_indices.append(i)

        if len(y_c_list) >= 10:  # 至少 10 个观测值，防止过少样本
            X_list.append(feat_series[valid_indices])
            y_clf_list.append(np.array(y_c_list))
            y_reg_list.append(np.array(y_r_list))
            admitted_count += 1

    if X_list:
        X_final = np.vstack(X_list)
        y_c_final = np.concatenate(y_clf_list)
        y_r_final = np.concatenate(y_reg_list)
        logger.info(f"📊 [{partition_name}] 准入审计完毕！全局池 {len(assets)} 只中，"
                    f"本阶段实际接纳有效活体资产: {admitted_count} 只，合并特征行数: {len(X_final)}")
        return X_final, y_c_final, y_r_final
    else:
        logger.critical(f"❌ 灾难！[{partition_name}] 没有任何资产满足时长准入条件！")
        return None, None, None