# -*- coding: utf-8 -*-
"""
step4/step4_labeling_weighting.py
Phase 4 总任务调度编排器 [2026 生产级联邦纯净化计算版]
彻底移除本地手写读写盘，生命周期生命周期全量上交 main.py 托管
"""

import logging
import pandas as pd
from .config import *
from .label_builder import build_dual_track_labels
from .sample_weighting import compute_exponential_decay_weights

logger = logging.getLogger("LabelingWeighting")

def execute(pipeline_context: dict) -> dict:
    """
    Phase 4 纯内存原子计算核心入口
    """
    logger.info("=" * 60)
    logger.info("Phase 4: 双轨标签构建与样本加权（分布式纯净架构版）")
    logger.info("=" * 60)

    # 1. 提取隔离切片边界
    config = pipeline_context.get("config", {})
    bus = pipeline_context["data_bus"]
    slices = pipeline_context.get("slices", {})
    train_dates_raw = slices.get("Train-A", [])

    if not train_dates_raw:
        raise ValueError("❌ 物理熔断：Train-A 时间切片数据为空，请先调度 Phase 2 模块。")

    train_dates = pd.DatetimeIndex(train_dates_raw).tz_localize(None)
    t_max = train_dates[-1]

    # 💡 架构演进：手写的本地 parquet_path/feather_path 读写缓存层已全量物理移除！

    # 2. 超参指纹清洗融合
    lambda_decay = config.get("lambda_decay", DEFAULT_LAMBDA_DECAY)
    vol_window = config.get("vol_window", DEFAULT_VOL_WINDOW)
    threshold_multiplier = config.get("threshold_multiplier", DEFAULT_THRESHOLD_MULTIPLIER)
    min_valid_obs = config.get("min_vol_obs", DEFAULT_MIN_VOL_OBS)
    crisis_windows_cfg = config.get("crisis_windows", CRISIS_WINDOWS)
    crisis_noise_weight_cfg = config.get("crisis_noise_weight", CRISIS_NOISE_WEIGHT)

    # 3. 确定资产宇宙快照
    assets = pipeline_context.get("assets")
    if not assets:
        assets = bus.get_universe()
    logger.info(f"目标资产池横截面对齐规模: {len(assets)} 只标的")

    # 4. 编译现货纯多头双轨标签面板
    y_clf_all, y_reg_all = build_dual_track_labels(
        assets=assets,
        train_dates=train_dates,
        bus=bus,
        vol_window=vol_window,
        min_valid_obs=min_valid_obs,
        threshold_multiplier=threshold_multiplier,
        global_vol_fallback=GLOBAL_VOL_MEDIAN_FALLBACK
    )

    # 5. 编译非平稳时序复合加权矩阵 (加入黑天鹅平抑)
    sample_weights = compute_exponential_decay_weights(
        sample_keys=y_reg_all.keys(),
        t_max=t_max,
        lambda_decay=lambda_decay,
        crisis_windows=crisis_windows_cfg,
        crisis_noise_weight=crisis_noise_weight_cfg
    )

    # 6. 生产安全性合规性静态断言审计
    if y_clf_all:
        labels = list(y_clf_all.values())
        n_pos = sum(1 for v in labels if v == 1)
        n_neg = sum(1 for v in labels if v == -1)
        n_zero = sum(1 for v in labels if v == 0)
        total = len(labels)
        logger.info(f"📊 训练集标签统计审计: 多头(1)={n_pos} [{n_pos/total*100:.2f}%], "
                    f"空头(-1)={n_neg} [{n_neg/total*100:.2f}%], "
                    f"中性(0)={n_zero} [{n_zero/total*100:.2f}%]")
        
        if n_neg > 0:
            raise RuntimeError("❌ [安全防线异常阻断] 底端标签生成了非法的做空信号 (-1)！严重偏离纯多头现货风控红线！")
    else:
        logger.warning("⚠️ 警告：底层未析出任何有效的联合实值标签对，请核查 DataBus 状态！")

    # 7. 构造向主总线移交的纯脏增量结果字典 (由 main.py 进行无损自动化双格式写盘)
    result_update = {
        "assets": assets,
        "y_clf_all": y_clf_all,
        "y_reg_all": y_reg_all,
        "sample_weights": sample_weights,
        "borrowable_stocks": set(),  # 强控融券池为空
        "labeling_ready": True
    }
    
    logger.info("✅ Step 4 纯内存分布式标签面板编译完成，移交主控总线中心化持久化。")
    return result_update