# -*- coding: utf-8 -*-
"""
step4/step4_labeling_weighting.py
Phase 4 总任务调度编排器 [202 production Release]
"""

import logging
import pandas as pd
from .config import *
from .borrow_manager import fetch_borrowable_stocks
from .label_builder import build_dual_track_labels
from .sample_weighting import compute_exponential_decay_weights

logger = logging.getLogger("LabelingWeighting")

def execute(pipeline_context: dict) -> dict:
    """
    Phase 4 核心物理执行入口
    """
    logger.info("=" * 60)
    logger.info("Phase 4: 双轨标签构建与样本加权（分布式原子架构）")
    logger.info("=" * 60)

    # 1. 检索全局上下文与总线
    config = pipeline_context.get("config", {})
    bus = pipeline_context["data_bus"]
    slices = pipeline_context.get("slices", {})
    train_dates_raw = slices.get("Train-A", [])

    if not train_dates_raw:
        raise ValueError("❌ Train-A 切片数据为空，请先执行 Phase 2 的时空轴物理隔离切分。")

    train_dates = pd.DatetimeIndex(train_dates_raw).tz_localize(None)

    # 2. 动态融合局部超参（允许外部配置覆盖本地默认值）
    lambda_decay = config.get("lambda_decay", DEFAULT_LAMBDA_DECAY)
    vol_window = config.get("vol_window", DEFAULT_VOL_WINDOW)
    threshold_multiplier = config.get("threshold_multiplier", DEFAULT_THRESHOLD_MULTIPLIER)
    min_valid_obs = config.get("min_vol_obs", DEFAULT_MIN_VOL_OBS)

    logger.info(f"融合调参指纹: λ={lambda_decay}, vol_window={vol_window}, "
                f"threshold_multiplier={threshold_multiplier}, min_vol_obs={min_valid_obs}")

    # 3. 确定资产宇宙
    assets = pipeline_context.get("assets")
    if not assets:
        logger.info("上下文中无可用资产池快照，启动总线级全量检测...")
        assets = bus.get_universe()
        pipeline_context["assets"] = assets
    logger.info(f"目标标的池交叉对齐规模: {len(assets)} 只")

    # 4. 执行融券可用性刺探
    t_max = train_dates[-1]
    borrowable_set = fetch_borrowable_stocks(pipeline_context, t_max)
    logger.info(f"当前截面融券可用对冲标的数: {len(borrowable_set)}")

    # 5. 编译双轨标签面板
    y_clf_all, y_reg_all = build_dual_track_labels(
        assets=assets,
        train_dates=train_dates,
        bus=bus,
        borrowable_set=borrowable_set,
        vol_window=vol_window,
        min_valid_obs=min_valid_obs,
        threshold_multiplier=threshold_multiplier,
        global_vol_fallback=GLOBAL_VOL_MEDIAN_FALLBACK
    )
    logger.info(f"总计成功清洗出有效联合样本数: {len(y_reg_all)}")

    # 6. 编译时间序列非平稳加权矩阵
    sample_weights = compute_exponential_decay_weights(
        sample_keys=y_reg_all.keys(),
        t_max=t_max,
        lambda_decay=lambda_decay
    )

    # 7. 统计学分布显式审计
    if y_clf_all:
        labels = list(y_clf_all.values())
        n_pos = sum(1 for v in labels if v == 1)
        n_neg = sum(1 for v in labels if v == -1)
        n_zero = sum(1 for v in labels if v == 0)
        total = len(labels)
        logger.info(f"📊 训练集标签横截面分布: 多头 {n_pos} ({n_pos/total*100:.1f}%), "
                    f"空头 {n_neg} ({n_neg/total*100:.1f}%), "
                    f"中性 {n_zero} ({n_zero/total*100:.1f}%)")
    else:
        logger.warning("⚠️ 警告：系统未生成任何有效物理标签！请排查前置清洗基建！")

    # 8. 生产环境状态安全落盘与交接
    pipeline_context["y_clf_all"] = y_clf_all
    pipeline_context["y_reg_all"] = y_reg_all
    pipeline_context["sample_weights"] = sample_weights
    pipeline_context["borrowable_stocks"] = borrowable_set
    pipeline_context["labeling_ready"] = True

    pipeline_context["_phase_status"] = pipeline_context.get("_phase_status", {})
    pipeline_context["_phase_status"]["step4_labeling_weighting"] = "success"

    return pipeline_context