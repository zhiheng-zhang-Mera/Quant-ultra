# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_4 Main Entry Tracker & Timezone Neutralizer Orchestrator
"""
import logging
import pandas as pd
import json
from Phase_4.config import *
from Phase_4.label_builder import build_dual_track_labels
from Phase_4.sample_weighting import compute_exponential_decay_weights

logger = logging.getLogger("LabelingWeighting")

def execute(pipeline_context: dict) -> dict:
    logger.info("=" * 60)
    logger.info("[OP] Activate Phase_4 Core Pipeline Orchestrator | [SOURCE] Central Pipeline Signal Matrix | [RESULT] Ingesting temporal cross-sections | [SIGNIFICANCE] Resolves timezone sensor aware mismatch flaws and compiles whitebox target frames")
    logger.info("[操作] 激活 Phase_4 核心调度主控编排器 | [来源] 中央工作流信号大干线 | [结果] 正在导入时空跨度切片 | [意义] 消除时区感应(tz-aware)导致的矩阵比对不对齐冲突，编译白盒机器学习样本底座")
    logger.info("=" * 60)

    config = pipeline_context.get("config", {})
    bus = pipeline_context["data_bus"]
    slices = pipeline_context.get("slices", {}) or pipeline_context.get("data_slices", {})
    train_dates_raw = []
    
    if isinstance(slices, dict): train_dates_raw = slices.get("Train-A", [])

    def safe_vectorized_date_filter(dates_iterable, threshold_str="2018-06-25"):
        if dates_iterable is None or len(dates_iterable) == 0: return []
        try:
            idx = pd.DatetimeIndex(dates_iterable)
            if idx.tz is not None: idx = idx.tz_localize(None)
            threshold = pd.to_datetime(threshold_str).tz_localize(None)
            return idx[idx <= threshold].tolist()
        except Exception as e:
            logger.warning(f"Timezone vectorization filter micro-breakout exception: {e}"); return []

    if not train_dates_raw and "calendar_alignment" in pipeline_context:
        try:
            cal_align = pipeline_context["calendar_alignment"]
            if isinstance(cal_align, str): cal_align = json.loads(cal_align)
            if isinstance(cal_align, dict):
                align_table = cal_align.get("alignment_table")
                if isinstance(align_table, str): align_table = json.loads(align_table)
                full_timeline = []
                if isinstance(align_table, pd.DataFrame) and "ashare_date" in align_table.columns:
                    full_timeline = align_table["ashare_date"].tolist()
                elif isinstance(align_table, dict) and "ashare_date" in align_table:
                    dates_data = align_table["ashare_date"]
                    full_timeline = list(dates_data.values()) if isinstance(dates_data, dict) else dates_data
                if full_timeline: train_dates_raw = safe_vectorized_date_filter(full_timeline)
        except Exception as e: logger.warning(f"Hydration fail on contract recovery level 1: {e}")

    if not train_dates_raw:
        for cal_key in ["trading_days_dt_cn", "ashare_timeline", "trading_days_dt_us"]:
            if cal_key in pipeline_context and pipeline_context[cal_key]:
                train_dates_raw = safe_vectorized_date_filter(pipeline_context[cal_key]); break

    if not train_dates_raw: raise ValueError("Hard Halt: Adaptive chronology fails to heal Train-A axis.")

    train_dates = pd.DatetimeIndex(train_dates_raw)
    if train_dates.tz is not None: train_dates = train_dates.tz_localize(None)
    t_max = train_dates[-1]

    lambda_decay = config.get("lambda_decay", DEFAULT_LAMBDA_DECAY)
    vol_window = config.get("vol_window", DEFAULT_VOL_WINDOW)
    threshold_multiplier = config.get("threshold_multiplier", DEFAULT_THRESHOLD_MULTIPLIER)
    min_valid_obs = config.get("min_vol_obs", DEFAULT_MIN_VOL_OBS)
    crisis_windows_cfg = config.get("crisis_windows", CRISIS_WINDOWS)
    crisis_noise_weight_cfg = config.get("crisis_noise_weight", CRISIS_NOISE_WEIGHT)

    assets = pipeline_context.get("assets") or bus.get_universe()
    
    y_clf_all, y_reg_all = build_dual_track_labels(
        assets=assets, train_dates=train_dates, bus=bus, vol_window=vol_window,
        min_valid_obs=min_valid_obs, threshold_multiplier=threshold_multiplier, global_vol_fallback=GLOBAL_VOL_MEDIAN_FALLBACK
    )

    sample_weights = compute_exponential_decay_weights(
        sample_keys=y_reg_all.keys(), t_max=t_max, lambda_decay=lambda_decay,
        crisis_windows=crisis_windows_cfg, crisis_noise_weight=crisis_noise_weight_cfg
    )

    if y_clf_all:
        labels = list(y_clf_all.values())
        n_pos, n_neg, n_zero = sum(1 for v in labels if v == 1), sum(1 for v in labels if v == -1), sum(1 for v in labels if v == 0)
        
        logger.info("[OP] Execute Static Assertion Audit | [SOURCE] Finished Target Label Frame Arrays | [RESULT] Distribution: Long(1)=%s, Short(-1)=%s, Neutral(0)=%s | [SIGNIFICANCE] Regulatory safety verification against unauthorized derivative short leakage", n_pos, n_neg, n_zero)
        logger.info("[操作] 执行静态断言合规审计 | [来源] 已编译完成的目标标签总体阵列 | [结果] 标签分布: 多头(1)=%s, 空头(-1)=%s, 中性(0)=%s | [意义] 现货风控红线硬校验，严防未经授权的恶意做空(-1)信号污染样本泄露")
        
        if n_neg > 0: raise RuntimeError("Compliance breach: Unauthorized short selling tokens detected (-1) in long-only spot base configuration.")

    return {
        "assets": assets, "y_clf_all": y_clf_all, "y_reg_all": y_reg_all,
        "sample_weights": sample_weights, "borrowable_stocks": set(), "labeling_ready": True
    }