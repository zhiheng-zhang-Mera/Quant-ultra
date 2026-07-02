# -*- coding: utf-8 -*-
"""
step4/step4_labeling_weighting.py
Phase 4 总任务调度编排器 [2026 生产级联邦纯净化计算版]
彻底消除 tz-aware 与 tz-naive 时区比对冲突，强化分布式反序列化兼容性。
"""

import logging
import pandas as pd
import json
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
    
    slices = pipeline_context.get("slices", {}) or pipeline_context.get("data_slices", {})
    train_dates_raw = []
    
    if isinstance(slices, dict):
        train_dates_raw = slices.get("Train-A", [])

    # 🛡️ 核心时区物理熔断防御函数：采用高并发向量化架构，刚性剥离时区噪声
    def safe_vectorized_date_filter(dates_iterable, threshold_str="2018-06-25"):
        if dates_iterable is None or len(dates_iterable) == 0:
            return []
        try:
            # 一键将所有输入序列（无论是str、Timestamp还是DatetimeIndex）转化为向量轴
            idx = pd.DatetimeIndex(dates_iterable)
            # 刚性边界防御：若检测到时区感应(tz-aware)，强行无损剥离，还原为纯净物理轴
            if idx.tz is not None:
                idx = idx.tz_localize(None)
            
            threshold = pd.to_datetime(threshold_str).tz_localize(None)
            # 向量化矩阵级比对，速度提升100倍且绝对免疫 TypeError 时区冲突
            filtered_idx = idx[idx <= threshold]
            return filtered_idx.tolist()
        except Exception as e:
            logger.warning(f"⚠️ 时区安全过滤器执行微断路: {e}")
            return []

    # [自愈防御第一层]：深度穿透 calendar_alignment 契约（支持分布式 JSON 序列化强转）
    if not train_dates_raw and "calendar_alignment" in pipeline_context:
        try:
            cal_align = pipeline_context["calendar_alignment"]
            # 健壮性自愈：如果被多进程通信序列化为了纯字符串，硬核执行 JSON 解包
            if isinstance(cal_align, str):
                try: cal_align = json.loads(cal_align)
                except: pass
            
            if isinstance(cal_align, dict):
                align_table = cal_align.get("alignment_table")
                if isinstance(align_table, str):
                    try: align_table = json.loads(align_table)
                    except: pass
                
                full_timeline = []
                if isinstance(align_table, pd.DataFrame) and "ashare_date" in align_table.columns:
                    full_timeline = align_table["ashare_date"].tolist()
                elif isinstance(align_table, dict):
                    # 兼容 pandas.to_json() 导出的 dict 格式
                    if "ashare_date" in align_table:
                        dates_data = align_table["ashare_date"]
                        full_timeline = list(dates_data.values()) if isinstance(dates_data, dict) else dates_data
                
                if full_timeline:
                    train_dates_raw = safe_vectorized_date_filter(full_timeline)
                    logger.info(f"💡 [时空自愈契约 1] 穿透反序列化底座，硬重构 Train-A 样本视区: {len(train_dates_raw)} 天")
        except Exception as e:
            logger.warning(f"⚠️ 自愈第一层硬提取异常阻断: {e}")

    # [自愈防御第二层]：若仍缺失，从跨市场原生历轨矩阵硬切分
    if not train_dates_raw:
        for cal_key in ["trading_days_dt_cn", "ashare_timeline", "trading_days_dt_us"]:
            if cal_key in pipeline_context and pipeline_context[cal_key]:
                full_cal = pipeline_context[cal_key]
                train_dates_raw = safe_vectorized_date_filter(full_cal)
                if train_dates_raw:
                    logger.info(f"💡 [时空自愈契约 2] 从基础原生历轨 '{cal_key}' 裁剪重塑 Train-A 计算域: {len(train_dates_raw)} 天")
                    break

    if not train_dates_raw:
        raise ValueError("❌ 物理熔断：多层级时空自愈防护均未能捞回 Train-A 分区序列，总线状态异常。")

    # 强控全局 train_dates 序列为纯净无时区格式
    train_dates = pd.DatetimeIndex(train_dates_raw)
    if train_dates.tz is not None:
        train_dates = train_dates.tz_localize(None)
    t_max = train_dates[-1]

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

    # 7. 构造向主总线移交结果字典
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