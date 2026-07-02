# -*- coding: utf-8 -*-
"""
step4/step4_labeling_weighting.py
Phase 4 总任务调度编排器 [2026 生产级安全闭环 Release]
完美集成跨市场双格式物理持久化与纯现货多头方向硬红线审计
"""

import logging
import os
import pandas as pd
from .config import *
from .label_builder import build_dual_track_labels
from .sample_weighting import compute_exponential_decay_weights

logger = logging.getLogger("LabelingWeighting")

def execute(pipeline_context: dict) -> dict:
    """
    Phase 4 物理核心入口
    """
    logger.info("=" * 60)
    logger.info("Phase 4: 双轨标签构建与样本加权（分布式原子架构 - 缓存与纯现货多头版）")
    logger.info("=" * 60)

    # 1. 抽取主干隔离切片边界
    config = pipeline_context.get("config", {})
    bus = pipeline_context["data_bus"]
    slices = pipeline_context.get("slices", {})
    train_dates_raw = slices.get("Train-A", [])

    if not train_dates_raw:
        raise ValueError("❌ 物理熔断：Train-A 时间切片数据为空，请先调度 Phase 2 模块。")

    train_dates = pd.DatetimeIndex(train_dates_raw).tz_localize(None)
    t_max = train_dates[-1]
    date_str = t_max.strftime("%Y%m%d")

    # 2. 会话生命周期缓存管理器 (严格响应 README.md 设计红线)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Phase Result"))
    parquet_dir = os.path.join(base_dir, "parquet", "Phase 4", date_str)
    feather_dir = os.path.join(base_dir, "feather", "Phase 4", date_str)
    parquet_path = os.path.join(parquet_dir, "step4_result.parquet")
    feather_path = os.path.join(feather_dir, "step4_result.feather")

    update_required = config.get("update_required", False)

    if os.path.exists(parquet_path) and os.path.exists(feather_path) and not update_required:
        logger.info(f"💾 [Cache Hit] 侦测到本地已存在历史固化持久化文件，且当前为 (Update Not Need) 模式。启动流式秒级载入...")
        try:
            df_cache = pd.read_parquet(parquet_path)
            
            # 高性能向量化恢复会话上下文
            dates_col = pd.to_datetime(df_cache["date"])
            assets_col = df_cache["asset"].astype(str)
            keys = list(zip(dates_col, assets_col))
            
            y_clf_all = dict(zip(keys, df_cache["y_clf"].astype(int)))
            y_reg_all = dict(zip(keys, df_cache["y_reg"].astype(float)))
            sample_weights = dict(zip(keys, df_cache["weight"].astype(float)))
            
            pipeline_context["y_clf_all"] = y_clf_all
            pipeline_context["y_reg_all"] = y_reg_all
            pipeline_context["sample_weights"] = sample_weights
            pipeline_context["borrowable_stocks"] = set()  # 纯多头合规，融券池无条件彻底清空
            pipeline_context["labeling_ready"] = True
            
            pipeline_context["_phase_status"] = pipeline_context.get("_phase_status", {})
            pipeline_context["_phase_status"]["step4_labeling_weighting"] = "success_via_cache"
            logger.info(f"🚀 成功自缓存反序列化加载 {len(y_reg_all)} 条横截面样本，完美阻断重复运算。")
            return pipeline_context
        except Exception as cache_err:
            logger.error(f"⚠️ 缓存文件结构损坏: {cache_err}，自动降级为正常重新计算流程。")

    # 3. 超参指纹清洗融合
    lambda_decay = config.get("lambda_decay", DEFAULT_LAMBDA_DECAY)
    vol_window = config.get("vol_window", DEFAULT_VOL_WINDOW)
    threshold_multiplier = config.get("threshold_multiplier", DEFAULT_THRESHOLD_MULTIPLIER)
    min_valid_obs = config.get("min_vol_obs", DEFAULT_MIN_VOL_OBS)
    crisis_windows_cfg = config.get("crisis_windows", CRISIS_WINDOWS)
    crisis_noise_weight_cfg = config.get("crisis_noise_weight", CRISIS_NOISE_WEIGHT)

    # 4. 确定资产宇宙快照
    assets = pipeline_context.get("assets")
    if not assets:
        assets = bus.get_universe()
        pipeline_context["assets"] = assets
    logger.info(f"目标资产池横截面对齐规模: {len(assets)} 只标的")

    # 5. 编译现货纯多头双轨标签面板
    y_clf_all, y_reg_all = build_dual_track_labels(
        assets=assets,
        train_dates=train_dates,
        bus=bus,
        vol_window=vol_window,
        min_valid_obs=min_valid_obs,
        threshold_multiplier=threshold_multiplier,
        global_vol_fallback=GLOBAL_VOL_MEDIAN_FALLBACK
    )

    # 6. 编译非平稳时序复合加权矩阵 (加入黑天鹅平抑)
    sample_weights = compute_exponential_decay_weights(
        sample_keys=y_reg_all.keys(),
        t_max=t_max,
        lambda_decay=lambda_decay,
        crisis_windows=crisis_windows_cfg,
        crisis_noise_weight=crisis_noise_weight_cfg
    )

    # 7. 生产安全性合规性静态断言审计
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

    # 8. 双格式结果同步持久化存盘 (完美落地 README 规范要求)
    logger.info("启动会话缓存层物理双格式持久化归档...")
    try:
        records = []
        for key in y_reg_all.keys():
            dt, sym = key
            records.append({
                "date": dt,
                "asset": sym,
                "y_clf": y_clf_all[key],
                "y_reg": y_reg_all[key],
                "weight": sample_weights[key]
            })
        
        df_save = pd.DataFrame(records)
        if not df_save.empty:
            # A. 固化 Parquet 格式 (供 Python 脚本高速内消)
            os.makedirs(parquet_dir, exist_ok=True)
            df_save.to_parquet(parquet_path, index=False)
            
            # B. 固化 Feather 格式 (供未来 C 语言跨语言消费)
            os.makedirs(feather_dir, exist_ok=True)
            df_save.to_feather(feather_path)
            
            logger.info(f"💾 缓存成功双固化到物理磁盘：\n - [Python] Parquet 路径: {parquet_path}\n - [C-Engine] Feather 路径: {feather_path}")
    except Exception as save_err:
        logger.error(f"⚠️ 警告：双格式缓存序列化存盘失败: {save_err} (但不干扰当前进程运行)")

    # 9. 管道上下文状态安全移交 downstream
    pipeline_context["y_clf_all"] = y_clf_all
    pipeline_context["y_reg_all"] = y_reg_all
    pipeline_context["sample_weights"] = sample_weights
    pipeline_context["borrowable_stocks"] = set()  # 强制使融券池为空，阻绝后续策略任何借贷做空的妄想
    pipeline_context["labeling_ready"] = True

    pipeline_context["_phase_status"] = pipeline_context.get("_phase_status", {})
    pipeline_context["_phase_status"]["step4_labeling_weighting"] = "success"

    return pipeline_context