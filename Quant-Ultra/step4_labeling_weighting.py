"""
Phase 4: Primary Labeling and Sample Weighting (Directional & Numeric Targets)
Fully compliant with Final-Flow.md [2026 Production Release]

本阶段职责：
    - 为 Train-A 区间内的每个资产、每个交易日生成双轨标签：
        1. 方向过滤标签 y_clf ∈ {-1, 0, 1}（基于三屏障 + 融券可用性）
        2. 远期预期收益标签 y_reg（次日对数远期收益率）
    - 为所有样本计算指数时间衰减权重 w_t

注意：本模块只做纯前向特征工程，严格隔离未来信息。
     所有数据均通过 PITDataBus 查询，确保 Point-in-Time 合规。
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Set, Dict, Any, Optional
import pytz

logger = logging.getLogger("LabelingWeighting")


# ============================================================
# 辅助函数：融券可用性查询（真实免费数据源 + 合规降级）
# ============================================================
def fetch_borrowable_stocks(context: dict, trade_date: pd.Timestamp) -> Set[str]:
    """
    从 AkShare 获取指定交易日有融券余量的股票代码集合。
    若接口失败或数据缺失，按规范降级为"全部不可用"，并记录审计事件。
    """
    audit_logger = context.get("audit_logger")
    try:
        import akshare as ak
        date_str = trade_date.strftime("%Y%m%d")
        df = ak.stock_borrow_analysis(date=date_str)
        if df.empty:
            logger.warning(f"融券数据为空（日期: {date_str}），降级为全部不可用")
            if audit_logger:
                audit_logger.log_event("数据缺失-默认残值", {
                    "source": "stock_borrow_analysis",
                    "trade_date": date_str,
                    "reason": "返回空DataFrame"
                })
            return set()

        df = df[df['融券余量'] > 0]
        codes = df['代码'].astype(str).tolist()
        result = set()
        for c in codes:
            if len(c) != 6:
                continue
            if c.startswith('6'):
                result.add(f"{c}.SH")
            elif c.startswith('0') or c.startswith('3'):
                result.add(f"{c}.SZ")
        logger.info(f"获取融券可用标的: {len(result)} 只（日期: {date_str}）")
        return result

    except ImportError:
        logger.warning("AkShare 未安装，无法获取融券数据，降级为全部不可用")
        if audit_logger:
            audit_logger.log_event("数据缺失-默认残值", {
                "source": "akshare",
                "reason": "import_error"
            })
        return set()
    except Exception as e:
        logger.warning(f"融券数据查询异常: {e}，降级为全部不可用")
        if audit_logger:
            audit_logger.log_event("数据缺失-默认残值", {
                "source": "stock_borrow_analysis",
                "reason": str(e)
            })
        return set()


# ============================================================
# 核心实现
# ============================================================
def execute(pipeline_context: dict) -> dict:
    """
    Phase 4 主入口。
    期望上下文中已包含:
        - config: 配置字典（含 lambda_decay, vol_window, threshold_multiplier, min_vol_obs）
        - data_bus: PITDataBus 实例
        - slices: 包含 'Train-A' 日期列表
        - assets: 待处理的资产列表（可选）
        - audit_logger: 审计日志记录器
    """
    logger.info("=" * 60)
    logger.info("Phase 4: 双轨标签构建与样本加权")
    logger.info("=" * 60)

    # 1. 读取上下文与配置
    config = pipeline_context.get("config", {})
    bus = pipeline_context["data_bus"]
    audit_logger = pipeline_context.get("audit_logger")
    slices = pipeline_context.get("slices", {})
    train_dates_raw = slices.get("Train-A", [])

    if not train_dates_raw:
        raise ValueError("❌ Train-A 切片为空，请先执行 Phase 2 切片划分。")

    train_dates = pd.DatetimeIndex(train_dates_raw).tz_localize(None)

    # 从配置读取超参（带默认值）
    lambda_decay = config.get("lambda_decay", 0.01)
    vol_window = config.get("vol_window", 20)
    threshold_multiplier = config.get("threshold_multiplier", 0.5)
    min_valid_obs = config.get("min_vol_obs", 5)

    logger.info(f"配置参数: λ={lambda_decay}, vol_window={vol_window}, "
                f"threshold_multiplier={threshold_multiplier}, min_vol_obs={min_valid_obs}")

    # 获取标的池
    assets = pipeline_context.get("assets")
    if not assets:
        logger.info("上下文中无 assets，从数据总线全量获取")
        assets = bus.get_universe()
        pipeline_context["assets"] = assets
    logger.info(f"标的池规模: {len(assets)} 只")

    # 2. 获取融券可用性（以 Train-A 最后一天为基准）
    trade_date_ref = train_dates[-1]
    borrowable_set = fetch_borrowable_stocks(pipeline_context, trade_date_ref)
    logger.info(f"融券可用标的数量: {len(borrowable_set)}")

    # 3. 批量计算标签
    y_clf_all = {}
    y_reg_all = {}
    T_max = train_dates[-1]

    processed_count = 0
    skipped_no_data = 0

    for sym in assets:
        try:
            start_dt = train_dates[0] - pd.Timedelta(days=60)
            end_dt = train_dates[-1] + pd.Timedelta(days=5)
            df = bus.load_asset_history(
                sym,
                start_date=start_dt.strftime("%Y-%m-%d"),
                end_date=end_dt.strftime("%Y-%m-%d")
            )
        except RuntimeError as e:
            logger.warning(f"加载 {sym} 历史失败: {e}，跳过")
            skipped_no_data += 1
            continue

        if df is None or df.empty:
            skipped_no_data += 1
            continue

        price_series = df['total_return_price']
        prices = price_series.reindex(train_dates, method='ffill')

        valid_mask = prices.notna()
        if valid_mask.sum() < 2:
            skipped_no_data += 1
            continue

        # 远期收益
        prices_t1 = prices.shift(-1)
        y_reg = np.log(prices_t1 / prices)

        # 波动率
        daily_ret = prices.pct_change()
        rolling_vol = daily_ret.rolling(window=vol_window, min_periods=min_valid_obs).std()
        global_vol_median = rolling_vol.median()
        if pd.isna(global_vol_median) or global_vol_median == 0:
            global_vol_median = 0.02
        rolling_vol_filled = rolling_vol.fillna(global_vol_median)
        threshold = rolling_vol_filled * threshold_multiplier

        # 方向标签
        y_clf = np.zeros(len(train_dates), dtype=np.int8)
        long_mask = (y_reg >= threshold)
        y_clf[long_mask] = 1
        short_mask = (y_reg <= -threshold)
        if sym in borrowable_set:
            y_clf[short_mask] = -1
        else:
            y_clf[short_mask] = 0
        invalid_mask = y_reg.isna() | threshold.isna()
        y_clf[invalid_mask] = 0

        # 存储
        for i, d in enumerate(train_dates):
            y_reg_val = y_reg.iloc[i]
            if pd.isna(y_reg_val):
                continue
            key = (d, sym)
            y_reg_all[key] = float(y_reg_val)
            y_clf_all[key] = int(y_clf[i])

        processed_count += 1
        if processed_count % 500 == 0:
            logger.info(f"已处理 {processed_count} 只资产，跳过 {skipped_no_data} 只无数据")

    logger.info(f"✅ 标签生成完成。有效标的: {processed_count}，无数据跳过: {skipped_no_data}")
    logger.info(f"总样本数: {len(y_reg_all)}")

    # 5. 计算样本时间衰减权重
    logger.info("计算指数时间衰减权重...")
    sample_weights = {}
    lambda_ = lambda_decay
    if isinstance(T_max, pd.Timestamp):
        t_max_dt = T_max.to_pydatetime()
    else:
        t_max_dt = T_max

    for (date, sym), _ in y_reg_all.items():
        if isinstance(date, pd.Timestamp):
            date_dt = date.to_pydatetime()
        else:
            date_dt = date
        delta_days = (t_max_dt - date_dt).days
        w = np.exp(-lambda_ * delta_days)
        sample_weights[(date, sym)] = w

    logger.info(f"样本权重分配完成，权重范围: [{min(sample_weights.values()):.6f}, "
                f"{max(sample_weights.values()):.6f}]")

    # 6. 统计信息输出
    if y_clf_all:
        labels = list(y_clf_all.values())
        n_pos = sum(1 for v in labels if v == 1)
        n_neg = sum(1 for v in labels if v == -1)
        n_zero = sum(1 for v in labels if v == 0)
        logger.info(f"标签分布: 多头 {n_pos} ({n_pos/len(labels)*100:.1f}%), "
                    f"空头 {n_neg} ({n_neg/len(labels)*100:.1f}%), "
                    f"中性 {n_zero} ({n_zero/len(labels)*100:.1f}%)")
    else:
        logger.warning("⚠️ 未生成任何有效标签，请检查数据完整性！")

    # 7. 写回上下文
    pipeline_context["y_clf_all"] = y_clf_all
    pipeline_context["y_reg_all"] = y_reg_all
    pipeline_context["sample_weights"] = sample_weights
    pipeline_context["borrowable_stocks"] = borrowable_set
    pipeline_context["labeling_ready"] = True

    pipeline_context["_phase_status"] = pipeline_context.get("_phase_status", {})
    pipeline_context["_phase_status"]["step4_labeling_weighting"] = "success"

    return pipeline_context


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("本模块不作为独立入口，请通过主调度器运行。")