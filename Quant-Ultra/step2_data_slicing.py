"""
Phase 2: Full-Flow Two-Tier Data Slicing and Isolation Architecture
Strictly implements: [Train-A] -> [Train-B1] -> [Train-B2] -> [Validation] -> [Test]
With Purge & Embargo triple-barrier constraints.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from statsmodels.tsa.stattools import acf
import logging
import math

logger = logging.getLogger("DataSlicing")

def step_2_1_moving_window_slicing(context: dict):
    """
    核心切分逻辑：考虑 Purge 与 Embargo 的硬性截断。
    规范强制 Embargo >= max(holding_period, significant_autocorrelation_lag)
    """
    print("[Step 2.1] Executing multi-stage physical window partition with Purge & Embargo.")
    
    trading_days_dt = context.get('trading_days_dt', [])
    if not trading_days_dt:
        # 兼容旧版上下文变量名
        trading_days_dt = context.get('trading_days', [])
        if not trading_days_dt:
            raise ValueError("交易日历 (trading_days_dt) 未在上下文总线中初始化。")
    
    # 策略持有期硬性定义（应与阶段七回测引擎保持一致）
    HOLDING_PERIOD = 5  # 默认 5 个交易日
    
    # 1. 计算该数据集上的最大显著自相关滞后（动态 Embargo 基准）
    # 注意：这里使用整个数据集的尾部来估算，但为了符合 PIT，我们只取 Train-A 起始后的数据
    # 更稳健：取前 60% 的数据来估算市场平均记忆性
    cutoff_idx = int(len(trading_days_dt) * 0.6)
    sample_days = trading_days_dt[:cutoff_idx]
    
    # 获取价格序列（从上下文总线查询，仅用于估算 ACF）
    bus = context.get('data_bus')
    assets = context.get('assets', [])
    if not assets:
        assets = ["600519.SH"]  # Fallback
    
    max_lag = 1
    try:
        # 使用第一个主力资产估算市场微观结构记忆性
        sym = assets[0]
        prices = []
        # 逆序查询样本区间内的价格
        for dt in sorted(sample_days, reverse=True):
            p = bus.query_by_pit(sym, dt, "total_return_price")
            if p:
                prices.append(p)
            if len(prices) > 50:  # 限制样本量防止过拟合
                break
        if len(prices) > 20:
            returns = np.diff(np.log(prices))
            # 计算 10 阶自相关，置信区间 95%
            try:
                acf_vals, confint = acf(returns, nlags=10, alpha=0.05, fft=False)
                # 安全提取显著滞后
                for i in range(1, len(acf_vals)):
                    if i < len(confint):
                        lower, upper = confint[i]
                        if acf_vals[i] < lower or acf_vals[i] > upper:
                            max_lag = max(max_lag, i)
            except Exception as e:
                logger.warning(f"ACF 计算异常，使用默认滞后 1。错误: {e}")
    except Exception as e:
        logger.warning(f"价格查询异常，使用默认滞后 1。错误: {e}")
    
    # 2. 刚性确定禁运区长度 (Embargo)
    embargo_window = max(HOLDING_PERIOD, max_lag)
    context['embargo_window'] = embargo_window
    context['holding_period'] = HOLDING_PERIOD
    print(f"[校准] 显著自相关最大滞后: {max_lag}, 持有期: {HOLDING_PERIOD}, 最终禁运窗口: {embargo_window}")

    # 3. 五段式切分（考虑 Purge & Embargo 截断）
    n = len(trading_days_dt)
    
    # 原始切分点 (占比)
    raw_split = {
        "Train-A_end": int(n * 0.50),
        "Train-B1_end": int(n * 0.60),
        "Train-B2_end": int(n * 0.70),
        "Validation_end": int(n * 0.85),
        "Test_end": n
    }
    
    # 应用 Purge/Embargo 截断：
    # 规范：交界处强制挖空。具体执行：左侧尾部截断 embargo_window，右侧头部截断 embargo_window
    # 即 [A_start : A_end - embargo] -> [B1_start + embargo : B1_end - embargo] -> ...
    # 对于首个区间，头部无需截断；对于末个区间，尾部无需截断
    
    slices = {}
    # Train-A: 从 0 到 raw_A_end - embargo (防止 A 的标签落入 B1)
    a_end = max(0, raw_split["Train-A_end"] - embargo_window)
    slices["Train-A"] = trading_days_dt[0:a_end]
    
    # Train-B1: 从 raw_A_end + embargo 到 raw_B1_end - embargo
    b1_start = min(raw_split["Train-A_end"] + embargo_window, raw_split["Train-B1_end"])
    b1_end = max(0, raw_split["Train-B1_end"] - embargo_window)
    slices["Train-B1"] = trading_days_dt[b1_start:b1_end] if b1_start < b1_end else []
    
    # Train-B2: 从 raw_B1_end + embargo 到 raw_B2_end - embargo
    b2_start = min(raw_split["Train-B1_end"] + embargo_window, raw_split["Train-B2_end"])
    b2_end = max(0, raw_split["Train-B2_end"] - embargo_window)
    slices["Train-B2"] = trading_days_dt[b2_start:b2_end] if b2_start < b2_end else []
    
    # Validation: 从 raw_B2_end + embargo 到 raw_Val_end - embargo
    val_start = min(raw_split["Train-B2_end"] + embargo_window, raw_split["Validation_end"])
    val_end = max(0, raw_split["Validation_end"] - embargo_window)
    slices["Validation"] = trading_days_dt[val_start:val_end] if val_start < val_end else []
    
    # Test: 从 raw_Val_end + embargo 到最后 (Test 无需尾部截断，因为无需预测更远的未来)
    test_start = min(raw_split["Validation_end"] + embargo_window, raw_split["Test_end"])
    slices["Test"] = trading_days_dt[test_start:raw_split["Test_end"]] if test_start < raw_split["Test_end"] else []
    
    # 存入上下文
    context['slices'] = slices
    context['raw_split_indices'] = raw_split  # 用于调试审计
    
    print("[切分结果]")
    for k, v in slices.items():
        print(f"   - {k}: {len(v)} 个交易日 (起始: {v[0] if v else 'N/A'}, 结束: {v[-1] if v else 'N/A'})")

def step_2_2_purge_and_embargo_validation(context: dict):
    """
    执行交尾审查：验证相邻区间是否真的物理隔离。
    若存在重叠或边界间距不足，抛出硬异常。
    """
    print("[Step 2.2] Performing Purge & Embargo integrity validation.")
    
    slices = context.get('slices', {})
    holding = context.get('holding_period', 5)
    embargo = context.get('embargo_window', 5)
    
    keys = ["Train-A", "Train-B1", "Train-B2", "Validation", "Test"]
    for i in range(len(keys) - 1):
        left = slices.get(keys[i], [])
        right = slices.get(keys[i+1], [])
        if not left or not right:
            continue
        
        # 检查左侧最后一个交易日与右侧第一个交易日的间隔（实际相隔天数）
        # 由于列表中只含交易日，间隔应在日历日上 > embargo
        last_left = left[-1]
        first_right = right[0]
        # 计算两个日期之间的差值（日历日）
        delta = (first_right - last_left).days
        
        # 规范要求：禁运区长度 >= holding 且 >= ACF_lag (日历日)
        # 由于周末存在，delta 日历日应该大于等于 embargo_window（但如果是连续交易日，diff=1，但实际日历日可能跨周末）
        # 严格审计：若 diff < embargo_window，则警报
        if delta < embargo_window:
            raise RuntimeError(
                f"隔离带坍塌！{keys[i]} 与 {keys[i+1]} 交界处间隔仅 {delta} 天，"
                f"低于硬性禁运区下限 {embargo_window} 天。请检查切分算法。"
            )
    
    # 额外检查：确保所有切片非空，否则后续模型无法训练
    for k in keys:
        if len(slices.get(k, [])) < 5:
            raise RuntimeError(f"切片 {k} 包含交易日过少 ({len(slices[k])})，无法支撑模型训练或校准。")

def execute(pipeline_context: dict):
    # 确保 datetime 对象列表存在（step1 应该已生成）
    if 'trading_days_dt' not in pipeline_context:
        # 尝试从字符串日历转换（兼容旧版）
        str_cal = pipeline_context.get('trading_calendar', [])
        if str_cal:
            tz = pipeline_context.get('data_bus')._tz
            pipeline_context['trading_days_dt'] = [datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=tz) for d in str_cal]
        else:
            raise ValueError("无法定位交易日历。请确保 Phase 1 已正确生成 trading_days_dt。")
    
    step_2_1_moving_window_slicing(pipeline_context)
    step_2_2_purge_and_embargo_validation(pipeline_context)
    pipeline_context['slices_isolated'] = True
    return pipeline_context