# step2/acf_analyzer.py
import logging
import numpy as np
from statsmodels.tsa.stattools import acf
from step2.config import MIN_DAYS_FOR_ACF, ACF_NLAGS, ACF_ALPHA, MAX_SAMPLE_ASSETS

logger = logging.getLogger("DataSlicing.ACF")

def compute_dynamic_acf_lag(context: dict) -> int:
    """
    基于资产池中多只高流动性资产的 log_return ACF 表现，动态估算显著自相关滞后阶数。
    策略 B：向后锚定靠近训练集终点的活跃期窗口，严防未来数据泄露。
    """
    trading_days_dt = context.get('trading_days_dt', [])
    assets = context.get('assets', [])
    data_bus = context.get('data_bus')
    
    max_lag = 1  # 基础保底值
    
    if not assets:
        logger.warning("资产池为空，无法进行 ACF 估算，使用默认滞后 1")
        return max_lag

    total_days = len(trading_days_dt)
    if total_days <= MIN_DAYS_FOR_ACF:
        logger.warning(f"总交易日样本不足（仅 {total_days} 天），无法可靠计算 ACF")
        return max_lag

    # --------------------------------------------------------
    # 安全边界内的“近期锚定”
    # --------------------------------------------------------
    end_idx = int(total_days * 0.6) 
    start_idx = max(0, end_idx - 500) 
    
    if (end_idx - start_idx) < MIN_DAYS_FOR_ACF:
        start_idx = 0
        end_idx = int(total_days * 0.6)

    sample_days = trading_days_dt[start_idx:end_idx]
    start_dt = sample_days[0].strftime("%Y-%m-%d")
    end_dt = sample_days[-1].strftime("%Y-%m-%d")

    logger.info(f"[ACF 策略B] 已锁定安全训练窗口 [{start_dt}] 至 [{end_dt}] 内抽取资产计算自相关...")

    sample_assets = assets[:MAX_SAMPLE_ASSETS] if len(assets) > MAX_SAMPLE_ASSETS else assets
    lags = []
    fail_count = 0
    no_sig_count = 0
    empty_df_count = 0
    
    # 诊断计数器
    reported_errors = 0

    for sym in sample_assets:
        try:
            df = data_bus.load_asset_history(sym, start_dt, end_dt)
            if df is None or df.empty:
                empty_df_count += 1
                fail_count += 1
                continue
                
            if len(df) < MIN_DAYS_FOR_ACF:
                fail_count += 1
                if reported_errors < 5:
                    logger.warning(f"[ACF死因剖析] ❌ {sym} 数据行数过短: 实际仅 {len(df)} 行，要求 {MIN_DAYS_FOR_ACF} 行")
                    reported_errors += 1
                continue

            # 💡 智能化多列名容灾对齐
            if 'log_return' not in df.columns:
                # 自动探测可能的潜在收盘价列名
                possible_close_cols = [col for col in ['close', 'total_return_price', 'adj_close', '全收益价格'] if col in df.columns]
                if possible_close_cols:
                    target_col = possible_close_cols[0]
                    df['log_return'] = np.log(df[target_col] / df[target_col].shift(1))
                else:
                    fail_count += 1
                    if reported_errors < 5:
                        logger.warning(f"[ACF死因剖析] ❌ {sym} 缺失核心价格列。现有列为: {list(df.columns)}")
                        reported_errors += 1
                    continue

            # 强力剔除 Inf / -Inf / NaN，规避 statsmodels 熔断
            returns = df['log_return'].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(returns) > MIN_DAYS_FOR_ACF:
                acf_vals, confint = acf(returns, nlags=ACF_NLAGS, alpha=ACF_ALPHA, fft=False)
                sig_lags = []
                for i in range(1, len(acf_vals)):
                    if i < len(confint):
                        lower, upper = confint[i]
                        if acf_vals[i] < lower or acf_vals[i] > upper:
                            sig_lags.append(i)
                if sig_lags:
                    lags.append(max(sig_lags))
                else:
                    no_sig_count += 1
            else:
                fail_count += 1
                if reported_errors < 5:
                    logger.warning(f"[ACF死因剖析] ❌ {sym} 清洗 Inf 后有效样本行数不足 ({len(returns)} 天)")
                    reported_errors += 1
        except Exception as e:
            fail_count += 1
            if reported_errors < 5:
                logger.warning(f"[ACF死因剖析] ❌ {sym} statsmodels 计算崩溃: {e}")
                reported_errors += 1

    # 统计与保底输出
    if lags:
        max_lag = int(np.median(lags))
        logger.info(f"基于 {len(lags)} 只股票的有效 ACF，中位数显著滞后: {max_lag}")
    else:
        logger.warning(f"无法提取有效 ACF（空缓存 {empty_df_count} 只，无效 {fail_count - empty_df_count} 只，无信号 {no_sig_count} 只）。系统启动保底 max_lag=1")

    return max_lag