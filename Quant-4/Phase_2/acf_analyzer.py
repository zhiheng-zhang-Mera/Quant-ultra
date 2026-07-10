# -*- coding: utf-8 -*-
import logging
import numpy as np
from statsmodels.tsa.stattools import acf
from Phase_2.config import MIN_DAYS_FOR_ACF, ACF_NLAGS, ACF_ALPHA, MAX_SAMPLE_ASSETS

logger = logging.getLogger("DataSlicing.ACF")

def _calculate_market_lag(assets, data_bus, start_dt, end_dt, market_label):
    lags = []
    fail_count = 0
    no_sig_count = 0
    reported_errors = 0
    sample_assets = assets[:MAX_SAMPLE_ASSETS]
    
    for sym in sample_assets:
        try:
            df = data_bus.load_asset_history(sym, start_dt, end_dt)
            if df is None or df.empty:
                fail_count += 1; continue
            
            # 刚性绑定：阶段1.2生存者偏差治理产出的对数实际全收益率
            if 'actual_log_return' in df.columns: target_col = 'actual_log_return'
            elif 'log_return' in df.columns: target_col = 'log_return'
            else:
                possible_close = [c for c in ['close', 'adj_close', 'total_return_price'] if c in df.columns]
                if possible_close:
                    target_col = 'derived_log_return'
                    df[target_col] = np.log(df[possible_close[0]] / df[possible_close[0]].shift(1))
                else: fail_count += 1; continue
            
            returns = df[target_col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns) <= MIN_DAYS_FOR_ACF:
                fail_count += 1; continue
            
            acf_vals, confint = acf(returns, nlags=ACF_NLAGS, alpha=ACF_ALPHA, fft=False)
            sig_lags = []
            for i in range(1, len(acf_vals)):
                if i < len(confint):
                    lower, upper = confint[i]
                    if acf_vals[i] < lower or acf_vals[i] > upper: sig_lags.append(i)
            if sig_lags: lags.append(max(sig_lags))
            else: no_sig_count += 1
        except Exception as e:
            fail_count += 1
            if reported_errors < 2:
                logger.debug(f"[{market_label}] Asset {sym} ACF anomaly: {e}")
                reported_errors += 1
                
    if lags:
        median_lag = int(np.median(lags))
        logger.info("[OP] Resample Single Node ACF Lags | [SOURCE] Edge Domain Asset Returns Stream | [RESULT] Track: %s, Valid Samples: %s, Median Lag: %s | [SIGNIFICANCE] Quantifies historical micro-memory persistence to guide embargo constraints", market_label, len(lags), median_lag)
        logger.info("[操作] 重采样单节点 ACF 滞后阶数 | [来源] 边缘域资产收益率流 | [结果] 市场轨: %s, 有效样本数: %s, 中位数滞后: %s | [意义] 量化历史微观记忆持续性以引导数据禁运约束")
        return median_lag
    return 1

def compute_dynamic_acf_lag(context: dict) -> int:
    trading_days_dt_cn = context.get('trading_days_dt_cn', [])
    assets = context.get('assets', [])
    data_bus = context.get('data_bus')
    
    logger.info("[OP] Deploy Federated Coordinated ACF Auditor | [SOURCE] Shared Cross-Sectional Asset Pool Tokens | [RESULT] Initializing dual-market lag analysis loops | [SIGNIFICANCE] Scans informational linkage structure across remote asset realms")
    logger.info("[操作] 部署联邦协同 ACF 审计器 | [来源] 共享横截面资产池令牌 | [结果] 正在初始化双市场滞后分析循环 | [意义] 扫描跨远程资产领域的非对称信息联动结构")
    
    max_lag = 1
    if not assets or len(trading_days_dt_cn) <= MIN_DAYS_FOR_ACF: return max_lag
        
    end_idx = int(len(trading_days_dt_cn) * 0.6)
    start_idx = max(0, end_idx - 500)
    sample_days_cn = trading_days_dt_cn[start_idx:end_idx]
    start_dt = sample_days_cn[0].strftime("%Y-%m-%d")
    end_dt = sample_days_cn[-1].strftime("%Y-%m-%d")
    
    cn_assets = [a for a in assets if not ('.US' in a or any(c.isalpha() for c in a.split('.')[0]))]
    us_assets = [a for a in assets if ('.US' in a or any(c.isalpha() for c in a.split('.')[0]))]
    
    lag_cn = _calculate_market_lag(cn_assets, data_bus, start_dt, end_dt, "A股节点")
    lag_us = _calculate_market_lag(us_assets, data_bus, start_dt, end_dt, "美股节点")
    
    final_lag = max(lag_cn, lag_us)
    logger.info("[OP] Aggregate Consolidated Market Lag Bounds | [SOURCE] Decoupled Multi-Domain ACF Estimates | [RESULT] CN Lag: %s, US Lag: %s -> Global Envelope: %s | [SIGNIFICANCE] Sets the cross-sectional safety cushions required to eliminate overlap taint", lag_cn, lag_us, final_lag)
    logger.info("[操作] 聚合归并跨市场滞后边界 | [来源] 解耦的多域 ACF 估计值 | [结果] A股滞后: %s, 美股滞后: %s -> 全局生效上限: %s | [意义] 设定消除样本重叠交叉污染所需的横截面安全防震垫")
    return final_lag