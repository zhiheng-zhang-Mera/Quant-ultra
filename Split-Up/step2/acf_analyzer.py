# step2/acf_analyzer.py
import logging
import numpy as np
from statsmodels.tsa.stattools import acf
from step2.config import MIN_DAYS_FOR_ACF, ACF_NLAGS, ACF_ALPHA, MAX_SAMPLE_ASSETS

logger = logging.getLogger("DataSlicing.ACF")

def _calculate_market_lag(assets, data_bus, start_dt, end_dt, market_label):
    """边缘节点内部独立自相关阶数计算核心，严格绑定实际全收益率层"""
    lags = []
    fail_count = 0
    no_sig_count = 0
    reported_errors = 0
    
    sample_assets = assets[:MAX_SAMPLE_ASSETS]
    
    for sym in sample_assets:
        try:
            df = data_bus.load_asset_history(sym, start_dt, end_dt)
            if df is None or df.empty:
                fail_count += 1
                continue
            
            # 刚性绑定：阶段1.2生存者偏差治理产出的对数实际全收益率
            if 'actual_log_return' in df.columns:
                target_col = 'actual_log_return'
            elif 'log_return' in df.columns:
                target_col = 'log_return'
            else:
                possible_close = [c for c in ['close', 'adj_close', 'total_return_price'] if c in df.columns]
                if possible_close:
                    target_col = 'derived_log_return'
                    df[target_col] = np.log(df[possible_close[0]] / df[possible_close[0]].shift(1))
                else:
                    fail_count += 1
                    continue
            
            returns = df[target_col].replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns) <= MIN_DAYS_FOR_ACF:
                fail_count += 1
                continue
            
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
        except Exception as e:
            fail_count += 1
            if reported_errors < 2:
                logger.debug(f"[{market_label}] 资产 {sym} ACF计算异常: {e}")
                reported_errors += 1
                
    if lags:
        median_lag = int(np.median(lags))
        logger.info(f"[{market_label}] 节点抽样审计完毕. 有效样本数: {len(lags)}, 中位数滞后: {median_lag}")
        return median_lag
    return 1

def compute_dynamic_acf_lag(context: dict) -> int:
    """
    【分布式多节点联合自相关审计器】
    分别解算A股与美股私有资产池的自相关阶数，采用 max(lag_cn, lag_us) 刚性决定全局隔离禁运带。
    """
    trading_days_dt_cn = context.get('trading_days_dt_cn', [])
    assets = context.get('assets', [])
    data_bus = context.get('data_bus')
    
    max_lag = 1
    if not assets:
        logger.warning("共享资产池为空，触发兜底禁运阶数: 1")
        return max_lag
        
    total_days_cn = len(trading_days_dt_cn)
    if total_days_cn <= MIN_DAYS_FOR_ACF:
        logger.warning("交易日样本严重不足，无法启动ACF自相关审计，启动极端防守保底阶数: 1")
        return max_lag
        
    # 锚定近期安全训练窗口，绝对防止未来信息穿透
    end_idx = int(total_days_cn * 0.6)
    start_idx = max(0, end_idx - 500)
    
    sample_days_cn = trading_days_dt_cn[start_idx:end_idx]
    start_dt = sample_days_cn[0].strftime("%Y-%m-%d")
    end_dt = sample_days_cn[-1].strftime("%Y-%m-%d")
    
    logger.info(f"[ACF 分布式协同审计] 锁定安全观测历史视区: [{start_dt}] 至 [{end_dt}]")
    
    # 智能解构分流边缘资产 Pool (A股代码 vs 美股字母后缀代码)
    cn_assets = [a for a in assets if not ('.US' in a or any(c.isalpha() for c in a.split('.')[0]))]
    us_assets = [a for a in assets if ('.US' in a or any(c.isalpha() for c in a.split('.')[0]))]
    
    lag_cn = _calculate_market_lag(cn_assets, data_bus, start_dt, end_dt, "A股节点")
    lag_us = _calculate_market_lag(us_assets, data_bus, start_dt, end_dt, "美股节点")
    
    # 刚性交叉融合红线：取两市场最大自相关滞后，确保两端节点均具备防渗漏安全垫
    final_lag = max(lag_cn, lag_us)
    logger.info(f"[ACF 审计终审] A股滞后: {lag_cn}, 美股滞后: {lag_us}. 全局生效最大自相关滞后令牌: {final_lag}")
    return final_lag