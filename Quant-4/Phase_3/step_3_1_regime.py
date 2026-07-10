# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 3.1: Adaptive Cross-Sectional Volatility Regime Classifier
"""
import logging
import numpy as np
import pandas as pd
from Phase_3.config import DEFAULT_VOL_WINDOW

logger = logging.getLogger("Phase3.Regime")

def run_online_regime_labels(context: dict):
    assets = context.get('assets', [])
    trading_days_cn = context.get('trading_days_dt_cn', [])
    if not assets or not trading_days_cn: raise ValueError("Missing assets or China calendar base.")
        
    config = context.get('config', {})
    vol_window = config.get('vol_window', DEFAULT_VOL_WINDOW)
    current_date_cn = trading_days_cn[-1]
    calendar_alignment = context.get('calendar_alignment', {})
    asset_data = context.get('asset_ohlcv', {})
    vol_dict = {}
    
    for sym in assets:
        df = asset_data.get(sym)
        if df is None or df.empty: continue
        is_a_share = any(suffix in sym.upper() for suffix in [".SH", ".SZ", ".BJ"]) or sym.isdigit()
        
        # Flow-Pro 1.4: 激活跨市场序号代币硬映射对齐
        if is_a_share: target_date = current_date_cn
        else:
            if calendar_alignment:
                date_str_cn = current_date_cn.strftime("%Y-%m-%d") if hasattr(current_date_cn, 'strftime') else str(current_date_cn)[:10]
                seq = calendar_alignment["date_to_seq_cn"].get(date_str_cn)
                if seq is not None and seq in calendar_alignment["seq_to_date_us"]:
                    target_date = pd.to_datetime(calendar_alignment["seq_to_date_us"][seq])
                else: target_date = current_date_cn
            else: target_date = current_date_cn
                
        if df.index.tz is not None and getattr(target_date, 'tz', None) is None:
            target_date = pd.to_datetime(target_date).tz_localize(df.index.tz)
        elif df.index.tz is None and getattr(target_date, 'tz', None) is not None:
            target_date = pd.to_datetime(target_date).tz_localize(None)
            
        idx = df.index.searchsorted(target_date, side='right') - 1
        if idx < 4 or idx < vol_window - 1: continue
        prices = df.iloc[max(0, idx - vol_window + 1):idx+1]['close'].values
        if len(prices) < 5: continue
            
        rets = np.diff(np.log(prices))
        if len(rets) > 0: vol_dict[sym] = np.std(rets) * np.sqrt(252)
            
    if not vol_dict:
        regime_map = {sym: 1 for sym in assets}
    else:
        vols = np.array(list(vol_dict.values()))
        lower_q, upper_q = np.percentile(vols, 33), np.percentile(vols, 67)
        
        logger.info("[OP] Compute Cross-Sectional Volatility Quantiles | [SOURCE] Rolling Log Return Multi-Track Array | [RESULT] Lower Boundary (33%%): %.4f, Upper Boundary (67%%): %.4f | [SIGNIFICANCE] Sets adaptive macro risk labels for down-stream model gating logic", lower_q, upper_q)
        logger.info("[操作] 计算横截面波动率分位数 | [来源] 滚动对数收益率多轨矩阵 | [结果] 33%%分位低波线: %.4f, 67%%分位高波线: %.4f | [意义] 动态切分市场风险宏观体制，为下游多层级拓扑网络输出条件门控标签")
        
        regime_map = {}
        for sym in assets:
            vol = vol_dict.get(sym)
            if vol is None or np.isnan(vol): regime_map[sym] = 1
            elif vol < lower_q: regime_map[sym] = 0
            elif vol > upper_q: regime_map[sym] = 2
            else: regime_map[sym] = 1
                
    context['online_regime_state'] = regime_map