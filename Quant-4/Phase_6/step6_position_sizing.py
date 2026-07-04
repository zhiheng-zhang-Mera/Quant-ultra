# -*- coding: utf-8 -*-
"""
Phase 6: Multi-Market Conformal Position Sizing and Optimization Engine Layer
"""
import logging
import pandas as pd
import numpy as np
from Phase_6.config import DEFAULT_CONFIG
from Phase_6.directional_mask import step_m_1_directional_mask
from Phase_6.bl_fusion import step_m_2_black_litterman_fusion
from Phase_6.convex_optimizer import step_m_3_convex_optimization

logger = logging.getLogger("PositionSizing")

def execute(pipeline_context: dict) -> dict:
    logger.info("=" * 60)
    logger.info("[OP] Enter Phase_6 Sizing Orchestrator | [SOURCE] Global Main Pipeline Stream | [RESULT] Launching convex optimization engine | [SIGNIFICANCE] Maps predictive views to compliant asset weights accurately")
    logger.info("[操作] 锁锁 Phase_6 分配主编排器入口 | [来源] 全局系统主计算干线流 | [结果] 正在激活单边多头优化引擎 | [意义] 将离线预标定的远期多维预测视点精准映射为符合合规物理契约的分配资产权重")
    logger.info("=" * 60)

    config = pipeline_context.setdefault('config', DEFAULT_CONFIG.copy())
    for k, v in DEFAULT_CONFIG.items(): config.setdefault(k, v)
        
    local_context = pipeline_context.copy()
    slices = local_context.get('slices', {})
    test_dates_raw = slices.get('Test', []) if isinstance(slices, dict) else []
    
    if not test_dates_raw: raise ValueError("Test chronology partition vacant.")
    test_dates = sorted([pd.Timestamp(d) for d in test_dates_raw])
    
    assets = local_context['assets']
    n_assets = len(assets)
    data_bus = local_context['data_bus']
    
    weight_records = []
    w_prev = np.zeros(n_assets)
    current_nav = config.get('individual_account_equity', 10000000.0)

    # 顺次沿着测试样本外时间轴推进每日横截面凸优化解算
    for t_date in test_dates:
        local_context['directional_symbol_masks'] = step_m_1_directional_mask(local_context, t_date)
        
        R_BL, Sigma_robust, Q_view, Omega_diag = step_m_2_black_litterman_fusion(local_context, t_date, w_prev)
        local_context.update({'R_BL': R_BL, 'Sigma_robust': Sigma_robust, 'Q_view': Q_view, 'Omega_diag': Omega_diag})
        
        w_new = step_m_3_convex_optimization(local_context, t_date, current_nav, w_prev)
        weight_records.append(w_new)
        w_prev = w_new.copy()

    weights_df = pd.DataFrame(weight_records, index=[d.strftime('%Y-%m-%d') for d in test_dates], columns=assets)
    intervals_df = pd.DataFrame(0.02, index=[d.strftime('%Y-%m-%d') for d in test_dates], columns=assets)
    
    adv_records = []
    for date in test_dates:
        daily_adv = []
        for asset in assets:
            try:
                hist = data_bus.load_asset_history(asset, (date - pd.Timedelta(days=40)).strftime("%Y-%m-%d"), date.strftime("%Y-%m-%d"))
                adv_val = float((hist['volume'].tail(20) * hist['close'].tail(20)).mean()) if (hist is not None and len(hist) >= 20) else 20000000.0
            except Exception: adv_val = 20000000.0
            daily_adv.append(adv_val)
        adv_records.append(daily_adv)
        
    adv20_df = pd.DataFrame(adv_records, index=[d.strftime('%Y-%m-%d') for d in test_dates], columns=assets)

    logger.info("[OP] Finish Phase_6 Sizing Loop | [SOURCE] Optimized Time-series Allocation Stack | [RESULT] Emitted Weight Matrix Shape: %s | [SIGNIFICANCE] Fully complies with multi-market cross-phase input contracts", weights_df.shape)
    logger.info("[操作] 终结 Phase_6 核心解算分配循环 | [来源] 优化完成的时序头寸分配大表栈 | [结果] 输出最终调配权重面板维度: %s | [意义] 满足流水线数据全闭环契约，无缝向策略有限状态机层交底交付")
    
    return {
        'daily_weights': weights_df,
        'daily_intervals': intervals_df,
        'daily_adv20': adv20_df,
        'position_sizing_ready': True
    }