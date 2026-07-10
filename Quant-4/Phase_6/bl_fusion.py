# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 6.2: Prior Equilibrium & Epistemic Uncertainty Black-Litterman Blender
"""
import logging
import numpy as np
from datetime import datetime
from typing import Optional, Tuple
from Phase_6.utils import _get_features_for_date, _compute_robust_covariance, _compute_market_weights

logger = logging.getLogger("PositionSizing.BLFusion")

def step_m_2_black_litterman_fusion(context: dict, date: datetime, prev_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bus = context['data_bus']
    assets = context['assets']
    n = len(assets)
    masks = context.get('directional_symbol_masks', {})
    quant_models = context.get('quantile_models')
    
    if quant_models is None: raise RuntimeError("Quantile models absent.")
        
    config = context.get('config', {})
    tau = config.get('tau_BL', 0.02)
    omega_min = config.get('omega_min', 1e-8)
    omega_max = config.get('omega_max', 0.01)
    halflife = config.get('width_halflife', 21)
    
    context.setdefault('smoothed_width', {})
    
    Q_view = np.zeros(n)
    Omega_diag = np.zeros(n)
    
    for i, sym in enumerate(assets):
        feat = _get_features_for_date(sym, date, context)
        if feat is None: Omega_diag[i] = omega_max; continue
            
        try:
            # 抽取高维特征魔方投喂出的 95% 共形推断预期波动宽度 (High-Low Bounds)
            q_low = quant_models[0.025].predict(feat.reshape(1, -1))[0]
            q_mid = quant_models[0.5].predict(feat.reshape(1, -1))[0]
            q_high = quant_models[0.975].predict(feat.reshape(1, -1))[0]
            
            width = max(1e-4, float(q_high - q_low))
            
            # 使用指数移动平滑消解微观高频预测波动区间震荡，稳定不确定性估计
            alpha = 1 - 0.5 ** (1 / halflife)
            prev_smooth = context['smoothed_width'].get(sym, width)
            smoothed = alpha * width + (1 - alpha) * prev_smooth
            context['smoothed_width'][sym] = smoothed
            
            Omega_diag[i] = np.clip((smoothed ** 2) * tau, omega_min, omega_max)
            # 严格遵循多头合规指导：彻底物理剥离融券成本扣减，利用现货掩码对中性观点直接调零清净
            Q_view[i] = q_mid * masks.get(sym, 0)
        except Exception:
            Omega_diag[i] = omega_max
            
    Sigma_robust = _compute_robust_covariance(bus, assets, date, lookback=config.get('lookback_cov', 252))
    w_mkt = _compute_market_weights(bus, assets, date)
    
    try: lambda_mkt = bus.compute_market_risk_aversion(date.strftime('%Y-%m-%d'))
    except Exception: lambda_mkt = 2.5
        
    # Black-Litterman 核心：逆推市场共识超额收益先验 Pi 向量
    Pi = lambda_mkt * (Sigma_robust @ w_mkt)
    P_mat = np.eye(n)
    Omega = np.diag(Omega_diag)
    
    inv_Sigma_tau = np.linalg.inv(tau * Sigma_robust)
    inv_Omega = np.linalg.inv(Omega)
    
    # 解析联合后验预期超额收益率核心代数方程
    inv_A = np.linalg.inv(inv_Sigma_tau + P_mat.T @ inv_Omega @ P_mat)
    R_BL = inv_A @ (inv_Sigma_tau @ Pi + P_mat.T @ inv_Omega @ Q_view)
    
    logger.info("[OP] Fuse Posterior Expectations | [SOURCE] Prior Pi Array & Epistemic Uncertainty Omega | [RESULT] R_BL Vector Length: %s | [SIGNIFICANCE] Stabilizes forward-looking optimization inputs mathematically", len(R_BL))
    logger.info("[操作] 融合求解后验期望收益率 | [来源] 均衡先验乘子与认知不确定性对角阵 Omega | [结果] 融合后的 R_BL 向量长度: %s | [意义] 完成 Black-Litterman 贝叶斯信息大综合，输出具备数学稳定性的远期预期向量")
    return R_BL, Sigma_robust, Q_view, Omega_diag