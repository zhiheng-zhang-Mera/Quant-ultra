import numpy as np
import logging
from datetime import datetime
from typing import Optional, Tuple
from .utils import _get_features_for_date, _compute_robust_covariance, _compute_market_weights

logger = logging.getLogger("PositionSizing.BLFusion")

def step_m_2_black_litterman_fusion(context: dict, date: datetime, prev_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    bus = context['data_bus']
    assets = context['assets']
    n = len(assets)
    masks = context.get('directional_symbol_masks', {})
    quant_models = context.get('quantile_models')
    if quant_models is None:
        logger.critical("[Orchestrator 断层] 上下文中找不到 quantile_models 分位数回归预测模型簇！")
        raise RuntimeError("quantile_models 未找到。")
        
    config = context.get('config', {})
    tau = config.get('tau_BL', 0.02)
    omega_min = config.get('omega_min', 1e-8)
    omega_max = config.get('omega_max', 0.01)
    halflife = config.get('width_halflife', 21)
    lookback_cov = config.get('lookback_cov', 252)
    
    if 'width_history' not in context:
        context['width_history'] = {sym: [] for sym in assets}
    if 'smoothed_width' not in context:
        context['smoothed_width'] = {}

    q_low, q_mid, q_high = np.zeros(n), np.zeros(n), np.zeros(n)
    Q_view = np.zeros(n)
    Omega_diag = np.zeros(n)

    for i, sym in enumerate(assets):
        feat = _get_features_for_date(sym, date, context)
        if feat is None:
            q_low[i] = q_mid[i] = q_high[i] = 0.0
            Q_view[i] = 0.0
            Omega_diag[i] = omega_max
            continue
            
        try:
            X = feat.reshape(1, -1)
            pred_low = quant_models[0.025].predict(X)[0]
            pred_mid = quant_models[0.5].predict(X)[0]
            pred_high = quant_models[0.975].predict(X)[0]
            
            pred_low = min(pred_low, pred_mid)
            pred_high = max(pred_high, pred_mid)
            
            err_thresh = context.get('q_error_threshold_dict', {}).get(sym, 0.0)
            q_low[i] = pred_low - err_thresh
            q_mid[i] = pred_mid
            q_high[i] = pred_high + err_thresh

            width = q_high[i] - q_low[i]
            context['width_history'][sym].append(width)
            if len(context['width_history'][sym]) > lookback_cov * 2:
                context['width_history'][sym] = context['width_history'][sym][-lookback_cov:]
                
            alpha = 1 - 0.5 ** (1 / halflife)
            prev_smooth = context['smoothed_width'].get(sym, width)
            smoothed = alpha * width + (1 - alpha) * prev_smooth
            context['smoothed_width'][sym] = smoothed
            
            omega_ii = np.clip((smoothed ** 2) * tau, omega_min, omega_max)
            Omega_diag[i] = omega_ii

            short_rate = bus.get_short_rate(sym) if masks.get(sym, 0) == -1 else 0.0
            Q_view[i] = q_mid[i] - short_rate
        except Exception as e:
            logger.warning(f"[{sym}] 分位数不确定性估值发生溢出: {e}，强制重置置信矩阵度量。")
            Omega_diag[i] = omega_max

    Sigma_robust = _compute_robust_covariance(bus, assets, date, lookback=lookback_cov)
    w_mkt = _compute_market_weights(bus, assets, date)
    
    try:
        lambda_mkt = bus.compute_market_risk_aversion(date.strftime('%Y-%m-%d'))
    except Exception as e:
        logger.debug(f"[风险厌恶估算异常] 无法抽取市场风险共识风险厌恶度: {e}，启用常规先验均值 2.5。")
        lambda_mkt = 2.5
        
    Pi = lambda_mkt * (Sigma_robust @ w_mkt)
    Omega = np.diag(Omega_diag)
    
    try:
        inv_prior = np.linalg.inv(tau * Sigma_robust)
        inv_omega = np.linalg.inv(Omega)
        P = np.eye(n)
        R_BL = np.linalg.inv(inv_prior + P.T @ inv_omega @ P) @ (inv_prior @ Pi + P.T @ inv_omega @ Q_view)
    except np.linalg.LinAlgError as e:
        logger.error(f"[BL融合矩阵奇异异常] 日期 {date.strftime('%Y-%m-%d')} 求逆失败: {e}，触发物理安全熔断，强制降级使用平衡均衡市场收益 Pi。")
        R_BL = Pi.copy()
        
    return R_BL, Sigma_robust, q_low, q_high