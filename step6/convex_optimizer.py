import numpy as np
import cvxpy as cp
import logging
from datetime import datetime
from typing import Optional
from .utils import _compute_shares_upper

logger = logging.getLogger("PositionSizing.ConvexOptimizer")

def step_m_3_convex_optimization(context: dict, date: datetime, nav: float, prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
    assets = context['assets']
    n = len(assets)
    R_BL = context.get('R_BL')
    Sigma = context.get('Sigma_robust')
    bus = context['data_bus']
    
    config = context.get('config', {})
    gamma_risk = config.get('gamma_risk_initial', 2.5)
    max_leverage = config.get('max_leverage', 1.0)
    sector_limit = config.get('sector_limit', 0.3)
    eps = config.get('epsilon', 0.001)
    trans_cost = config.get('transaction_cost_coeff', 0.0003)
    short_limit = config.get('short_upper_limit', -0.25)
    borrow_liquidity = context.get('borrowable_today', set())
    
    if R_BL is None or Sigma is None:
        raise RuntimeError("缺少前置 Black-Litterman 融合收益向量或 Robust-Covariance 协方差矩阵。")
        
    w_prev = np.array(prev_weights) if prev_weights is not None else np.zeros(n)
    w = cp.Variable(n)
    utility = w.T @ R_BL - (gamma_risk / 2) * cp.quad_form(w, Sigma) - trans_cost * cp.sum(cp.abs(w - w_prev))
    
    constraints = [cp.sum(cp.abs(w)) <= max_leverage]

    # 动态股本上限和两融资格约束约束
    for i, sym in enumerate(assets):
        upper = _compute_shares_upper(bus, sym, date, nav, config)
        constraints.append(w[i] <= upper)
        if sym in borrow_liquidity:
            constraints.append(w[i] >= short_limit)
        else:
            constraints.append(w[i] >= 0.0)

    # 行业暴露敞口合规约束
    sector_map = context.get('sector_map', {})
    if not sector_map:
        for sym in assets:
            try:
                sector_map[sym] = bus.get_sector(sym)
            except:
                sector_map[sym] = "Unknown"
        context['sector_map'] = sector_map
        
    sectors = set(sector_map.values())
    for sec in sectors:
        idx = [i for i, sym in enumerate(assets) if sector_map.get(sym) == sec]
        if idx:
            constraints.append(cp.sum(w[idx]) <= sector_limit)

    try:
        prob = cp.Problem(cp.Maximize(utility), constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        
        if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
            logger.error(f"[求解器二次熔断] 日期 {date.strftime('%Y-%m-%d')} 求解器状态异常: {prob.status}！强制回滚至上一期持仓权重。")
            return w_prev.copy()
            
        final_w = w.value.flatten()
        final_w[np.abs(final_w) < eps] = 0.0
        
        # 记录关键日志指标
        logger.debug(f"[凸优化收敛成功] 日期: {date.strftime('%Y-%m-%d')} | 求解状态: {prob.status} | 最优目标函数: {prob.value:.5f} | 组合多头敞口: {np.sum(final_w[final_w>0]):.4f} | 组合空头敞口: {np.sum(final_w[final_w<0]):.4f}")
        return final_w
        
    except Exception as e:
        logger.error(f"[凸优化底层矩阵错误] 日期 {date.strftime('%Y-%m-%d')} 触发非线性计算溢出异常: {e}，回滚分配状态以保全资金安全。")
        return w_prev.copy()