import numpy as np
import cvxpy as cp
import logging
from datetime import datetime
from typing import Optional
from .utils import _compute_individual_shares_upper

logger = logging.getLogger("PositionSizing.ConvexOptimizer")

def step_m_3_convex_optimization(context: dict, date: datetime, nav: float, prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
    assets = context['assets']
    n = len(assets)
    R_BL = context.get('R_BL')
    Sigma = context.get('Sigma_robust')
    bus = context['data_bus']
    
    config = context.get('config', {})
    gamma_risk = config.get('gamma_risk_initial', 2.5)
    sector_limit = config.get('sector_limit', 0.3)
    eps = config.get('epsilon', 0.001)
    trans_cost = config.get('transaction_cost_coeff', 0.0003)
    
    if R_BL is None or Sigma is None:
        raise RuntimeError("缺少前置 Black-Litterman 融合收益向量或 Robust-Covariance 协方差矩阵。")
        
    w_prev = np.array(prev_weights) if prev_weights is not None else np.zeros(n)
    w = cp.Variable(n)
    
    # 单边多头凸优化方程，注入有符号调仓换手冲击惩罚项
    utility = w.T @ R_BL - (gamma_risk / 2) * cp.quad_form(w, Sigma) - trans_cost * cp.sum(cp.abs(w - w_prev))
    
    # 刚性硬防线：个体账户可用现金100%绝对硬顶，杜绝任何两融透支融资
    constraints = [cp.sum(w) <= 1.0]

    for i, sym in enumerate(assets):
        # 挂载个人微观流动性硬约束动态算子
        upper = _compute_individual_shares_upper(bus, sym, date, nav, config)
        constraints.append(w[i] <= upper)
        constraints.append(w[i] >= 0.0)  # 从底端刚性禁止负持仓（做空信号）

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
            logger.error(f"[求解器二次熔断] 日期 {date.strftime('%Y-%m-%d')} 求解器状态异常: {prob.status}！强制回滚至上一期持仓。")
            return w_prev.copy()
            
        final_w = w.value.flatten()
        final_w[np.abs(final_w) < eps] = 0.0
        
        logger.debug(f"[凸优化收敛成功] 日期: {date.strftime('%Y-%m-%d')} | 求解状态: {prob.status} | 组合净多头敞口: {np.sum(final_w):.4f}")
        return final_w
        
    except Exception as e:
        logger.error(f"[凸优化底层矩阵错误] 日期 {date.strftime('%Y-%m-%d')} 触发非线性计算溢出异常: {e}，强制回滚安全状态。")
        return w_prev.copy()
}