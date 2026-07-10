# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 6.3: Conformal Non-Linear Convex Optimizer Core (Pure Long Barrier Only)
"""
import numpy as np
import cvxpy as cp
import logging
from datetime import datetime
from typing import Optional
from Phase_6.utils import _compute_individual_shares_upper

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
    
    if R_BL is None or Sigma is None: raise RuntimeError("Pre-conditions metrics missing.")
    w_prev = np.array(prev_weights) if prev_weights is not None else np.zeros(n)
    
    w = cp.Variable(n)
    # 定义标准目标效用公式：后验预期多头收益 - 组合滚动方差惩罚 - 跨期调仓换手冲击惩罚
    utility = w.T @ R_BL - (gamma_risk / 2) * cp.quad_form(w, Sigma) - trans_cost * cp.norm1(w - w_prev)
    
    constraints = [cp.sum(w) == 1.0]
    for i, sym in enumerate(assets):
        upper = _compute_individual_shares_upper(bus, sym, date, nav, config)
        constraints.append(w[i] <= upper)
        # 🛡️ 多头凸优化绝对防御红线：底端物理界限刚性死锁 >= 0.0，彻底根除空头信号越界生成
        constraints.append(w[i] >= 0.0)

    # ----------------------------------------------------
    # 【修复核心防御线：破除 setdefault 贪婪求值陷阱】
    # ----------------------------------------------------
    if 'sector_map' not in context:
        try:
            context['sector_map'] = {s: bus.get_sector(s) for s in assets}
        except Exception as e:
            logger.warning("[GUARD] Pre-loaded sector_map missing and bus.get_sector failed. Falling back to default '综合'. Error: %s", e)
            logger.warning("[守护] 缺失预载行业映射且底层 bus.get_sector 异常！触发三级容灾自愈，刚性强制降级至统一'综合'行业。错误: %s", e)
            context['sector_map'] = {s: "综合" for s in assets}
            
    sector_map = context['sector_map']

    for sec in set(sector_map.values()):
        idx = [i for i, sym in enumerate(assets) if sector_map.get(sym) == sec]
        if idx: constraints.append(cp.sum(w[idx]) <= sector_limit)

    try:
        prob = cp.Problem(cp.Maximize(utility), constraints)
        prob.solve(solver=cp.ECOS, verbose=False)
        
        if w.value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
            logger.warning("[OP] Solve Quadratic Program | [SOURCE] CVX Subsystem | [RESULT] Solver Anomalous Status: %s | [SIGNIFICANCE] Solver secondary fallback triggered; rolling back to previous weights", prob.status)
            logger.warning("[操作] 求解二次凸优化规划 | [来源] CVX 求解物理子系统 | [结果] 求解器状态异常异常: %s | [意义] 触发求解器二级容灾自愈防火墙，无损回滚至前一期历史持仓分配")
            return w_prev.copy()
            
        final_w = w.value.flatten()
        final_w[final_w < eps] = 0.0
        total_w = np.sum(final_w)
        return final_w / (total_w if total_w > 0 else 1.0)
    except Exception:
        return w_prev.copy()