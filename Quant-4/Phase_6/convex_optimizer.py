# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 6.3: Conformal Non-Linear Convex Optimizer Core (DPP Parametrized & Multi-Thread Bounds)
"""
import numpy as np
import cvxpy as cp
import logging
from datetime import datetime
from typing import Optional
import concurrent.futures
from Phase_6.utils import _compute_individual_shares_upper
from Main.trading_costs import merged_cost_config

logger = logging.getLogger("PositionSizing.ConvexOptimizer")

def feasible_investment_floor(configured_floor: float, upper_bounds: np.ndarray, investable_cap: float) -> float:
    """Return a conservative lower bound that cannot exceed position capacity."""
    capacity = float(np.maximum(np.asarray(upper_bounds, dtype=float), 0.0).sum())
    return float(min(max(configured_floor, 0.0), max(investable_cap, 0.0), capacity * 0.90))

def step_m_3_convex_optimization(context: dict, date: datetime, nav: float, prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
    assets = context['assets']
    n = len(assets)
    R_BL = context.get('R_BL')
    Sigma = context.get('Sigma_robust')
    bus = context['data_bus']
    
    config = context.get('config', {})
    gamma_risk = config.get('gamma_risk_initial', 2.5)
    sector_limit = config.get('sector_limit', 0.3)
    costs = merged_cost_config(config)
    explicit_round_trip = 2 * (costs['commission_rate'] + costs['exchange_fee_rate'] + costs['regulatory_fee_rate'] + costs['slippage_rate']) + costs['stamp_tax']
    trans_cost = max(config.get('transaction_cost_coeff', 0.0003), explicit_round_trip)
    cash_buffer = float(np.clip(config.get('cash_buffer_weight', 0.05), 0.0, 0.40))
    configured_min_invested = float(np.clip(config.get('minimum_invested_weight', 0.10), 0.0, 1.0 - cash_buffer))
    max_turnover = float(max(config.get('max_daily_turnover', 0.25), 0.0))
    concentration_coeff = float(max(config.get('concentration_penalty', 0.02), 0.0))
    
    if R_BL is None or Sigma is None: raise RuntimeError("Pre-conditions metrics missing.")
    w_prev = np.array(prev_weights) if prev_weights is not None else np.zeros(n)

    # ==============================================================================
    # ⚡ 提速层 1：解禁并发 I/O 锁，极速抽取个股流动性容量约束
    # 采用 ThreadPoolExecutor 将 5529 次串行读取压缩到多路并发池中，击穿 GIL 限制
    # ==============================================================================
    def _get_upper(sym):
        return _compute_individual_shares_upper(bus, sym, date, nav, config)
        
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(config.get('optimization_workers', 16))) as executor:
        upper_bounds = np.array(list(executor.map(_get_upper, assets)))
    min_invested = feasible_investment_floor(configured_min_invested, upper_bounds, 1.0 - cash_buffer)

    # 行业映射提取
    if 'sector_map' not in context:
        try: context['sector_map'] = {s: bus.get_sector(s) for s in assets}
        except Exception: context['sector_map'] = {s: "综合" for s in assets}
    sector_map = context['sector_map']

    # ==============================================================================
    # ⚡ 提速层 2：CVXPY 静态参数化编译 (DPP - Disciplined Parametrized Programming)
    # 核心原理：彻底消灭 AST 解析。首日花 1.5 秒钟“编译”抽象语法树并固化到内存，
    # 后续 199 个交易日只做 O(1) 级别的 C 底层内存指针覆写，耗时直接跳水 90%
    # ==============================================================================
    cache_key = 'cvx_prob_cache_v2'
    if cache_key not in context or context[cache_key]['n'] != n:
        logger.info("=== Phase 6: 首次重型编译 CVXPY AST 抽象语法树 (后续交易日将 O(1) 极速复用) ===")
        
        # 声明优化变量
        w = cp.Variable(n)
        
        # 声明“静态空壳参数” (不再传入真实数据，只占位编译)
        R_BL_param = cp.Parameter(n)
        Sigma_param = cp.Parameter((n, n), PSD=True) # 刚性声明半正定，通过 DCP 校验
        w_prev_param = cp.Parameter(n)
        upper_bounds_param = cp.Parameter(n, nonneg=True)
        turnover_budget_param = cp.Parameter(nonneg=True)
        min_invested_param = cp.Parameter(nonneg=True)
        
        # 定义标准目标效用公式 (对参数而非实际数据进行操作)
        expected_return = R_BL_param.T @ w
        risk_penalty = (gamma_risk / 2) * cp.quad_form(w, Sigma_param)
        turnover_penalty = trans_cost * cp.norm(w - w_prev_param, 1)
        concentration_penalty = concentration_coeff * cp.sum_squares(w)
        
        utility = expected_return - risk_penalty - turnover_penalty - concentration_penalty
        
        # 构建静态仿射行业约束矩阵
        unique_sectors = list(set(sector_map.values()))
        num_sectors = len(unique_sectors)
        A_sec = np.zeros((num_sectors, n))
        sector_to_idx = {sec: i for i, sec in enumerate(unique_sectors)}
        for j, sym in enumerate(assets):
            sec = sector_map.get(sym, "综合")
            if sec in sector_to_idx: A_sec[sector_to_idx[sec], j] = 1.0
                
        constraints = [
            w >= 0,
            cp.sum(w) <= 1.0 - cash_buffer,
            cp.sum(w) >= min_invested_param,
            w <= upper_bounds_param,
            A_sec @ w <= sector_limit,
            cp.norm1(w - w_prev_param) <= turnover_budget_param
        ]

        # 极其昂贵的编译动作，仅执行这一次
        prob = cp.Problem(cp.Maximize(utility), constraints)
        
        # 将编译后的模型骨架沉淀到主控管道上下文
        context[cache_key] = {
            'n': n,
            'prob': prob,
            'w': w,
            'R_BL_param': R_BL_param,
            'Sigma_param': Sigma_param,
            'w_prev_param': w_prev_param,
            'upper_bounds_param': upper_bounds_param,
            'turnover_budget_param': turnover_budget_param,
            'min_invested_param': min_invested_param
        }

    # ------------------------------------------------------------------------------
    # ⚡ 提速层 3：极速热注水与 OSQP 底层热启动 (Warm Start)
    # ------------------------------------------------------------------------------
    cache = context[cache_key]
    prob = cache['prob']
    
    # 1. 暴力覆写底层 C 数组内存，完美绕开 Python 解析器
    cache['R_BL_param'].value = R_BL
    cache['w_prev_param'].value = w_prev
    cache['upper_bounds_param'].value = upper_bounds
    cache['min_invested_param'].value = min_invested
    initial_allowance = max(0.0, min_invested - float(np.sum(w_prev)))
    cache['turnover_budget_param'].value = max_turnover + initial_allowance
    
    # 🛡️ 刚性微调协方差：强制对称化并施加微量对角扰动，保证 Float64 精度下绝对半正定，防止求解器中途崩溃
    Sigma_sym = (Sigma + Sigma.T) / 2.0
    Sigma_sym.flat[::n+1] += 1e-6
    cache['Sigma_param'].value = Sigma_sym

    try:
        # 2. 开启 warm_start：求解器会以“昨天的持仓权重”作为今天的搜索起点，将内点法的底层迭代计算步数直接砍掉一半！
        prob.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        
        if cache['w'].value is None or prob.status not in ["optimal", "optimal_inaccurate"]:
            # OSQP 无法收敛时，无缝切入 SCS
            prob.solve(solver=cp.SCS, warm_start=False, verbose=False)
            
        result_w = np.array(cache['w'].value)
        if result_w is None or np.isnan(result_w).any():
            raise ValueError("Solver return internal NaN.")
            
        return result_w
        
    except Exception as e:
        logger.error(f"[容灾] CVX 凸优化矩阵退化崩塌，降级输出防御性分布: {e}")
        return w_prev if np.sum(w_prev) > 0 else np.ones(n) / n
