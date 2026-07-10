# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Step 6.2: Prior Equilibrium & Epistemic Uncertainty Black-Litterman Blender (CPU/GPU Multi-Thread Accelerated)
"""
import os
# ==============================================================================
# ⚡ 硬件级超线程解禁：强制底层 C/Fortran 线性代数库吃满所有物理核心
# ==============================================================================
num_cores = str(os.cpu_count() or 4)
os.environ["OMP_NUM_THREADS"] = num_cores
os.environ["OPENBLAS_NUM_THREADS"] = num_cores
os.environ["MKL_NUM_THREADS"] = num_cores
os.environ["VECLIB_MAXIMUM_THREADS"] = num_cores
os.environ["NUMEXPR_NUM_THREADS"] = num_cores

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Optional, Tuple
import concurrent.futures

# ==============================================================================
# ⚡ 异构计算探针：尝试无缝挂载 GPU 级联加速引擎 (CuPy)
# ==============================================================================
try:
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

from Phase_6.utils import _get_features_for_date, _compute_robust_covariance, _compute_market_weights

logger = logging.getLogger("PositionSizing.BLFusion")

def step_m_2_black_litterman_fusion(context: dict, date: datetime, prev_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not hasattr(step_m_2_black_litterman_fusion, "_call_count"):
        step_m_2_black_litterman_fusion._call_count = 0
    step_m_2_black_litterman_fusion._call_count += 1
    
    # 减少打印频次防止 I/O 阻塞
    if step_m_2_black_litterman_fusion._call_count % 10 == 0:
        print(f"=== Phase 6: BL 贝叶斯融合器 | 异构计算状态: {'GPU CUDA 激进模式' if GPU_AVAILABLE else 'CPU 多核狂暴模式'} | 计算日: {date.strftime('%Y-%m-%d')} ===")
        
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
    Omega_diag = np.ones(n) * omega_max # 刚性灾备筑底
    
    valid_indices = []
    valid_feats = []
    
    # 高速内存指针提取，避免 DataFrame 结构带来的开销
    for i, sym in enumerate(assets):
        feat = _get_features_for_date(sym, date, context)
        if feat is not None:
            valid_indices.append((i, sym))
            valid_feats.append(feat)
            
    if valid_feats:
        X_batch = np.vstack(valid_feats)
        try:
            # ==============================================================================
            # ⚡ 算力并发层：启用线程池 (ThreadPoolExecutor) 执行异步多路模型推理
            # 突破 Python GIL 封锁，利用 C++ 底层释放的多核性能同时跑 3 个 XGBoost/LightGBM
            # ==============================================================================
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                future_low = executor.submit(quant_models[0.025].predict, X_batch)
                future_mid = executor.submit(quant_models[0.5].predict, X_batch)
                future_high = executor.submit(quant_models[0.975].predict, X_batch)
                
                # 线程汇聚屏障 (Join Barrier)
                q_low_batch = future_low.result()
                q_mid_batch = future_mid.result()
                q_high_batch = future_high.result()
            
            alpha = 1 - 0.5 ** (1 / halflife)
            for idx, (i, sym) in enumerate(valid_indices):
                q_low = q_low_batch[idx]
                q_mid = q_mid_batch[idx]
                q_high = q_high_batch[idx]
                
                width = max(1e-4, float(q_high - q_low))
                
                prev_smooth = context['smoothed_width'].get(sym, width)
                smoothed = alpha * width + (1 - alpha) * prev_smooth
                context['smoothed_width'][sym] = smoothed
                
                Omega_diag[i] = np.clip((smoothed ** 2) * tau, omega_min, omega_max)
                Q_view[i] = q_mid * masks.get(sym, 0)
        except Exception as e:
            logger.warning(f"[优化层] 并发预测矩阵崩溃，降级为默认兜底: {e}")
            for i, sym in valid_indices:
                Omega_diag[i] = omega_max
                Q_view[i] = 0.0
                
    Sigma_robust = _compute_robust_covariance(bus, assets, date, lookback=config.get('lookback_cov', 252))
    w_mkt = _compute_market_weights(bus, assets, date)
    
    try: lambda_mkt = bus.compute_market_risk_aversion(date.strftime('%Y-%m-%d'))
    except Exception: lambda_mkt = 2.5
        
    # ==============================================================================
    # ⚡ GPU 代数接管层：基于 Alternative 恒等式的 CUDA 并行矩阵求解器
    # O(N^2) GPU 填充替代 CPU 的慢速循环，释放数万个流处理单元的矩阵运算霸权
    # ==============================================================================
    try:
        if GPU_AVAILABLE:
            # 数据从 Host (内存) 穿透至 Device (显存)
            cp_Sigma = cp.array(Sigma_robust)
            cp_w_mkt = cp.array(w_mkt)
            cp_Q_view = cp.array(Q_view)
            cp_Omega_diag = cp.array(Omega_diag)

            cp_Pi = lambda_mkt * (cp_Sigma @ cp_w_mkt)
            cp_M = tau * cp_Sigma
            
            # GPU 极速主对角线 O(N) 寻址与加载
            cp_idx = cp.arange(n)
            cp_M[cp_idx, cp_idx] += cp_Omega_diag
            
            # 激活 CuPy 的 CUDA 线性方程组求解器
            cp_v = cp.linalg.solve(cp_M, cp_Q_view - cp_Pi)
            cp_R_BL = cp_Pi + (tau * cp_Sigma) @ cp_v
            
            # 运算完毕，Device -> Host 无损回传
            R_BL = cp.asnumpy(cp_R_BL)
            Pi = cp.asnumpy(cp_Pi)
            
        else:
            # [原生 CPU 满载备选流] (当 GPU 未挂载时，由上方 os.environ 接管开启 CPU 狂暴满载)
            Pi = lambda_mkt * (Sigma_robust @ w_mkt)
            M = tau * Sigma_robust.copy()
            np.fill_diagonal(M, M.diagonal() + Omega_diag)
            
            v = np.linalg.solve(M, Q_view - Pi)
            R_BL = Pi + (tau * Sigma_robust) @ v

    except Exception as matrix_err:
        # 三级防线：自愈降级
        logger.warning(f"[守护] 异构代数引擎抛出奇异退化，尝试蒂霍诺夫微扰自愈... ({matrix_err})")
        try:
            # 即使 GPU 失败，也切回 CPU 进行高容错加载修复
            Pi = lambda_mkt * (Sigma_robust @ w_mkt)
            M_healed = tau * Sigma_robust.copy()
            np.fill_diagonal(M_healed, M_healed.diagonal() + Omega_diag + 1e-4)
            v = np.linalg.solve(M_healed, Q_view - Pi)
            R_BL = Pi + (tau * Sigma_robust) @ v
        except Exception:
            logger.error("[守护] 终极熔断：代数流彻底锁死，强制无损回滚至市场均衡收益先验")
            Pi = lambda_mkt * (Sigma_robust @ w_mkt)
            R_BL = Pi.copy()
            
    return R_BL, Sigma_robust, Q_view, Omega_diag