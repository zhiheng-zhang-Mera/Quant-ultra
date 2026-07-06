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

def _robust_matrix_inverse(matrix: np.ndarray, name: str = "matrix") -> np.ndarray:
    """
    高确定性矩阵健壮求逆器：内嵌 Tikhonov 对角线收缩加载与奇异值伪逆双重刚性防御垫
    """
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError as e:
        if "singular matrix" in str(e).lower():
            # 引入警告计数器，防止极端数据退化时控制台被崩溃式日志淹没
            if not hasattr(_robust_matrix_inverse, "_warn_count"):
                _robust_matrix_inverse._warn_count = 0
            _robust_matrix_inverse._warn_count += 1
            
            # 仅在前 5 次或每隔 100 次矩阵退化时才触发控制台打印
            if _robust_matrix_inverse._warn_count <= 5 or _robust_matrix_inverse._warn_count % 100 == 0:
                logger.warning("[GUARD] Singular matrix detected during native inversion of [%s]! Applying Tikhonov diagonal loading. (Total warnings: %d)", name, _robust_matrix_inverse._warn_count)
                logger.warning("[守护] 在原生对 [%s] 执行矩阵求逆时检测到奇异矩阵！正在施加蒂霍诺夫对角线加载共形自愈调谐。(当前累计警报: %d 次)", name, _robust_matrix_inverse._warn_count)
            
            # 提取矩阵对角线绝对值的均值作为扰动基准，若全零则用 1e-6 强制筑底
            diag_vals = np.abs(np.diag(matrix))
            diag_mean = np.mean(diag_vals) if len(diag_vals) > 0 else 0.0
            noise_base = 1e-6 if diag_mean == 0 else diag_mean * 1e-4
            
            healed_matrix = matrix.copy()
            # 渐进式递增扰动级数，直至资产协方差网络特征值被完全激活至可逆空间
            for multiplier in [1.0, 10.0, 100.0, 1000.0]:
                try:
                    healed_matrix += np.eye(matrix.shape[0]) * (noise_base * multiplier)
                    return np.linalg.inv(healed_matrix)
                except np.linalg.LinAlgError:
                    continue
        
        # 终极长尾兜底：若对角加载后依然无法求逆（例如极端全零退化），强行拉起 Moore-Penrose 伪逆，死锁量化管线绝不断流
        logger.error("[GUARD] Tikhonov loading completely failed for [%s]. Forcing Moore-Penrose pseudo-inverse fallback.", name)
        logger.error("[守护] 蒂霍诺夫正则化对角加载完全失败 [%s]。强行拉起摩尔-彭罗斯（Moore-Penrose）伪逆执行终极断电保护。", name)
        return np.linalg.pinv(matrix)

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
        
    # Black-Litterman 核心：逆推 market 一致预期超额收益先验 Pi 向量
    Pi = lambda_mkt * (Sigma_robust @ w_mkt)
    P_mat = np.eye(n)
    Omega = np.diag(Omega_diag)
    
    # 【内生防御替换】将所有的原生 np.linalg.inv 强力替换为自愈求逆算子，彻底根治奇异矩阵退化崩溃
    inv_Sigma_tau = _robust_matrix_inverse(tau * Sigma_robust, "tau * Sigma_robust")
    inv_Omega = _robust_matrix_inverse(Omega, "Omega")
    
    # 解析联合后验预期超额收益率核心代数方程
    inv_A = _robust_matrix_inverse(inv_Sigma_tau + P_mat.T @ inv_Omega @ P_mat, "inv_Sigma_tau + P_mat.T @ inv_Omega @ P_mat")
    R_BL = inv_A @ (inv_Sigma_tau @ Pi + P_mat.T @ inv_Omega @ Q_view)
    
    # 终端降噪：引入静态计数器，防止每天调用打印一次阻塞系统
    if not hasattr(step_m_2_black_litterman_fusion, "_call_count"):
        step_m_2_black_litterman_fusion._call_count = 0
    step_m_2_black_litterman_fusion._call_count += 1
    
    if step_m_2_black_litterman_fusion._call_count == 1 or step_m_2_black_litterman_fusion._call_count % 100 == 0:
        logger.info("[OP] Fuse Posterior Expectations | [SOURCE] Prior Pi Array & Epistemic Uncertainty Omega | [RESULT] R_BL Vector Length: %s | [SIGNIFICANCE] Stabilizes forward-looking optimization inputs mathematically (Total days: %d)", len(R_BL), step_m_2_black_litterman_fusion._call_count)
        logger.info("[操作] 融合求解后验期望收益率 | [来源] 均衡先验乘子与认知不确定性对角阵 Omega | [结果] 融合后的 R_BL 向量长度: %s | [意义] 完成 Black-Litterman 贝叶斯信息大综合，输出具备数学稳定性的远期预期向量 (当前累计执行: %d 天)", len(R_BL), step_m_2_black_litterman_fusion._call_count)
        
    return R_BL, Sigma_robust, Q_view, Omega_diag