# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase 6 增强模块：层次风险平价 (HRP) 与校准置信度融合
提供可替换凸优化的 HRP 分配器，并集成方向掩码、流动性约束、换手率惩罚。
"""

import numpy as np
import pandas as pd
import logging
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from datetime import datetime
from typing import Optional, Tuple

# ==================== 1. HRP 核心算法 ====================

def _get_cluster_var(cov: np.ndarray, cluster_items: list) -> float:
    """计算簇内资产组合的方差（等权）"""
    sub_cov = cov[np.ix_(cluster_items, cluster_items)]
    w = np.ones(len(cluster_items)) / len(cluster_items)
    return w @ sub_cov @ w

def _get_quasi_diag(linkage_matrix: np.ndarray) -> list:
    """
    根据层次聚类链接矩阵返回准对角化后的资产索引顺序。
    """
    linkage_matrix = linkage_matrix.astype(int)
    # 从最顶层开始递归展开
    n = linkage_matrix.shape[0] + 1
    def _recursive(idx):
        if idx < n:
            return [idx]
        else:
            left = int(linkage_matrix[idx - n, 0])
            right = int(linkage_matrix[idx - n, 1])
            return _recursive(left) + _recursive(right)
    return _recursive(2 * n - 2)

def _hrp_weights_recursive(cov: np.ndarray, items: list) -> np.ndarray:
    """
    递归二分分配权重
    """
    if len(items) == 1:
        return np.array([1.0])
    
    # 分割成两个子簇（按准对角顺序拆分）
    split = len(items) // 2
    left = items[:split]
    right = items[split:]
    
    # 计算子簇方差
    var_left = _get_cluster_var(cov, left)
    var_right = _get_cluster_var(cov, right)
    
    # 逆方差分配
    alpha = var_left / (var_left + var_right)   # 左簇权重系数（注意：逆方差比例）
    # 实际上，分配给左簇的权重应为 var_right / (var_left + var_right)，
    # 但此处我们按原始 HRP 公式：w_left = 1 - var_left/(var_left+var_right) = var_right/(var_left+var_right)
    # 更常见的是，递归分配：整体权重 * 子簇权重系数
    # 返回左子簇和右子簇的权重向量，并乘以各自的系数
    w_left = _hrp_weights_recursive(cov, left) * (var_right / (var_left + var_right))
    w_right = _hrp_weights_recursive(cov, right) * (var_left / (var_left + var_right))
    
    return np.concatenate([w_left, w_right])

def hrp_weights(cov: np.ndarray, method: str = 'ward') -> np.ndarray:
    """
    计算层次风险平价权重
    :param cov: n×n 协方差矩阵
    :param method: 层次聚类方法（'ward', 'single', 'complete', 'average'）
    :return: n维权重向量
    """
    n = cov.shape[0]
    # 计算距离矩阵：d = sqrt(0.5 * (1 - rho))
    corr = np.corrcoef(cov)
    dist = np.sqrt(0.5 * (1 - corr))
    # 压缩距离矩阵（上三角）
    dist_condensed = squareform(dist, checks=False)
    # 层次聚类
    link = linkage(dist_condensed, method=method)
    # 准对角化
    ordered_indices = _get_quasi_diag(link)
    # 递归计算权重（在准对角顺序上）
    weights_ordered = _hrp_weights_recursive(cov, ordered_indices)
    # 映射回原始顺序
    weights = np.zeros(n)
    for orig_idx, w in zip(ordered_indices, weights_ordered):
        weights[orig_idx] = w
    return weights

# ==================== 2. 增强型 BL 融合（校准置信度） ====================

def step_m_2_bl_fusion_calibrated(context: dict, date: datetime, prev_weights: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    改进版 BL 融合：使用校准概率的方差（Brier Score 变体）填充 Ω 对角。
    若校准概率不可用，则回退到原始区间宽度逻辑。
    """
    # 此处仅示意接口，实际可复用 bl_fusion.py 中的代码，仅替换 Omega 填充方式。
    # 由于原 bl_fusion 已使用宽度，我们可以增加一个选项。
    # 为了保持兼容，直接调用原函数，但在增强模块中可提供替代函数。
    from Phase_6.bl_fusion import step_m_2_black_litterman_fusion
    return step_m_2_black_litterman_fusion(context, date, prev_weights)

# ==================== 3. 可替换凸优化的 HRP 入口 ====================

def step_m_3_hrp_optimization(
    context: dict,
    date: datetime,
    nav: float,
    prev_weights: Optional[np.ndarray] = None,
    turnover_lambda: float = 0.0003,
    min_trade_threshold: float = 0.005
) -> np.ndarray:
    """
    替代 step_m_3_convex_optimization 的 HRP 分配器。
    输入：context（含 R_BL, Sigma_robust, 方向掩码, sector_map, config 等）
    输出：满足约束的权重向量
    """
    assets = context['assets']
    n = len(assets)
    R_BL = context.get('R_BL')
    Sigma = context.get('Sigma_robust')
    if R_BL is None or Sigma is None:
        raise RuntimeError("Missing R_BL or Sigma_robust – run BL fusion first.")
    
    config = context.get('config', {})
    sector_limit = config.get('sector_limit', 0.3)
    # 获取方向掩码（只允许多头资产参与）
    masks = context.get('directional_symbol_masks', {})
    # 获取行业映射
    sector_map = context.get('sector_map', {})
    if not sector_map:
        sector_map = {s: "综合" for s in assets}
    # 获取流动性上限（与凸优化一致）
    bus = context['data_bus']
    from Phase_6.utils import _compute_individual_shares_upper
    upper_bounds = np.array([_compute_individual_shares_upper(bus, s, date, nav, config) for s in assets])
    
    # ---- 1. 计算原始 HRP 权重（基于 Sigma） ----
    raw_weights = hrp_weights(Sigma, method='ward')
    
    # ---- 2. 施加方向掩码：仅保留多头资产，其余置零 ----
    mask_vec = np.array([masks.get(s, 0) for s in assets])
    if np.sum(mask_vec) == 0:
        # 无多头信号，返回防御性均匀分布（或等权）
        return np.ones(n) / n
    raw_weights = raw_weights * mask_vec
    if np.sum(raw_weights) > 0:
        raw_weights = raw_weights / np.sum(raw_weights)
    else:
        return np.ones(n) / n
    
    # ---- 3. 行业敞口约束（投影） ----
    # 若某行业超限，则将该行业内权重按比例压缩，使总和 <= sector_limit
    unique_sectors = list(set(sector_map.values()))
    for sec in unique_sectors:
        idx = [i for i, s in enumerate(assets) if sector_map.get(s) == sec]
        if idx:
            sec_weight = np.sum(raw_weights[idx])
            if sec_weight > sector_limit:
                scale = sector_limit / sec_weight
                raw_weights[idx] *= scale
    # 归一化（确保总和=1）
    raw_weights = raw_weights / np.sum(raw_weights)
    
    # ---- 4. 个股流动性硬上限 ----
    raw_weights = np.minimum(raw_weights, upper_bounds)
    if np.sum(raw_weights) == 0:
        return np.ones(n) / n
    raw_weights = raw_weights / np.sum(raw_weights)
    
    # ---- 5. 换手率惩罚（L1 正则化近似） ----
    # 直接对权重进行截断：仅当偏离 prev_weights 超过阈值时调整，否则保持原权重。
    if prev_weights is not None:
        diff = raw_weights - prev_weights
        # 计算换手成本：sum(|diff|) * turnover_lambda，但因为 HRP 输出无解析解，我们采用软阈值：
        # 如果 diff 小于阈值，则将该部分置零，实现低换手
        soft_threshold = min_trade_threshold
        # 应用软阈值：将小于阈值的 diff 置零，但保留符号
        mask_diff = np.abs(diff) > soft_threshold
        # 调整：仅对变化较大的资产进行更新，其余维持前值
        new_weights = prev_weights.copy()
        # 对于变化大的资产，将原diff值加上prev_weights，然后调整总和
        for i in range(n):
            if mask_diff[i]:
                new_weights[i] = prev_weights[i] + diff[i]
            else:
                new_weights[i] = prev_weights[i]  # 不变
        # 重新归一化（确保和=1）
        new_weights = np.clip(new_weights, 0, None)
        if np.sum(new_weights) > 0:
            new_weights = new_weights / np.sum(new_weights)
        else:
            new_weights = raw_weights
        raw_weights = new_weights
    
    # ---- 6. 最终裁剪与归一化 ----
    raw_weights = np.nan_to_num(raw_weights, nan=0.0)
    raw_weights = np.clip(raw_weights, 0, 1)
    if np.sum(raw_weights) == 0:
        return np.ones(n) / n
    return raw_weights / np.sum(raw_weights)

# ==================== 4. 装饰器（扩展原约束） ====================

def hrp_constraint_enforcer(max_weight=0.15, min_weight=0.0):
    """
    增强版约束装饰器：对任意权重向量施加 [min, max] 裁剪 + 归一化
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            weights = func(*args, **kwargs)
            weights = np.clip(weights, min_weight, max_weight)
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(weights)) / len(weights)
            return weights
        return wrapper
    return decorator

# ==================== 5. 简易切换函数（供主流程调用） ====================

def use_hrp_allocation(context: dict, date: datetime, nav: float, prev_weights: Optional[np.ndarray] = None) -> np.ndarray:
    """
    与凸优化函数签名完全一致，可直接替换 step_m_3_convex_optimization。
    """
    return step_m_3_hrp_optimization(context, date, nav, prev_weights)

# ==================== 6. 辅助函数：校准置信度计算 ====================

def compute_calibrated_omega(calibrated_probs: np.ndarray, tau: float = 0.02) -> np.ndarray:
    """
    根据校准概率计算观点协方差 Ω 的对角元素。
    使用 Brier Score 的变体：var(p) = p*(1-p) / N (或类似)，此处简化取 p*(1-p)
    :param calibrated_probs: n×k 概率矩阵（每行是各分类的概率）
    :param tau: 缩放因子
    :return: 对角阵（n×n）
    """
    # 取上涨概率（假设索引为2，或根据实际分类器调整）
    # 假设 calibrated_probs 形状为 (n, 3) 对应 [负, 中性, 正]
    if calibrated_probs.shape[1] == 3:
        prob_pos = calibrated_probs[:, 2]
    else:
        prob_pos = calibrated_probs[:, 1]  # 二分类
    # 方差 = p*(1-p)
    var = prob_pos * (1 - prob_pos)
    # 添加极小值防止零
    var = np.clip(var, 1e-6, 0.25)
    Omega = np.diag(var * tau)
    return Omega

# ==================== 示例用法 ====================
if __name__ == "__main__":
    # 测试 HRP
    np.random.seed(42)
    n_assets = 10
    cov = np.random.randn(n_assets, n_assets)
    cov = cov @ cov.T
    cov += np.eye(n_assets) * 0.01
    w = hrp_weights(cov)
    print("HRP weights:", w)
    print("Sum:", np.sum(w))