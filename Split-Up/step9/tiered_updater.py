"""
Quant-Ultra Flow - Step 9.2: Tiered MLOps Updating Protocols & Drift Management
Calculates precise PSI metrics and coordinates linear staircase model transitions.
Fully refactored to reuse Phase 5 training feature space network (Resolves Flaw A-3).
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from .config import PSI_THRESHOLD, PSI_CONSECUTIVE_DAYS, SMOOTHING_PERIOD, LOOKBACK_WINDOW_PSI

logger = logging.getLogger("MLOps.TieredUpdater")

def evaluate_distribution_drift(context: dict) -> dict:
    """
    固定分箱 PSI 稳定性追踪审计。
    核心修复 A-3：彻底废除本地3维粗糙因子拼凑，全面复用阶段5固化的定点高维特征魔方 (fractional_features_cube)，
    实现监控特征空间与训练特征空间（5维共享+各节点垂直私有特征）的同质化对齐。
    """
    logger.info("[Step 9.2] 启动高维特征时序分布 PSI 联合审计协议。")
    
    target_weights = context.get('target_weights', {})
    current_date_str = context.get('current_date')
    
    # 优先穿透提取阶段5输出的完整高维特征魔方
    feature_cube = context.get('fractional_features_cube')
    selected_features = context.get('selected_features', None)
    
    if isinstance(current_date_str, datetime):
        current_date_str = current_date_str.strftime("%Y-%m-%d")
    elif current_date_str is None:
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        
    if feature_cube horizon_data is None if isinstance(feature_cube, type(None)) else False:
        # 若高维魔方未就绪，向阶段3共享面板降级融合
        feature_cube = context.get('feature_panel_shared')

    if feature_cube is None or not target_weights:
        logger.warning("🚨 [PSI审计挂起] 全局总线未检测到预计算的高维特征魔方或资产清单。")
        return context

    assets = list(target_weights.keys())
    
    # ====================================================
    # 核心修复 A-3：自适应解析 MultiIndex 结构，提取真实模型特征
    # ====================================================
    try:
        # 确保 DataFrame 索引规范对齐 (date, asset)
        df_features = feature_cube.copy()
        if not isinstance(df_features.index, pd.MultiIndex):
            logger.error("特征魔方格式不符合 MultiIndex (date, asset) 规范，挂起审计。")
            return context
            
        # 精准锁定列空间（若存在特征选择则过滤，否则全量对齐5维+私有特征）
        feature_cols = selected_features if selected_features else list(df_features.columns)
        df_features = df_features[feature_cols]
        
        # 2. 动态切分当前观测窗口与历史基准窗口
        end_dt = pd.to_datetime(current_date_str)
        start_current_dt = end_dt - timedelta(days=LOOKBACK_WINDOW_PSI)
        
        start_baseline_dt = start_current_dt - timedelta(days=LOOKBACK_WINDOW_PSI)
        end_baseline_dt = start_current_dt - timedelta(days=1)
        
        # 横截面索引切片抽取
        idx = pd.IndexSlice
        current_slice = df_features.loc[idx[start_current_dt:end_dt, assets], :]
        baseline_slice = df_features.loc[idx[start_baseline_dt:end_baseline_dt, assets], :]
        
        if current_slice.empty or baseline_slice.empty:
            logger.warning("⚠️ 当前窗口或基准窗口内的特征样本数不足，跳过本轮分布审计。")
            return context
            
        current_matrix = current_slice.to_numpy()
        baseline_matrix = baseline_slice.to_numpy()
        
    except Exception as ex:
        logger.error(f"🚨 从高维特征矩阵中切片真实训练特征空间发生致命溃败: {str(ex)}")
        return context

    # 3. 严格应用 10 箱等频固定分箱规则计算高维多列 PSI
    num_bins = 10
    psi_by_feature = []
    
    for col_idx in range(baseline_matrix.shape[1]):
        base_col = baseline_matrix[:, col_idx]
        curr_col = current_matrix[:, col_idx]
        
        # 剔除空值干扰
        base_col = base_col[~np.isnan(base_col)]
        curr_col = curr_col[~np.isnan(curr_col)]
        if len(base_col) == 0 or len(curr_col) == 0:
            continue
            
        combined_vals = np.concatenate([base_col, curr_col])
        quantiles = np.percentile(combined_vals, np.linspace(0, 100, num_bins + 1))
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        
        def calculate_probabilities(values):
            counts = np.zeros(num_bins)
            for val in values:
                for b in range(num_bins):
                    if quantiles[b] <= val < quantiles[b+1]:
                        counts[b] += 1
                        break
            probs = counts / len(values) if len(values) > 0 else np.ones(num_bins) / num_bins
            return np.clip(probs, 1e-6, 1.0)
            
        base_probs = calculate_probabilities(base_col)
        curr_probs = calculate_probabilities(curr_col)
        
        # 严格执行经典 PSI 金融统计公式
        psi_val = float(np.sum((curr_probs - base_probs) * np.log(curr_probs / base_probs)))
        psi_by_feature.append(psi_val)

    if not psi_by_feature:
        return context
        
    mean_psi = float(np.mean(psi_by_feature))
    context['current_mean_psi'] = mean_psi
    logger.info(f"✅ 模型全量高维特征(维度={len(psi_by_feature)})平稳度审计完成：平均 PSI = {mean_psi:.4f}")

    # 累加连续触警交易日天数
    consecutive_breaches = context.get('psi_consecutive_breaches', 0)
    if mean_psi > PSI_THRESHOLD:
        consecutive_breaches += 1
        logger.warning(f"⚠️ [平稳度触警] 生产环境高维特征空间 PSI 突破警戒线！当前连续天数: {consecutive_breaches}/{PSI_CONSECUTIVE_DAYS}")
    else:
        consecutive_breaches = 0
    context['psi_consecutive_breaches'] = consecutive_breaches

    # 综合研判是否激活 Tier 3 分布式联邦迁移重训总线
    trigger_retrain = False
    if consecutive_breaches >= PSI_CONSECUTIVE_DAYS:
        logger.critical(f"🚨 特征空间分布失真连续 {PSI_CONSECUTIVE_DAYS} 天超标，物理激活 Tier 3 全量联邦重训总线。")
        trigger_retrain = True
    elif context.get('negative_migration_detected', False):
        logger.critical("🚨 检测到跨市场对抗微调触发负迁移红线！强行激活重训总线，执行本地模型全方位回归。")
        trigger_retrain = True

    # 4. 新老模型线性步进切换平滑控制
    transition_day = context.get('transition_day', 0)
    if trigger_retrain:
        context['tier3_retrain_active'] = True
        context['transition_day'] = 1
        context['alpha_new_model'] = 1.0 / SMOOTHING_PERIOD
    elif transition_day > 0:
        if transition_day >= SMOOTHING_PERIOD:
            logger.info("🎉 [生产轨无缝接管] 新模型平滑步进切换完毕，旧模型执行物理销毁。")
            context['transition_day'] = 0
            context['alpha_new_model'] = 1.0
            context['tier3_retrain_active'] = False
        else:
            transition_day += 1
            alpha_new = transition_day / float(SMOOTHING_PERIOD)
            context['transition_day'] = transition_day
            context['alpha_new_model'] = alpha_new
            logger.info(f"[新老模型平滑混配中] 步进周期: {transition_day}/{SMOOTHING_PERIOD}，新内核权重 = {alpha_new:.2f}")
    else:
        context['tier3_retrain_active'] = False
        context['alpha_new_model'] = 0.0

    return context