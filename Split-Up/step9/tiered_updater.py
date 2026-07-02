"""
Quant-Ultra Flow - Step 9.2: Tiered MLOps Updating Protocols & Drift Management
Calculates precise PSI metrics and coordinates linear staircase model transitions.
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
    管理 Tier 3 全量分布式联邦迁移重训总线激活，并维持 25 日线性步进平滑过渡。
    """
    logger.info("[Step 9.2] 触发阶梯式运维更新协议：执行点状特征分布 PSI 审计。")
    
    data_bus = context.get('data_bus')
    target_weights = context.get('target_weights', {})
    current_date_str = context.get('current_date')
    
    if isinstance(current_date_str, datetime):
        current_date_str = current_date_str.strftime("%Y-%m-%d")
    elif current_date_str is None:
        current_date_str = datetime.now().strftime("%Y-%m-%d")
        
    if data_bus is None or not target_weights:
        logger.warning("缺失联邦数据总线 DataBus 或目标权重。挂起 PSI 稳定性审计。")
        return context

    assets = list(target_weights.keys())
    
    # 1. 抽取当前时间轴的真实特征分布矩阵
    current_features = []
    end_dt = datetime.strptime(current_date_str, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=LOOKBACK_WINDOW_PSI + 20)
    
    for asset in assets:
        try:
            df = data_bus.load_asset_history(asset, start_dt.strftime("%Y-%m-%d"), current_date_str)
            if df is None or len(df) < 21:
                continue
            df = df.sort_index()
            # 提炼跨市场核心因子代理：20日对数全收益、20日滚动波动率、成交量滚动残差变动
            ret_20 = float(np.log(df['close'].iloc[-1] / df['close'].iloc[-21]))
            daily_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
            vol_20 = float(daily_returns.tail(20).std())
            turnover_vol = float(df['volume'].tail(20).std() / (df['volume'].tail(20).mean() + 1e-8))
            
            vec = [ret_20, vol_20, turnover_vol]
            if not np.isnan(vec).any():
                current_features.append(vec)
        except Exception:
            continue

    if not current_features:
        logger.warning("当前时段未采集到足够有效特征点，挂起分箱计算。")
        return context
        
    current_matrix = np.array(current_features)  # (N_assets, N_features)

    # 2. 抽取/缓存背景基准特征分布矩阵
    baseline_matrix = context.get('baseline_factor_samples')
    if baseline_matrix is None:
        base_samples = []
        base_end = end_dt - timedelta(days=1)
        base_start = base_end - timedelta(days=LOOKBACK_WINDOW_PSI)
        for asset in assets:
            try:
                df = data_bus.load_asset_history(asset, base_start.strftime("%Y-%m-%d"), base_end.strftime("%Y-%m-%d"))
                if df is None or len(df) < 21:
                    continue
                df = df.sort_index()
                ret_20 = float(np.log(df['close'].iloc[-1] / df['close'].iloc[-21]))
                daily_returns = np.log(df['close'] / df['close'].shift(1)).dropna()
                vol_20 = float(daily_returns.tail(20).std())
                turnover_vol = float(df['volume'].tail(20).std() / (df['volume'].tail(20).mean() + 1e-8))
                
                vec = [ret_20, vol_20, turnover_vol]
                if not np.isnan(vec).any():
                    base_samples.append(vec)
            except Exception:
                continue
        if base_samples:
            baseline_matrix = np.array(base_samples)
            context['baseline_factor_samples'] = baseline_matrix
        else:
            logger.warning("未能成功构建背景基准特征分布参考矩阵。")
            return context

    # 3. 严格应用 10 箱等频固定分箱规则计算 PSI
    num_bins = 10
    psi_by_feature = []
    
    for col_idx in range(baseline_matrix.shape[1]):
        base_col = baseline_matrix[:, col_idx]
        curr_col = current_matrix[:, col_idx]
        
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

    mean_psi = float(np.mean(psi_by_feature))
    context['current_mean_psi'] = mean_psi
    logger.info(f"特征时序平稳度模型计算完成：平均 PSI = {mean_psi:.4f}")

    # 累加连续触警交易日天数
    consecutive_breaches = context.get('psi_consecutive_breaches', 0)
    if mean_psi > PSI_THRESHOLD:
        consecutive_breaches += 1
        logger.warning(f"⚠️ [平稳度触警] 特征 PSI 高于安全警戒线！当前连续天数: {consecutive_breaches}/{PSI_CONSECUTIVE_DAYS}")
    else:
        consecutive_breaches = 0
    context['psi_consecutive_breaches'] = consecutive_breaches

    # 综合研判是否挂起增量流式更新并启动全量重训
    trigger_retrain = False
    if consecutive_breaches >= PSI_CONSECUTIVE_DAYS:
        logger.critical(f"特征 PSI 分布连续 {PSI_CONSECUTIVE_DAYS} 天超标，一键激活 Tier 3 全量联邦重训。")
        trigger_retrain = True
    elif context.get('negative_migration_detected', False):
        logger.critical("检测到 A 股验证集损失连续超标触发负迁移红线！强行强制切入物理重训总线。")
        trigger_retrain = True

    # 4. 新老模型 25 交易日线性步进切换平滑控制逻辑
    transition_day = context.get('transition_day', 0)
    if trigger_retrain:
        logger.info("联邦中央协调器广播重训令牌。挂起在线流式微调，全面刷新领域对抗矩阵。")
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