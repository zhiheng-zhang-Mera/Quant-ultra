import logging
import pandas as pd
import numpy as np
from .config import DEFAULT_CONFIG
from .directional_mask import step_m_1_directional_mask
from .bl_fusion import step_m_2_black_litterman_fusion
from .convex_optimizer import step_m_3_convex_optimization

logger = logging.getLogger("PositionSizing")

def execute(pipeline_context: dict) -> dict:
    logger.info("=" * 60)
    logger.info("Phase 6: 纯多头现货不确定性头寸分配与个人化凸优化 [联邦纯净化计算版]")
    logger.info("=" * 60)

    config = pipeline_context.get('config', {}).copy()
    for k, v in DEFAULT_CONFIG.items():
        if k not in config:
            config[k] = v
            
    local_context = pipeline_context.copy()
    local_context['config'] = config

    if 'assets' not in local_context or not local_context['assets']:
        logger.error("[Orchestrator 物理中止] 全局上下文中未检测到可用的交易池核心资产清单 'assets'。")
        raise ValueError("上下文中缺少 assets。")
        
    slices = local_context.get('slices', {})
    test_dates = slices.get('Test', [])
    if not test_dates:
        logger.error("[Orchestrator 物理中止] 数据隔离切片中的 'Test' 时间轴集为空，权重预计算无资产可用。")
        raise ValueError("Test 集为空，无法计算权重。")

    assets = local_context['assets']
    n_assets = len(assets)

    # =========================================================================
    # 💡 架构演进说明：
    # 彻底移除本地手写的 Phase Result 读写盘逻辑、cache_id 拼接与路径生成。
    # 模块的生命周期、跳过判定及双格式落盘全量上交给 main.py Orchestrator 集中托管。
    # =========================================================================

    weight_records = []
    interval_records = []
    prev_weights = None
    
    individual_account_equity = config.get('individual_account_equity', 10000000.0)
    total_days = len(test_dates)
    logger.info(f"[头寸分配器拉起] 成功读取验证周期，共包含 {total_days} 个交易日，启动前向递推优化...")

    # 清空可能残留的跨阶段内存缓冲区，强控物理磁盘解耦防渗透
    if hasattr(local_context['data_bus'], '_cache'):
        local_context['data_bus']._cache.clear()
        logger.info("[总线防火墙激活] 内存脏残值清除完成，数据流防火墙硬对齐启动。")

    # 执行核心前向递推优化循环
    for idx, date in enumerate(test_dates):
        date_dt = pd.to_datetime(date) if isinstance(date, str) else date
        local_context['current_date'] = date_dt
        
        # 顺序执行原子现货资产管道串联
        masks = step_m_1_directional_mask(local_context, date_dt)
        local_context['directional_symbol_masks'] = masks
        
        R_BL, Sigma, q_low, q_high = step_m_2_black_litterman_fusion(local_context, date_dt, prev_weights)
        local_context['R_BL'] = R_BL
        local_context['Sigma_robust'] = Sigma
        
        # 传入个人总本金底座以执行流动性截断
        weights = step_m_3_convex_optimization(local_context, date_dt, individual_account_equity, prev_weights)

        weight_records.append(weights)
        interval_records.append((q_low, q_high))
        prev_weights = weights
        
        if (idx + 1) % 50 == 0 or (idx + 1) == total_days:
            logger.info(f" ⌛ [头寸计算中] 递推状态：{idx + 1} / {total_days} 个交易日凸优化收敛完成...")

    # 面板结构重新对齐规整化 DataFrame
    weight_df = pd.DataFrame(weight_records, index=test_dates, columns=assets)
    q_low_df = pd.DataFrame([low for low, _ in interval_records], index=test_dates, columns=assets)
    q_high_df = pd.DataFrame([high for _, high in interval_records], index=test_dates, columns=assets)

    # 构造原子阶段仅需向主控返回的资产字典（由 main.py 统一、安全写盘）
    result_update = {
        'daily_weights': weight_df,
        'daily_intervals': {'q_low': q_low_df, 'q_high': q_high_df},
        'target_weights': {assets[i]: prev_weights[i] for i in range(n_assets)},
        'allocation_weights_ready': True
    }
    
    logger.info(f"✅ Step 6 凸优化多头头寸分配面板全量计算完成，向总线提交维度: {weight_df.shape}！")
    return result_update