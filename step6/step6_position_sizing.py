import logging
import pandas as pd
import numpy as np
from datetime import datetime
from .config import DEFAULT_CONFIG
from .directional_mask import step_m_1_directional_mask
from .bl_fusion import step_m_2_black_litterman_fusion
from .convex_optimizer import step_m_3_convex_optimization
from .utils import _fetch_borrowable_stocks

logger = logging.getLogger("PositionSizing")

def execute(pipeline_context: dict) -> dict:
    logger.info("=" * 60)
    logger.info("Phase 6: 多空双向不确定性头寸分配与凸优化 [模块化引擎调度]")
    logger.info("=" * 60)

    # 1. 深度影子复制全局配置，绝对不允许修改、硬编码篡改全局全局公共参数
    config = pipeline_context.get('config', {}).copy()
    for k, v in DEFAULT_CONFIG.items():
        if k not in config:
            config[k] = v
            
    # 构建当前阶段的独立执行局部沙盒上下文，保护全局管道不被污染
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
    weight_records = []
    interval_records = []
    prev_weights = None
    
    # 预计算阶段假设资产初始基础净值 NAV = 1.0 
    current_nav = 1.0
    total_days = len(test_dates)
    logger.info(f"[头寸分配器拉起] 成功读取 Test 验证集，共包含 {total_days} 个交易日，开始全量矩阵预计算...")

    # 物理清空跨阶段总线缓存残值，强控硬盘物理重新交互，阻断潜在的内存泄露和污染
    if hasattr(local_context['data_bus'], '_cache'):
        local_context['data_bus']._cache.clear()
        logger.info("[总线防火墙启动] 成功清除跨阶段残留内存缓存，强控物理磁盘加载安全对齐。")

    for idx, date in enumerate(test_dates):
        # 兼容处理前置环节传递过程中的可能发生的时间强类型断层 (str / datetime / Timestamp)
        date_dt = pd.to_datetime(date) if isinstance(date, str) else date

        local_context['current_date'] = date_dt
        
        # 逐日实时动态审计全池融券池，确定个股今日做空约束
        local_context['borrowable_today'] = _fetch_borrowable_stocks(date_dt)
        
        # 顺序执行原子管道级串联调度
        masks = step_m_1_directional_mask(local_context, date_dt)
        local_context['directional_symbol_masks'] = masks
        
        R_BL, Sigma, q_low, q_high = step_m_2_black_litterman_fusion(local_context, date_dt, prev_weights)
        weights = step_m_3_convex_optimization(local_context, date_dt, current_nav, prev_weights)

        weight_records.append(weights)
        interval_records.append((q_low, q_high))
        prev_weights = weights
        
        if (idx + 1) % 50 == 0 or (idx + 1) == total_days:
            logger.info(f" ⌛ [头寸计算中] 矩阵预计算进度: {idx + 1} / {total_days} 个交易日优化收敛完成...")

    # 面板结构重新对齐规整化 DataFrame
    weight_df = pd.DataFrame(weight_records, index=test_dates, columns=assets)
    
    q_low_df = pd.DataFrame([low for low, _ in interval_records], index=test_dates, columns=assets)
    q_high_df = pd.DataFrame([high for _, high in interval_records], index=test_dates, columns=assets)

    # 最终结果平稳写回原始全局总线管道中，向前和向后适配
    pipeline_context['daily_weights'] = weight_df
    pipeline_context['daily_intervals'] = {'q_low': q_low_df, 'q_high': q_high_df}
    pipeline_context['target_weights'] = {assets[i]: prev_weights[i] for i in range(n_assets)}
    pipeline_context['allocation_weights_ready'] = True
    
    logger.info(f"✅ Step 6 头寸分配与凸优化面板全量计算成功，面板维度: {weight_df.shape} 规整落盘！")
    return pipeline_context