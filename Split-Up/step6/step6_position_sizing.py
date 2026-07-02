import logging
import os
import pandas as pd
import numpy as np
from datetime import datetime
from .config import DEFAULT_CONFIG
from .directional_mask import step_m_1_directional_mask
from .bl_fusion import step_m_2_black_litterman_fusion
from .convex_optimizer import step_m_3_convex_optimization

logger = logging.getLogger("PositionSizing")

def execute(pipeline_context: dict) -> dict:
    logger.info("=" * 60)
    logger.info("Phase 6: 纯多头现货不确定性头寸分配与个人化凸优化 [联邦合规升级]")
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
    
    # ------------------ 会话缓存校验体系（响应 README 挂起任务） ------------------
    # 以测试集的边界日期令牌构建缓存唯一标志符
    cache_id = f"phase6_records_{test_dates[0]}_{test_dates[-1]}".replace("-", "")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # 定位到 Split-Up/
    parquet_dir = os.path.join(base_dir, "Phase Result", "parquet", "Phase 6")
    feather_dir = os.path.join(base_dir, "Phase Result", "feather", "Phase 6")
    
    parquet_path = os.path.join(parquet_dir, f"{cache_id}.parquet")
    feather_path = os.path.join(feather_dir, f"{cache_id}.feather")
    
    update_required = config.get('update_required_phase6', False)
    
    if os.path.exists(parquet_path) and not update_required:
        logger.info(f"[会话缓存命中] 发现历史已落盘头寸分配记录，执行快速直通短路：{parquet_path}")
        weight_df = pd.read_parquet(parquet_path)
        pipeline_context['daily_weights'] = weight_df
        pipeline_context['target_weights'] = weight_df.iloc[-1].to_dict()
        pipeline_context['allocation_weights_ready'] = True
        return pipeline_context
    # -------------------------------------------------------------------------

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

    # ------------------ 会话缓存双格式持久化落盘（响应 README 挂起任务） ------------------
    os.makedirs(parquet_dir, exist_ok=True)
    os.makedirs(feather_dir, exist_ok=True)
    
    # 固化落盘供本 Python 链条内部消费的 Parquet 格式
    weight_df.to_parquet(parquet_path)
    # 固化落盘供未来 C 语言消费的 Feather 格式（重置索引确保原生存储特征平稳）
    weight_df.reset_index().to_feather(feather_path)
    logger.info(f"[双格式持久化完成] 成功建立物理缓存防线: {parquet_path} 与 {feather_path}")
    # -------------------------------------------------------------------------

    pipeline_context['daily_weights'] = weight_df
    pipeline_context['daily_intervals'] = {'q_low': q_low_df, 'q_high': q_high_df}
    pipeline_context['target_weights'] = {assets[i]: prev_weights[i] for i in range(n_assets)}
    pipeline_context['allocation_weights_ready'] = True
    
    logger.info(f"✅ Step 6 凸优化多头头寸分配面板全量计算完成，落盘维度: {weight_df.shape}！")
    return pipeline_context