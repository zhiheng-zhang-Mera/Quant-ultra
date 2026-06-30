"""
Phase 2: Full-Flow Two-Tier Data Slicing and Isolation Architecture
High-Level Lifecycle Orchestrator Bus.
Correctly binds atomic submodules: config, acf_analyzer, step2_1_slicing, step2_2_validation
"""
import os
import pytz
import logging
import pandas as pd
from datetime import datetime

# 💡 核心修正：从您打散的原子子模块中引流，消灭死码生存
from step2.step2_1_slicing import run_moving_window_slicing
from step2.step2_2_validation import run_purge_and_embargo_validation

logger = logging.getLogger("DataSlicing")


def _determine_start_year(pipeline_context: dict) -> int:
    """
    【智能时空对齐器 - 最晚起点健壮过滤版】
    动态扫描本地缓存区，提取样本资产库中合理的最新上市/下载时间点，刚性确保横截面数据满流。
    """
    data_bus = pipeline_context.get('data_bus')
    data_manager = pipeline_context.get('data_manager')
    
    if not data_manager and data_bus and hasattr(data_bus, 'manager'):
        data_manager = data_bus.manager
        
    assets = pipeline_context.get('assets', [])
    fallback_year = 2022  # 安全兜底年份
    
    if not data_manager or not hasattr(data_manager, 'cache_dir'):
        return fallback_year
        
    cache_dir = data_manager.cache_dir
    if not os.path.exists(cache_dir):
        return fallback_year

    # 抽取核心资产作为代表性截面
    raw_samples = assets[:100] if assets else [
        f.split('_')[0] for f in os.listdir(cache_dir) if f.endswith('.parquet')
    ][:100]
    
    if not raw_samples:
        return fallback_year

    earliest_dates = []
    
    # 执行无感轻量级 I/O 探测
    for sym in raw_samples:
        file_path = os.path.join(cache_dir, f"{sym}_history.parquet")
        if os.path.exists(file_path):
            try:
                # 性能优化：只读取 date 列的第 1 行
                df_head = pd.read_parquet(file_path, columns=['date'], engine='pyarrow').head(1)
                if not df_head.empty:
                    first_date = df_head['date'].iloc[0]
                    # 刚性边界限制：剔除 2024 年之后的极端次新股，防止回测时空极度压缩
                    if pd.to_datetime(first_date).year < 2024:
                        earliest_dates.append(first_date)
            except Exception:
                continue

    if earliest_dates:
        global_latest_start_date = max(earliest_dates)
        if isinstance(global_latest_start_date, str):
            global_latest_start_date = pd.to_datetime(global_latest_start_date)
            
        determined_year = int(global_latest_start_date.year)
        logger.info(f"[智能对齐] 📥 严格交集审计：抽样池中最晚有效源头锁定为: {global_latest_start_date.strftime('%Y-%m-%d')}，全局起点: {determined_year} 年")
        return determined_year

    return fallback_year


def execute(pipeline_context: dict) -> dict:
    """
    Step 2 全生命周期调度主入口（包级总线版）
    """
    logger.info("====== 开始执行阶段 2: 数据切片与隔离体系构建 ======")
    
    # --------------------------------------------------------
    # 🛡️ 刚性内存冲刷：彻底粉碎 Step 1 残留的毒化内存缓存
    # 强迫 Step 2 顺次调用时，必须重新去硬盘解密健康的、深度的 Parquet 数据
    # --------------------------------------------------------
    if 'data_bus' in pipeline_context and hasattr(pipeline_context['data_bus'], '_cache'):
        pipeline_context['data_bus']._cache.clear()
        logger.info("[总线隔离] 🧼 成功冲刷跨阶段内存残留，物理 I/O 隔离带已构建")

    # 1. 强力探测本地硬盘数据实际起点
    detected_start_year = _determine_start_year(pipeline_context)
    current_calendar = pipeline_context.get('trading_days_dt', [])
    need_rebuild = False
    
    # 2. 主动纠偏逻辑：对抗前置快照带来的老日历污染
    if not current_calendar:
        need_rebuild = True
    else:
        cal_start_year = current_calendar[0].year
        if cal_start_year != detected_start_year:
            logger.warning(
                f"[智能对齐] ⚠️ 拦截到时空断层污染！前置快照自带日历起点为 {cal_start_year} 年，"
                f"但集群核心有效数据起点应为 {detected_start_year} 年。强制清刷并重构交易日历！"
            )
            need_rebuild = True

    # 3. 强行重置纠偏
    if need_rebuild:
        current_year = datetime.now().year
        cal = pipeline_context.get('data_bus').manager.fetch_trading_calendar(detected_start_year, current_year)
        pipeline_context['trading_days_dt'] = cal.tz_localize(pytz.timezone("Asia/Shanghai")).tolist()
        logger.info(f"[智能对齐] ✅ 全局交易日历已成功对齐重构，全新时空范围: {detected_start_year} ~ {current_year}")

    # 4. 顺次拉起原子清洗控制流（正式接通拆分后的子模块）
    run_moving_window_slicing(pipeline_context)
    run_purge_and_embargo_validation(pipeline_context)
    
    pipeline_context['slices_isolated'] = True
    logger.info("====== 阶段 2 执行完毕，管道数据已完全物理隔离 ======")
    return pipeline_context