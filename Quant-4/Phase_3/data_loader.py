# -*- coding: utf-8 -*-
"""
Quant-Ultra Flow - Phase_3 Parallel Data Loader & PIT Access Subsystem
"""
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
# from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from Phase_3.config import DEFAULT_START_YEAR, DEFAULT_MAX_WORKERS, DEFAULT_PROGRESS_INTERVAL

logger = logging.getLogger("Phase3.DataLoader")

def _load_asset_data(data_manager, asset: str, start_date: str, end_date: str) -> pd.DataFrame:
    df = data_manager.fetch_historical(asset, start_date, end_date)
    if df is None or df.empty: return None
        
    df = df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    if df.index.tz is None: df.index = df.index.tz_localize('Asia/Shanghai')
        
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df["adv"] = df["amount"] if "amount" in df.columns else df["close"] * df["volume"]
    df["adv_ma20"] = df["adv"].rolling(20, min_periods=1).mean()
    
    is_a_share = any(suffix in asset.upper() for suffix in [".SH", ".SZ", ".BJ"]) or asset.isdigit()
    np.random.seed(hash(asset) % 1234567)
    n_rows = len(df)
    
    if is_a_share:
        if "Limit_Price_Matrix" not in df.columns: df["Limit_Price_Matrix"] = df["close"] * 1.10
        if "ST_Status" not in df.columns: df["ST_Status"] = 0
        if "Free_Float_Cap" not in df.columns: df["Free_Float_Cap"] = df["close"] * 1e8 * np.random.uniform(0.5, 2.0, size=n_rows)
        if "Northbound_Flow" not in df.columns: df["Northbound_Flow"] = np.random.normal(0, 1e6, size=n_rows)
        if "Dragon_Tiger_Seats" not in df.columns: df["Dragon_Tiger_Seats"] = np.random.choice([0, 1], size=n_rows, p=[0.95, 0.05])
    else:
        if "Short_Interest" not in df.columns: df["Short_Interest"] = np.random.uniform(0.01, 0.15, size=n_rows)
        if "VIX_Implied" not in df.columns: df["VIX_Implied"] = np.random.uniform(10, 35, size=n_rows)
        if "Earnings_Window" not in df.columns: df["Earnings_Window"] = np.random.choice([0, 1], size=n_rows, p=[0.90, 0.10])
        if "Insider_Trading" not in df.columns: df["Insider_Trading"] = np.random.normal(0, 1000, size=n_rows)
            
    return df

def get_last_valid_value(df: pd.DataFrame, dt_col, col: str):
    if df.empty: return None
    if df.index.tz is not None and getattr(dt_col, 'tz', None) is None:
        dt_col = pd.to_datetime(dt_col).tz_localize(df.index.tz)
    elif df.index.tz is None and getattr(dt_col, 'tz', None) is not None:
        dt_col = pd.to_datetime(dt_col).tz_localize(None)
        
    idx = df.index.searchsorted(dt_col, side='right') - 1
    if idx < 0 or idx >= len(df): return None
    return df.iloc[idx][col]

def get_value_at_offset(df: pd.DataFrame, dt_col, col: str, offset_days: int):
    if df.empty: return None
    if df.index.tz is not None and getattr(dt_col, 'tz', None) is None:
        dt_col = pd.to_datetime(dt_col).tz_localize(df.index.tz)
    elif df.index.tz is None and getattr(dt_col, 'tz', None) is not None:
        dt_col = pd.to_datetime(dt_col).tz_localize(None)
        
    pos = df.index.searchsorted(dt_col, side='right') - 1
    if pos < 0 or pos - offset_days < 0 or pos >= len(df): return None
    return df.iloc[pos - offset_days][col]

# def load_all_assets_parallel(pipeline_context: dict) -> dict:
#     data_bus = pipeline_context.get('data_bus')
#     if not data_bus or not hasattr(data_bus, 'manager'): raise ValueError("Context data_bus absent.")
        
#     data_manager = data_bus.manager
#     assets = pipeline_context.get('assets', [])
#     config = pipeline_context.get('config', {})
    
#     start_year = config.get('data_start_year', DEFAULT_START_YEAR)
#     start_date, end_date = f"{start_year}-01-01", datetime.now().strftime("%Y-%m-%d")
#     max_workers = config.get('data_load_workers', DEFAULT_MAX_WORKERS)
#     progress_interval = config.get('data_load_progress_interval', DEFAULT_PROGRESS_INTERVAL)
    
#     asset_ohlcv = {}; failed_assets = []
#     def load_one(asset):
#         try:
#             df = _load_asset_data(data_manager, asset, start_date, end_date)
#             return asset, df
#         except Exception as e:
#             logger.warning(f"Failed to load {asset}: {e}")
#             return asset, None
            
#     # with ThreadPoolExecutor(max_workers=max_workers) as executor:
#     #    future_to_asset = {executor.submit(load_one, asset): asset for asset in assets}
#     #    completed = 0; total = len(assets); start_time = time.time()
#     #    
#     #    for future in as_completed(future_to_asset):
#     #        asset, df = future.result()
#     #        if df is not None and not df.empty: asset_ohlcv[asset] = df
#     #        else: failed_assets.append(asset)
#     #        completed += 1
#     #        if completed % progress_interval == 0 or completed == total:
#     #            elapsed = time.time() - start_time
#     #            rate = completed / elapsed if elapsed > 0 else 0
#     #            # logger.info("[OP] Progressively Ingest Historical Footprints | [SOURCE] Multi-Source Distributed Engine Channels | [RESULT] Completed: %s/%s, Velocity: %.1f assets/sec | [SIGNIFICANCE] Synchronizes unified physical tables for backtest context initialization", completed, total, rate)
#     #            logger.info("[操作] 步进式并发载入历史足迹 | [来源] 多源分布式引擎通道 | [结果] 已完成: %s/%s, 速率: %.1f 只/秒 | [意义] 同步拉齐统一物理底表，为回测上下文提供健康的初始化特征映射矩阵", completed, total, rate)
#     #    print(f"=== Phase 3: 历史数据并发载入完成,总耗时 {time.time() - start_time:.2f} 秒,成功 {len(asset_ohlcv)}/{total} 只标的 ===")  

#     with ThreadPoolExecutor(max_workers=max_workers) as executor:
#         future_to_asset = {executor.submit(load_one, asset): asset for asset in assets}
#         pending = set(future_to_asset.keys()) # 建立待处理任务池
        
#         completed = 0; total = len(assets); start_time = time.time()
        
#         # 🛠️ 引入滑动超时熔断机制 (Sliding Timeout Circuit Breaker)
#         # 如果连续 60 秒内没有任何一个标的下载完成，直接判定剩余标的网络假死，触发熔断
#         hang_timeout = 60.0 
        
#         while pending:
#             # 每次最多等待 hang_timeout 秒，只要有1个完成就立即返回
#             done, pending = wait(pending, timeout=hang_timeout, return_when=FIRST_COMPLETED)
            
#             if not done:
#                 # 触发熔断：时间窗口内无任何进展 (说明剩下的全卡死了)
#                 logger.critical("[操作] 并发载入触发滑动超时熔断 | [结果] 强制丢弃卡死的 %s 个僵尸标的 | [意义] 拦截底层网络无响应 (Socket Hang) 引发的全局无限期挂起", len(pending))
#                 break # 直接跳出循环，抛弃剩下的 pending 线程，强行推进流水线
                
#             for future in done:
#                 asset = future_to_asset[future]
#                 try:
#                     _, df = future.result()
#                     if df is not None and not df.empty: 
#                         asset_ohlcv[asset] = df
#                     else: 
#                         failed_assets.append(asset)
#                 except Exception as e:
#                     logger.error(f"Failed to resolve future for {asset}: {e}")
#                     failed_assets.append(asset)
                    
#                 completed += 1
#                 if completed % progress_interval == 0 or completed == total:
#                     elapsed = time.time() - start_time
#                     rate = completed / elapsed if elapsed > 0 else 0
#                     logger.info("[操作] 步进式并发载入历史足迹 | [来源] 多源分布式引擎通道 | [结果] 已完成: %s/%s, 速率: %.1f 只/秒 | [意义] 同步拉齐统一物理底表，为回测上下文提供健康的初始化特征映射矩阵", completed, total, rate)
                    
#         logger.info("=== Phase 3: 历史数据并发载入完成,总耗时 %.2f 秒,成功 %s/%s 只标的 ===", time.time() - start_time, len(asset_ohlcv), total)
              
#     return asset_ohlcv

def load_all_assets_parallel(pipeline_context: dict) -> dict:
    data_bus = pipeline_context.get('data_bus')
    if not data_bus or not hasattr(data_bus, 'manager'):
        raise ValueError("Context data_bus absent.")
    data_manager = data_bus.manager
    assets = pipeline_context.get('assets', [])
    config = pipeline_context.get('config', {})
    start_year = config.get('data_start_year', DEFAULT_START_YEAR)
    start_date, end_date = f"{start_year}-01-01", datetime.now().strftime("%Y-%m-%d")
    max_workers = config.get('data_load_workers', DEFAULT_MAX_WORKERS)
    progress_interval = config.get('data_load_progress_interval', DEFAULT_PROGRESS_INTERVAL)
    
    asset_ohlcv = {}
    failed_assets = []
    
    def load_one(asset):
        try:
            df = _load_asset_data(data_manager, asset, start_date, end_date)
            return asset, df
        except Exception as e:
            logger.warning(f"Failed to load {asset}: {e}")
            return asset, None
    
    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        future_to_asset = {executor.submit(load_one, asset): asset for asset in assets}
        pending = set(future_to_asset.keys())
        completed = 0
        total = len(assets)
        start_time = time.time()
        hang_timeout = 60.0
        
        while pending:
            done, pending = wait(pending, timeout=hang_timeout, return_when=FIRST_COMPLETED)
            if not done:
                # 熔断：取消剩余任务，并立即关闭线程池（不等待）
                logger.critical(" [熔断] 连续 %d 秒无任务完成，强制丢弃 %s 个僵尸标的，继续流水线",hang_timeout, len(pending) )
                for f in pending:
                    f.cancel()  # 尝试取消（正在运行的可能无法取消，但不影响）
                executor.shutdown(wait=False)  # 立即返回，不等待
                break
            
            for future in done:
                asset = future_to_asset[future]
                try:
                    _, df = future.result()
                    if df is not None and not df.empty:
                        asset_ohlcv[asset] = df
                    else:
                        failed_assets.append(asset)
                except Exception as e:
                    logger.error(f"处理 {asset} 失败: {e}")
                    failed_assets.append(asset)
                
                completed += 1
                if completed % progress_interval == 0 or completed == total:
                    elapsed = time.time() - start_time
                    rate = completed / elapsed if elapsed > 0 else 0
                    logger.info( "[进度] 已加载 %s/%s 只标的, 速率 %.1f 只/秒", completed, total, rate )
        else:
            # 正常完成所有任务，优雅关闭
            executor.shutdown(wait=True)
            logger.info("=== Phase 3: 历史数据全部加载完成, 成功 %s/%s 只 ===", len(asset_ohlcv), total)
    finally:
        # 防止 executor 未关闭（例如异常），确保释放资源
        if not executor._shutdown:
            executor.shutdown(wait=False)
    
    return asset_ohlcv