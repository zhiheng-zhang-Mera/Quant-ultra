# -*- coding: utf-8 -*-
"""
step3/data_loader.py
负责多线程历史行情载入、统一时区清洗与 PIT 时空检索工具
"""

import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from step3.config import DEFAULT_START_YEAR, DEFAULT_MAX_WORKERS, DEFAULT_PROGRESS_INTERVAL

logger = logging.getLogger("Phase3.DataLoader")


def _load_asset_data(data_manager, asset: str, start_date: str, end_date: str) -> pd.DataFrame:
    """加载单个资产的行情，做强类型转换与时区强行对齐"""
    df = data_manager.fetch_historical(asset, start_date, end_date)
    if df is None or df.empty:
        return None
        
    df = df.copy()
    df.set_index("date", inplace=True)
    
    # ---- 统一时区规范 ----
    if df.index.tz is None:
        df.index = df.index.tz_localize('Asia/Shanghai')
        
    # ---- 数值化强类型转换 ----
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df["adv"] = df["amount"]
    df["adv_ma20"] = df["adv"].rolling(20).mean()
    return df


def get_last_valid_value(df: pd.DataFrame, dt_col, col: str):
    """PIT 工具：获取当前截止时刻 T 及其之前的最新有效值"""
    if df.empty:
        return None
    idx = df.index.searchsorted(dt_col, side='right') - 1
    if idx < 0:
        return None
    return df.iloc[idx][col]


def get_value_at_offset(df: pd.DataFrame, dt_col, col: str, offset_days: int):
    """PIT 工具：获取当前截止时刻 T 之前偏置固定交易日的值（严防数据泄露）"""
    if df.empty:
        return None
    pos = df.index.searchsorted(dt_col, side='right') - 1
    if pos < 0 or pos - offset_days < 0:
        return None
    return df.iloc[pos - offset_days][col]


def load_all_assets_parallel(pipeline_context: dict) -> dict:
    """多线程并发加载全市场资产数据流"""
    data_bus = pipeline_context.get('data_bus')
    if not data_bus or not hasattr(data_bus, 'manager'):
        raise ValueError("Context data_bus initialization failed.")
        
    data_manager = data_bus.manager
    assets = pipeline_context.get('assets', [])
    config = pipeline_context.get('config', {})
    
    start_year = config.get('data_start_year', DEFAULT_START_YEAR)
    start_date = f"{start_year}-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    max_workers = config.get('data_load_workers', DEFAULT_MAX_WORKERS)
    progress_interval = config.get('data_load_progress_interval', DEFAULT_PROGRESS_INTERVAL)
    
    logger.info(f"Loading historical OHLCV data for {len(assets)} assets with {max_workers} threads...")
    
    asset_ohlcv = {}
    failed_assets = []
    
    def load_one(asset):
        try:
            df = _load_asset_data(data_manager, asset, start_date, end_date)
            return asset, df
        except Exception as e:
            logger.warning(f"Failed to load {asset}: {e}")
            return asset, None
            
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_asset = {executor.submit(load_one, asset): asset for asset in assets}
        completed = 0
        total = len(assets)
        start_time = time.time()
        
        for future in as_completed(future_to_asset):
            asset, df = future.result()
            if df is not None and not df.empty:
                asset_ohlcv[asset] = df
            else:
                failed_assets.append(asset)
                
            completed += 1
            if completed % progress_interval == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                logger.info(f"Progress: {completed}/{total} loaded ({rate:.1f} assets/sec), "
                            f"Success: {len(asset_ohlcv)}, Failed: {len(failed_assets)}")
                            
    logger.info(f"Finished concurrent loading. Success: {len(asset_ohlcv)}/{total}")
    if failed_assets:
        logger.warning(f"Failed assets snapshot (first 10): {failed_assets[:10]}")
        
    return asset_ohlcv