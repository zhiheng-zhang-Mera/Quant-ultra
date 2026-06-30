"""
Quant-Ultra Flow - Step 1.1: Asset Screening & AUM Capacity Estimation
"""
import logging
# from typing import Optional
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from step1.config import CONFIG
from step1.concurrency import AdaptiveConcurrencyLimiter, HAS_PSUTIL

logger = logging.getLogger("Orchestrator.Step1.Screening")

def _infer_board_type(symbol: str) -> str:
    if symbol.startswith('688'): return '科创板'
    if symbol.startswith(('300', '301')): return '创业板'
    return '主板'

def _worker_core(symbol: str, start_date: str, end_date: str, data_bus, now, limiter) -> Optional[dict]:
    if limiter and not limiter.acquire(): return None
    success = False
    try:
        # 核心修正：彻底改由原子数据总线加载历史 K 线，消灭私网调用
        hist_df = data_bus.load_asset_history(symbol, start_date, end_date)
        if hist_df is None or hist_df.empty:
            return None

        adv_series = hist_df['amount'].tail(CONFIG["ADV_WINDOW"])
        adv = adv_series.mean() if len(adv_series) >= CONFIG["ADV_WINDOW"] * 0.5 else 0
        if adv == 0: return None

        # 由数据总线映射基础属性
        board = _infer_board_type(symbol)
        list_date_str = data_bus.query_by_pit(symbol, now, "list_date") or "2000-01-01"
        list_date = pd.to_datetime(list_date_str).to_pydatetime().replace(tzinfo=data_bus._tz)

        # 写入原子存储
        data_bus.append_atom(symbol, now, adv, "adv", now - timedelta(days=1))
        data_bus.append_atom(symbol, now, list_date, "listing_date", now)
        data_bus.append_atom(symbol, now, board, "board", now)

        success = True
        return {'symbol': symbol, 'hist_df': hist_df, 'adv': adv, 'list_date': list_date, 'board': board}
    except Exception as e:
        logger.debug(f"评估单股资产失败 {symbol}: {e}")
        return None
    finally:
        if limiter: limiter.release(success)

def run_screening(context: dict, data_bus, data_manager):
    logger.info("启动资产池精细化流动性筛选与容量测算...")
    now = datetime.now(data_bus._tz)
    
    trading_days = context.get('trading_days_dt', [])
    past_days = [d for d in trading_days if d <= now]
    if not past_days: raise RuntimeError("交易日历回测时间轴发生断裂")
    latest_trading_day = past_days[-1]
    context['effective_latest_trading_day'] = latest_trading_day

    screening_cache = data_manager.cache_dir / "screening_results.parquet"
    if screening_cache.exists():
        try:
            df_cache = pd.read_parquet(screening_cache)
            if 'cache_date' in df_cache.columns and pd.to_datetime(df_cache['cache_date'].iloc[0]).date() == latest_trading_day.date():
                logger.info("🎯 命中全局初筛本地持仓缓存，加速跳过网络开销")
                context['assets'] = df_cache['symbol'].tolist()
                context['adv_data'] = {row['symbol']: row['adv'] for _, row in df_cache.iterrows()}
                return
        except Exception as e:
            logger.warning(f"读取筛选快照失败: {e}")

    # 获取系统全池代码
    symbols = data_bus.get_universe(refresh=True)
    symbols = [s for s in symbols if s.split('.')[0].isdigit()]
    
    end_date = now.strftime('%Y-%m-%d')
    start_date = (now - timedelta(days=400)).strftime('%Y-%m-%d')
    
    limiter = AdaptiveConcurrencyLimiter() if HAS_PSUTIL else None
    max_workers = CONFIG["ADAPTIVE_MAX_WORKERS"] if HAS_PSUTIL else CONFIG["DOWNLOAD_WORKERS"]
    
    raw_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_worker_core, sym, start_date, end_date, data_bus, now, limiter): sym 
            for sym in symbols
        }
        for future in tqdm(as_completed(future_map), total=len(symbols), desc="[流动性特征构建]", unit="只"):
            res = future.result()
            if res:
                raw_results.append(res)
                context.setdefault('asset_histories', {})[res['symbol']] = res['hist_df']

    if limiter: limiter.stop()

    filtered_list = []
    for item in raw_results:
        days_listed = (now - item['list_date']).days if item['list_date'] else 999
        if item['adv'] >= CONFIG["MIN_ADV_THRESHOLD"] and days_listed >= CONFIG["IPO_SAFETY_DAYS"]:
            filtered_list.append(item)

    context['assets'] = [x['symbol'] for x in filtered_list]
    context['adv_data'] = {x['symbol']: x['adv'] for x in filtered_list}

    # 导出持久化缓存
    if filtered_list:
        df_out = pd.DataFrame(filtered_list).drop(columns=['hist_df'], errors='ignore')
        df_out['cache_date'] = latest_trading_day.strftime('%Y-%m-%d')
        df_out.to_parquet(screening_cache, index=False)