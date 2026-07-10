"""
Quant-Ultra Flow - Step 1.1: Asset Screening & AUM Capacity Estimation (Dual-Market Edition)
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from step1.config import CONFIG
from step1.concurrency import AdaptiveConcurrencyLimiter, HAS_PSUTIL

logger = logging.getLogger("Orchestrator.Step1.Screening")

def _infer_board_type(symbol: str) -> str:
    if symbol.endswith(".US"): return "美股成分股"
    if symbol.startswith('688'): return '科创板'
    if symbol.startswith(('300', '301')): return '创业板'
    return '主板'

def _worker_core(symbol: str, start_date: str, end_date: str, data_bus, now, limiter) -> dict:
    if limiter and not limiter.acquire(): return None
    success = False
    try:
        hist_df = data_bus.load_asset_history(symbol, start_date, end_date)
        if hist_df is None or hist_df.empty: return None

        adv_series = hist_df['amount'].tail(CONFIG.get("ADV_WINDOW", 20))
        adv = adv_series.mean() if len(adv_series) >= CONFIG.get("ADV_WINDOW", 20) * 0.5 else 0
        if adv == 0: return None

        board = _infer_board_type(symbol)
        
        # 智能匹配历史第一根K线推断真实上市日期，对抗新股次新股数据缺失
        first_date = hist_df.index.min()
        list_date = first_date.to_pydatetime().replace(tzinfo=data_bus._tz) if hasattr(first_date, 'to_pydatetime') else now - timedelta(days=1000)

        data_bus.append_atom(symbol, now, adv, "adv", now - timedelta(days=1))
        data_bus.append_atom(symbol, now, list_date, "listing_date", now)
        data_bus.append_atom(symbol, now, board, "board", now)

        success = True
        return {'symbol': symbol, 'hist_df': hist_df, 'adv': adv, 'list_date': list_date, 'board': board}
    except Exception as e:
        logger.debug(f"评估资产流动性失败 {symbol}: {e}")
        return None
    finally:
        if limiter: limiter.release(success)

def run_screening(context: dict, data_bus, data_manager):
    logger.info("启动跨市场双标池精细化流动性筛选与容量前置测算...")
    now = datetime.now(data_bus._tz)
    
    trading_days = context.get('trading_days_dt', [])
    past_days = [d for d in trading_days if d <= now]
    if not past_days: raise RuntimeError("交易时序轴发生断裂")
    latest_trading_day = past_days[-1]
    context['effective_latest_trading_day'] = latest_trading_day

    screening_cache = data_manager.cache_dir / "screening_results.parquet"
    if screening_cache.exists():
        try:
            df_cache = pd.read_parquet(screening_cache)
            if 'cache_date' in df_cache.columns and pd.to_datetime(df_cache['cache_date'].iloc[0]).date() == latest_trading_day.date():
                logger.info("🎯 命中全局双市场初筛本地持仓缓存，加速越过数据网络层")
                context['assets'] = df_cache['symbol'].tolist()
                context['adv_data'] = {row['symbol']: row['adv'] for _, row in df_cache.iterrows()}
                return
        except Exception as e:
            logger.warning(f"读取筛选快照失败: {e}")

    # 解锁代码封锁：支持 A股纯数字 与 美股点状字母（如 AAPL.US）双轨并发初筛
    symbols = data_bus.get_universe(refresh=True)
    symbols = [s for s in symbols if s.split('.')[0].isdigit() or s.endswith('.US')]
    
    end_date = now.strftime('%Y-%m-%d')
    start_date = (now - timedelta(days=400)).strftime('%Y-%m-%d')
    
    limiter = AdaptiveConcurrencyLimiter() if HAS_PSUTIL else None
    max_workers = CONFIG.get("ADAPTIVE_MAX_WORKERS", 4) if HAS_PSUTIL else CONFIG.get("DOWNLOAD_WORKERS", 8)
    
    raw_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_worker_core, sym, start_date, end_date, data_bus, now, limiter): sym 
            for sym in symbols
        }
        for future in tqdm(as_completed(future_map), total=len(symbols), desc="[分布式流动性初筛]", unit="只"):
            res = future.result()
            if res:
                raw_results.append(res)
                context.setdefault('asset_histories', {})[res['symbol']] = res['hist_df']

    if limiter: limiter.stop()

    filtered_list = []
    ipo_safety_days = context['config'].get("ipo_safety_days", 20)
    min_adv_threshold = context['config'].get("min_adv_threshold", 1e7)

    for item in raw_results:
        days_listed = (now - item['list_date']).days if item['list_date'] else 999
        # 次新股安全垫硬约束：未满 M0 天直接进行物理剥离与拦截 (Flow-Pro 1.1)
        if item['adv'] >= min_adv_threshold and days_listed >= ipo_safety_days:
            filtered_list.append(item)

    context['assets'] = [x['symbol'] for x in filtered_list]
    context['adv_data'] = {x['symbol']: x['adv'] for x in filtered_list}

    # ====================================================
    # 容量前置估算模型构建 (Flow-Pro 1.1 刚性落地)
    # ====================================================
    if filtered_list:
        max_part_rate = context['config'].get("max_participation_rate", 0.05)
        expected_turnover = context['config'].get("expected_turnover", 0.05)
        max_single_weight = context['config'].get("max_single_stock_weight", 0.05)
        
        # 依据单票最大换手与微观流动性冲击上限约束，反推最大理论 AUM 上限基准
        capacities = [
            (x['adv'] * max_part_rate) / (expected_turnover * max_single_weight) 
            for x in filtered_list
        ]
        # 审慎安全原则取资产池中最窄流动性瓶颈作为基准规模上限提示
        theoretical_aum_limit = min(capacities) * len(filtered_list) * 0.15
        context['theoretical_aum_limit_base'] = theoretical_aum_limit
        logger.info(f"📊 [容量前置估算] 基于当前双市场成分池反推个人资产安全规模 (AUM) 上限基准为: {theoretical_aum_limit:,.2f} 元")

        df_out = pd.DataFrame(filtered_list).drop(columns=['hist_df'], errors='ignore')
        df_out['cache_date'] = latest_trading_day.strftime('%Y-%m-%d')
        df_out.to_parquet(screening_cache, index=False)