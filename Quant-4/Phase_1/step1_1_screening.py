"""
Quant-Ultra Flow - Step 1.1: Asset Screening & AUM Capacity Estimation (Dual-Market Edition)
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from Phase_1.config import CONFIG
from Phase_1.concurrency import AdaptiveConcurrencyLimiter, HAS_PSUTIL
from Phase_1.sector_rotation import select_sector_universe

logger = logging.getLogger("Orchestrator.Phase1.Screening")

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
    # logger.info("[OP] Deploy Cross-Market Liquidity Filter | [SOURCE] Federated Unified Universe Node | [RESULT] Launching parallel analytics pool | [SIGNIFICANCE] Screens micro-liquidity features and sets up capacity blueprints")
    logger.info("[操作] 部署跨市场流动性筛选器 | [来源] 联邦统一标的池节点 | [结果] 正在启动并行分析计算池 | [意义] 甄别微观流动性特征并构建容量上限蓝图")
    
    now = datetime.now(data_bus._tz)

    trading_days = context.get('trading_days_dt', [])
    past_days = [d for d in trading_days if pd.to_datetime(d) <= now]
    if not past_days: raise RuntimeError("Timeline sequence axis broken. | 交易时序轴发生断裂")
    latest_trading_day = past_days[-1]
    context['effective_latest_trading_day'] = latest_trading_day
    logger.info("[TIMING] Effective latest trading day resolved: %s", latest_trading_day.strftime('%Y-%m-%d'))

    if not context.get("config", {}).get("bounded_universe") and not getattr(data_manager, "offline_debug", False):
        symbols, sector_ranking = select_sector_universe(
            data_manager,
            latest_trading_day.strftime("%Y-%m-%d"),
            top_n=int(context.get("config", {}).get("sector_top_n", CONFIG.get("SECTOR_TOP_N", 3))),
        )
        data_bus.set_universe(symbols)
        context["sector_rotation_active"] = True
        context["selected_sectors"] = sector_ranking["sector"].tolist()
        context["sector_ranking"] = sector_ranking

    screening_cache = data_manager.cache_dir / "screening_results.parquet"
    if screening_cache.exists() and not context.get("sector_rotation_active"):
        try:
            df_cache = pd.read_parquet(screening_cache)
            if 'cache_date' in df_cache.columns and pd.to_datetime(df_cache['cache_date'].iloc[0]).date() == latest_trading_day.date():
                allowed = set(data_bus.get_universe())
                df_cache = df_cache[df_cache['symbol'].isin(allowed)].copy()
                if df_cache.empty:
                    raise ValueError("screening cache has no rows for the requested bounded universe")
                context['assets'] = df_cache['symbol'].tolist()
                context['adv_data'] = {row['symbol']: row['adv'] for _, row in df_cache.iterrows()}
                # logger.info("[OP] Trigger Local Checkpoint Recovery | [SOURCE] Local Parquet Cache Database | [RESULT] Hydrated context for %s symbols | [SIGNIFICANCE] Bypasses heavy historical computation and IO traps completely", len(context['assets']))
                logger.info("[操作] 触发本地检查点恢复 | [来源] 本地 Parquet 缓存数据库 | [结果] 成功还原 %s 只标的的上下文 | [意义] 完全绕过沉重的历史重算与网络 IO 陷阱", len(context['assets']))
                return
        except Exception as e: 
            logger.warning(f"Failed to ingest screen snapshot: {e}")

    symbols = data_bus.get_universe()
    symbols = [s for s in symbols if s.split('.')[0].isdigit() or s.endswith('.US')]
    # logger.info("[DATA] Total raw symbols from universe: %s, after filtering (A-share + US) usable: %s", len(data_bus.get_universe()), len(symbols))
    logger.info("[DATA] 原始标的池总数: %s, 经双市场过滤后可用的标的数: %s", len(data_bus.get_universe()), len(symbols))
    
    end_date = now.strftime('%Y-%m-%d')
    start_date = (now - timedelta(days=400)).strftime('%Y-%m-%d')
    logger.info("[RANGE] Data pull window: %s to %s", start_date, end_date)

    # Fill parquet caches in hardware-sized batches before analytics workers run.
    # Existing DataBus calls then become local reads, preserving the engine API.
    batch_fetch = getattr(data_manager, "fetch_historical_batch", None)
    if callable(batch_fetch) and not getattr(data_manager, "offline_debug", False):
        logger.info("[DOWNLOAD] Warming %s history caches in batches", len(symbols))
        batch_fetch(symbols, start_date, end_date)
    
    limiter = AdaptiveConcurrencyLimiter() if HAS_PSUTIL else None
    configured_workers = context.get("config", {}).get("download_workers")
    runtime_plan = getattr(data_manager, "download_plan", None)
    max_workers = configured_workers or (runtime_plan.workers if runtime_plan else (CONFIG.get("ADAPTIVE_MAX_WORKERS", 4) if HAS_PSUTIL else CONFIG.get("DOWNLOAD_WORKERS", 8)))
    logger.info("[PARALLEL] Using max_workers=%s, adaptive=%s", max_workers, HAS_PSUTIL)
    
    raw_results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {executor.submit(_worker_core, sym, start_date, end_date, data_bus, now, limiter): sym for sym in symbols}
        logger.info("[TASK] Submitted %s screening tasks to thread pool", len(future_map))
        for future in tqdm(as_completed(future_map), total=len(symbols), desc="[分布式流动性初筛]"):
            res = future.result()
            if res:
                raw_results.append(res)
                context.setdefault('asset_histories', {})[res['symbol']] = res['hist_df']

    if limiter: limiter.stop()
    # logger.info("[RESULT] Raw screening completed, %s assets passed the basic data availability check", len(raw_results))
    logger.info("[RESULT] 原始筛选完成, %s 只标的通过了基础数据可用性检查", len(raw_results))

    filtered_list = []
    ipo_safety_days = context['config'].get("ipo_safety_days", 20)
    min_adv_threshold = context['config'].get("min_adv_threshold", 1e7)

    ipo_rejected = 0
    adv_rejected = 0
    for item in raw_results:
        days_listed = (now - item['list_date']).days if item['list_date'] else 999
        if item['adv'] >= min_adv_threshold and days_listed >= ipo_safety_days:
            filtered_list.append(item)
        else:
            if item['adv'] < min_adv_threshold:
                adv_rejected += 1
            if days_listed < ipo_safety_days:
                ipo_rejected += 1

    context['assets'] = [x['symbol'] for x in filtered_list]
    context['adv_data'] = {x['symbol']: x['adv'] for x in filtered_list}
    # logger.info("[FILTER] Final filtered assets: %s (rejected: ADV insufficient %s, IPO not mature %s)", len(filtered_list), adv_rejected, ipo_rejected)
    logger.info("[FILTER] 最终筛选后资产数: %s (流动性不足剔除: %s, 次新股保护剔除: %s)", len(filtered_list), adv_rejected, ipo_rejected)

    # ====================================================
    # 容量前置估算模型构建 (Flow-Pro 1.1 刚性落地)
    # ====================================================
    if filtered_list:
        max_part_rate = context['config'].get("max_participation_rate", 0.05)
        expected_turnover = context['config'].get("expected_turnover", 0.05)
        max_single_weight = context['config'].get("max_single_stock_weight", 0.05)
        
        capacities = [(x['adv'] * max_part_rate) / (expected_turnover * max_single_weight) for x in filtered_list]
        theoretical_aum_limit = min(capacities) * len(filtered_list) * 0.15
        context['theoretical_aum_limit_base'] = theoretical_aum_limit
        
        #logger.info("[OP] Inverse Prudent Capacity Base | [SOURCE] Micro Liquidity Friction Matrix | [RESULT] AUM Limit Constant: %s CNY | [SIGNIFICANCE] Pinpoints investment scale choke points under strict turnover boundaries", theoretical_aum_limit)
        logger.info("[操作] 反推审慎容量基准 | [来源] 微观流动性摩擦矩阵 | [结果] 个人资产安全规模上限: %s 元 | [意义] 在严格的换手率边界下，精准锁定位资产池中窄通道流动性瓶颈上限", theoretical_aum_limit)
        # 额外输出容量分布统计
        logger.info("[DIST] Capacity distribution: min=%.2f, median=%.2f, max=%.2f (CNY)", min(capacities), pd.Series(capacities).median(), max(capacities))

        df_out = pd.DataFrame(filtered_list).drop(columns=['hist_df'], errors='ignore')
        df_out['cache_date'] = latest_trading_day.strftime('%Y-%m-%d')
        df_out.to_parquet(screening_cache, index=False)
        logger.info("[CACHE] Screening results cached to %s (records=%s)", screening_cache, len(df_out))
    else:
        # logger.warning("[WARN] No assets passed screening, capacity estimation skipped")
        logger.warning("[警告] 没有标的通过筛选，容量估算被跳过")
