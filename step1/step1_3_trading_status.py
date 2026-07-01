"""
Quant-Ultra Flow - Step 1.3: Trading Status & Price Boundary Mapping (Strict Priority Edition)
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

logger = logging.getLogger("Orchestrator.Step1.TradingStatus")

def run_status_mapping(context: dict, data_bus, data_manager):
    """
    涨跌停安全边界与 ST 状态矩阵映射（严格遵循数据源注册优先级顺序）
    """
    assets = context.get('assets', [])
    now = datetime.now(data_bus._tz)
    latest_trading_day = context.get('effective_latest_trading_day')
    
    cache_path = data_manager.cache_dir / "trading_status.parquet"
    if cache_path.exists():
        try:
            df_cache = pd.read_parquet(cache_path)
            if not df_cache.empty and pd.to_datetime(df_cache['date'].iloc[0]).date() == latest_trading_day.date():
                for _, row in df_cache.iterrows():
                    dt = row['date'].to_pydatetime().replace(tzinfo=data_bus._tz)
                    data_bus.append_atom(row['symbol'], dt, {
                        "board": row['board'], "is_st": row['is_st'], "days_listed": row['days_listed'],
                        "limit_up": row['limit_up'], "limit_down": row['limit_down'], "prev_close": row['prev_close']
                    }, "trading_status_mapping", dt)
                logger.info("🛡️ 成功命中交易状态映射与边界控制快照缓存")
                return
        except Exception:
            pass

    # ========================================================
    # 核心修正：严格按照 data_manager._sources 注册的优先级顺序动态轮询 ST 监控池
    # ========================================================
    st_codes = set()
    
    for name, _ in data_manager._sources:
        try:
            if name == "akshare":
                df_st = data_manager._ak.stock_zh_a_st_em()
                if df_st is not None and not df_st.empty:
                    st_codes = set(df_st["代码"].astype(str).str.strip().tolist())
                    logger.info(f"📡 优先级 [1]: 成功通过 AkShare 批量捕获 ST 监控池，共 {len(st_codes)} 只")
                    break
                    
            elif name == "baostock":
                if not data_manager._bs_logged:
                    data_manager._bs.login()
                    data_manager._bs_logged = True
                rs = data_manager._bs.query_all_stock()
                if rs is not None and rs.error_code == "0":
                    while rs.next():
                        row = rs.get_row_data()
                        if row[3] in ["1", "2", "3", "4"]:
                            st_codes.add(row[0].split(".")[1])
                    if st_codes:
                        logger.info(f"📡 优先级 [2]: 成功通过 BaoStock 降级捕获 ST 监控池，共 {len(st_codes)} 只")
                        break
                        
            elif name == "tushare":
                df_basic = data_manager._ts_pro.stock_basic(exchange='', list_status='L', fields='ts_code,name')
                if df_basic is not None and not df_basic.empty:
                    st_df = df_basic[df_basic['name'].str.contains('ST', na=False)]
                    st_codes = set(st_df['ts_code'].apply(lambda x: x.split('.')[0]).tolist())
                    if st_codes:
                        logger.info(f"📡 优先级 [3]: 成功通过 Tushare 降级筛选 ST 监控池，共 {len(st_codes)} 只")
                        break
                        
            elif name == "efinance":
                df_ef = data_manager._ef.stock.get_realtime_quotes()
                if df_ef is not None and not df_ef.empty:
                    st_df = df_ef[df_ef['股票名称'].str.contains('ST', na=False)]
                    st_codes = set(st_df['股票代码'].astype(str).str.strip().tolist())
                    if st_codes:
                        logger.info(f"📡 优先级 [4]: 成功通过 efinance 降级筛选 ST 监控池，共 {len(st_codes)} 只")
                        break
        except Exception as e:
            logger.warning(f"⚠️ 优先级流转：尝试通过 {name} 获取 ST 池失败或无权限，自动向下轮询: {e}")

    all_status = []
    for sym in tqdm(assets, desc="[生产控制: 涨跌停映射]", unit="只"):
        try:
            code = sym.split('.')[0]
            is_st = code in st_codes
            
            board = data_bus.query_by_pit(sym, now, "board") or '主板'
            list_date = data_bus.query_by_pit(sym, now, "listing_date")
            days_listed = (now - list_date).days if list_date else 999
            
            prev_date = now - timedelta(days=1)
            prev_price = data_bus.query_by_pit(sym, prev_date, "total_return_price") or 100.0

            limit_ratio = 0.05 if is_st else (0.20 if board in ["科创板", "创业板"] else 0.10)
            
            mapping = {
                "board": board, 
                "is_st": is_st, 
                "days_listed": days_listed,
                "limit_up": prev_price * (1 + limit_ratio),
                "limit_down": prev_price * (1 - limit_ratio),
                "prev_close": prev_price
            }
            data_bus.append_atom(sym, now, mapping, "trading_status_mapping", now)
            
            record = mapping.copy()
            record.update({'symbol': sym, 'date': latest_trading_day})
            all_status.append(record)
        except Exception as e:
            logger.warning(f"⚠️ 分析资产风控边界异常 {sym}: {e}")

    if all_status:
        try:
            pd.DataFrame(all_status).to_parquet(cache_path, index=False)
            logger.info("💾 交易状态控制字典已成功同步至持久化本地缓存")
        except Exception as e:
            logger.warning(f"保存状态控制缓存失败: {e}")