"""
Quant-Ultra Flow - Step 1.3: Trading Status & Price Boundary Mapping (Multi-Market Mechanism)
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

logger = logging.getLogger("Orchestrator.Phase1.TradingStatus")

def run_status_mapping(context: dict, data_bus, data_manager):
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
                logger.info("[OP] Restore Trading Constraint Map | [SOURCE] Local Status Schema File | [RESULT] Limits recovery: Succeeded | [SIGNIFICANCE] Secures cross-sectional boundaries without repeated IO overhead")
                logger.info("[操作] 恢复交易状态约束图谱 | [来源] 本地状态 Schema 文件 | [结果] 涨跌停边界恢复: 成功完成 | [意义] 固定截面交易限制红线，消除重复查询开销")
                return
        except Exception: pass

    st_codes = set()
    for name, _ in data_manager._sources:
        try:
            if name == "akshare":
                df_st = data_manager._ak.stock_zh_a_st_em()
                if df_st is not None and not df_st.empty:
                    st_codes = set(df_st["代码"].astype(str).str.strip().tolist())
                    logger.info("[OP] Poll Risk Registry Channel | [SOURCE] AkShare Engine Primary Track | [RESULT] Ingested %s active ST tokens | [SIGNIFICANCE] High priority validation pass completed", len(st_codes))
                    logger.info("[操作] 轮询风险监控注册通道 | [来源] AkShare 引擎主轨通道 | [结果] 捕获到 %s 只活跃 ST 风险标的 | [意义] 高优先级风控白名单过滤机制生效完成")
                    break
            elif name == "baostock":
                if not data_manager._bs_logged: data_manager._bs.login(); data_manager._bs_logged = True
                rs = data_manager._bs.query_all_stock()
                if rs is not None and rs.error_code == "0":
                    while rs.next():
                        row = rs.get_row_data()
                        if row[3] in ["1", "2", "3", "4"]: st_codes.add(row[0].split(".")[1])
                    if st_codes: break
        except Exception as e:
            logger.warning(f"Priority routing fallback pass error over {name}: {e}")

    all_status = []
    for sym in tqdm(assets, desc="[多套涨跌幅限额联动映射]"):
        try:
            code = sym.split('.')[0]
            is_st = code in st_codes
            board = data_bus.query_by_pit(sym, now, "board") or '主板'
            list_date = data_bus.query_by_pit(sym, now, "listing_date")
            days_listed = (now - list_date).days if list_date else 999
            prev_date = now - timedelta(days=1)
            prev_price = data_bus.query_by_pit(sym, prev_date, "total_return_price") or 100.0

            if board == "美股成分股": limit_ratio = 50.0  # 海外市场自适应弹性防御安全边界
            else: limit_ratio = 0.05 if is_st else (0.20 if board in ["科创板", "创业板"] else 0.10)
            
            mapping = {
                "board": board, "is_st": is_st, "days_listed": days_listed,
                "limit_up": prev_price * (1 + limit_ratio), "limit_down": prev_price * (1 - limit_ratio), "prev_close": prev_price
            }
            data_bus.append_atom(sym, now, mapping, "trading_status_mapping", now)
            record = mapping.copy(); record.update({'symbol': sym, 'date': latest_trading_day})
            all_status.append(record)
        except Exception as e: logger.warning(f"Map analytics error on asset risk margin: {sym}: {e}")

    if all_status:
        try: pd.DataFrame(all_status).to_parquet(cache_path, index=False)
        except Exception as e: logger.warning(f"Synchronize barrier state failed: {e}")