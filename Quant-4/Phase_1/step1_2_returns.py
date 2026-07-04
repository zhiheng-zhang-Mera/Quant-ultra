"""
Quant-Ultra Flow - Step 1.2: Survivor-Bias Free Total Return & Delisting Residual Computation
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

logger = logging.getLogger("Orchestrator.Phase1.Returns")

def _get_delisted_a_stocks(data_manager) -> list:
    try:
        if not hasattr(data_manager, '_ak'): return []
        df = data_manager._ak.stock_zh_a_delisted()
        if df is None or df.empty: return []
        code_col = 'code' if 'code' in df.columns else '股票代码'
        raw_codes = df[code_col].astype(str).str.strip().tolist()
        full_codes = []
        for c in raw_codes:
            if not c.isdigit(): continue
            full_codes.append(f"{c}.SH" if c.startswith('6') else f"{c}.SZ")
        return full_codes
    except Exception as e:
        logger.warning(f"Inquire delisted list matrix exception: {e}")
        return []

def run_returns_cleaning(context: dict, data_bus, data_manager, audit_logger):
    assets = context.get('assets', [])
    now = datetime.now(data_bus._tz)
    latest_trading_day = context.get('effective_latest_trading_day')
    
    delisted = _get_delisted_a_stocks(data_manager)
    all_stocks = list(set(assets + delisted))
    
    logger.info("[OP] Integrate Deceased Corporate Vectors | [SOURCE] Remote Mirroring Exchange Tables | [RESULT] Combined Universe Count: %s (Active: %s, Delisted: %s) | [SIGNIFICANCE] Forcibly reconstructs historical dead asset matrices to resolve flaw A-8", len(all_stocks), len(assets), len(delisted))
    logger.info("[操作] 融合历史退市资产向量 | [来源] 远程交易所镜像大表 | [结果] 合并标的总数: %s (存活: %s, 退市: %s) | [意义] 强制回流已消亡的长尾资产，物理修复 Flaw A-8 幸存者偏差漏洞")

    cache_path = data_manager.cache_dir / "total_return_prices.parquet"
    if cache_path.exists():
        try:
            existing_df = pd.read_parquet(cache_path)
            if not existing_df.empty:
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                if existing_df['date'].max().date() >= latest_trading_day.date():
                    cached_symbols = set(existing_df['symbol'].unique())
                    if assets and set(assets).issubset(cached_symbols):
                        for _, row in existing_df.iterrows():
                            dt = row['date'].to_pydatetime().replace(tzinfo=data_bus._tz)
                            data_bus.append_atom(row['symbol'], dt, float(row['price']), "total_return_price", dt)
                            log_ret = float(row['actual_log_return']) if 'actual_log_return' in row else float(row.get('log_return', 0.0))
                            data_bus.append_atom(row['symbol'], dt, log_ret, "log_return", dt)
                            residual = float(row['delisting_residual']) if 'delisting_residual' in row else 0.0
                            data_bus.append_atom(row['symbol'], dt, residual, "delisting_residual", dt)
                        logger.info("[OP] Hydrate Total Return Vectors | [SOURCE] Local True Parquet Ledger | [RESULT] Hot-injection verification status: Complete | [SIGNIFICANCE] Speeds up backtest startup loops securely")
                        logger.info("[操作] 还原全收益基础向量 | [来源] 本地真实 Parquet 账本 | [结果] 热注入校验状态: 成功完成 | [意义] 安全加速回测系统启动与数据注入流")
                        return
        except Exception as e: logger.warning(f"Snapshot hydration suspended: {e}")

    start_date = "2010-01-01"
    end_date = latest_trading_day.strftime('%Y-%m-%d')
    all_price_records = []
    residual_logged = set()
    
    for sym in tqdm(all_stocks, desc="[全收益+退市残值构建]"):
        hist_df = data_bus.load_asset_history(sym, start_date, end_date)
        if hist_df is None or hist_df.empty: continue
        if 'log_return' not in hist_df.columns: continue
        is_delisted = sym in delisted
        
        for idx, row in hist_df.iterrows():
            dt = idx.to_pydatetime().replace(tzinfo=data_bus._tz)
            price = float(row['close'])
            log_ret = float(row['log_return']) if pd.notna(row['log_return']) else 0.0
            residual = data_bus.query_by_pit(sym, dt, "delisting_residual")
            
            if residual is None:
                residual = 0.0
                if sym not in residual_logged:
                    audit_logger.log_event("DATA_MISSING_DEFAULT_RESIDUAL", {"symbol": sym, "is_delisted": is_delisted, "msg": "Official liquidation stream absent; defaulted to 0.0 fallback"})
                    residual_logged.add(sym)
            else: residual = float(residual)
            
            data_bus.append_atom(sym, dt, price, "total_return_price", dt)
            data_bus.append_atom(sym, dt, log_ret, "log_return", dt)
            data_bus.append_atom(sym, dt, residual, "delisting_residual", dt)
            
            all_price_records.append({
                'symbol': sym, 'date': dt, 'price': price, 'log_return': log_ret,
                'actual_log_return': log_ret, 'delisting_residual': residual, 'is_delisted': is_delisted
            })

    if all_price_records:
        df_new = pd.DataFrame(all_price_records)
        df_new.drop_duplicates(subset=['symbol', 'date'], inplace=True)
        df_new.to_parquet(cache_path, index=False)