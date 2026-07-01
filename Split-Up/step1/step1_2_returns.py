"""
Quant-Ultra Flow - Step 1.2: Total Return Computation
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

logger = logging.getLogger("Orchestrator.Step1.Returns")

def run_returns_cleaning(context: dict, data_bus, data_manager, audit_logger):
    assets = context.get('assets', [])
    now = datetime.now(data_bus._tz)
    latest_trading_day = context.get('effective_latest_trading_day')
    
    cache_path = data_manager.cache_dir / "total_return_prices.parquet"
    existing_df = None
    if cache_path.exists():
        try:
            existing_df = pd.read_parquet(cache_path)
            if not existing_df.empty:
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                if existing_df['date'].max().date() >= latest_trading_day.date():
                    for _, row in existing_df.iterrows():
                        dt = row['date'].to_pydatetime().replace(tzinfo=data_bus._tz)
                        data_bus.append_atom(row['symbol'], dt, float(row['price']), "total_return_price", dt)
                    logger.info("📈 全收益基础价格序列缓存完整且处于鲜活状态")
                    return
        except Exception as e:
            logger.warning(f"价格序列矩阵载入失败: {e}")

    start_date = (now - timedelta(days=400)).strftime('%Y-%m-%d')
    end_date = latest_trading_day.strftime('%Y-%m-%d')
    
    all_price_records = []
    for sym in tqdm(assets, desc="[特征对齐: 全收益价格]", unit="只"):
        hist_df = data_bus.load_asset_history(sym, start_date, end_date)
        if hist_df is None or hist_df.empty: continue
        for idx, row in hist_df.iterrows():
            dt = idx.to_pydatetime().replace(tzinfo=data_bus._tz)
            price = float(row['close'])
            data_bus.append_atom(sym, dt, price, "total_return_price", dt)
            all_price_records.append({'symbol': sym, 'date': dt, 'price': price})
            
        audit_logger.log_event("TOTAL_RETURN_ALIGN", {"symbol": sym, "msg": "使用调整后收盘价替代复权价格流"})

    if all_price_records:
        df_new = pd.DataFrame(all_price_records)
        df_new.drop_duplicates(subset=['symbol', 'date'], inplace=True)
        df_new.to_parquet(cache_path, index=False)