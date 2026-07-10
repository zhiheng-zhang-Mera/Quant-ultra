"""
Quant-Ultra Flow - Step 1.2: Survivor-Bias Free Total Return & Delisting Residual Computation
（生产级优化版：全量退市股数据补全 + 对数实际全收益 + 防日志爆炸）
"""
import logging
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm

logger = logging.getLogger("Orchestrator.Step1.Returns")

def _get_delisted_a_stocks(data_manager):
    """
    获取 A 股历史上全部退市股票代码（含后缀）
    优先使用 akshare，若不可用则返回空列表
    """
    try:
        if not hasattr(data_manager, '_ak'):
            logger.debug("AkShare 未加载，无法获取退市列表")
            return []
        df = data_manager._ak.stock_zh_a_delisted()
        if df is None or df.empty:
            return []
        # 兼容列名
        code_col = 'code' if 'code' in df.columns else '股票代码'
        raw_codes = df[code_col].astype(str).str.strip().tolist()
        # 补全市场后缀
        full_codes = []
        for c in raw_codes:
            if not c.isdigit():
                continue
            if c.startswith('6'):
                full_codes.append(f"{c}.SH")
            else:
                full_codes.append(f"{c}.SZ")
        logger.info(f"📋 获取到 {len(full_codes)} 只历史退市 A 股代码")
        return full_codes
    except Exception as e:
        logger.warning(f"获取退市股票列表失败: {e}")
        return []

def run_returns_cleaning(context: dict, data_bus, data_manager, audit_logger):
    """
    全收益价格、对数实际收益率清洗与历史退市股残值流式治理 (Flow-Pro 1.2)
    新增：主动拉取退市股列表，补全其历史日线，彻底消除生存者偏差
    """
    # 当前可交易资产池
    assets = context.get('assets', [])
    now = datetime.now(data_bus._tz)
    latest_trading_day = context.get('effective_latest_trading_day')
    
    # ---------- 获取全部股票（当前池 + 历史退市股） ----------
    delisted = _get_delisted_a_stocks(data_manager)
    all_stocks = list(set(assets + delisted))   # 去重合并
    
    # TODO: 根据 Flow-Pro 1.2 双市场规范，未来若扩展美股标池至全市场，需在此中继美股退市库（如通过 YFinance/Polygon 接口）

    logger.info(f"🔄 全量标的总数: {len(all_stocks)} (其中退市股 {len(delisted)} 只)")

    cache_path = data_manager.cache_dir / "total_return_prices.parquet"
    
    # ---------- 缓存加载路径 ----------
    if cache_path.exists():
        try:
            existing_df = pd.read_parquet(cache_path)
            if not existing_df.empty:
                existing_df['date'] = pd.to_datetime(existing_df['date'])
                # 检查缓存是否覆盖最新交易日
                if existing_df['date'].max().date() >= latest_trading_day.date():
                    # 检查缓存是否包含所有标的（至少覆盖 assets）
                    cached_symbols = set(existing_df['symbol'].unique())
                    if assets and set(assets).issubset(cached_symbols):
                        # 恢复缓存到总线
                        for _, row in existing_df.iterrows():
                            dt = row['date'].to_pydatetime().replace(tzinfo=data_bus._tz)
                            data_bus.append_atom(row['symbol'], dt, float(row['price']), "total_return_price", dt)
                            log_ret = float(row['actual_log_return']) if 'actual_log_return' in row else float(row.get('log_return', 0.0))
                            data_bus.append_atom(row['symbol'], dt, log_ret, "log_return", dt)
                            residual = float(row['delisting_residual']) if 'delisting_residual' in row else 0.0
                            data_bus.append_atom(row['symbol'], dt, residual, "delisting_residual", dt)
                        logger.info("📈 全收益及退市残值数据从缓存热注入成功")
                        return
                    else:
                        logger.info("缓存标的不足，将重新计算")
                else:
                    logger.info("缓存过期，重新计算")
        except Exception as e:
            logger.warning(f"本地快照载入失败，进入重新解算流: {e}")

    # ---------- 全量计算路径 ----------
    # 为退市股回溯更长时间（2010-01-01），活跃股可从 400 天前开始，但统一从 2010 开始保证一致性
    start_date = "2010-01-01"   # 足够覆盖大部分退市股
    end_date = latest_trading_day.strftime('%Y-%m-%d')
    
    all_price_records = []
    # 用于记录已发送过残值缺失日志的资产（防爆炸）
    residual_logged = set()
    
    for sym in tqdm(all_stocks, desc="[全收益+退市残值构建]", unit="只"):
        # 拉取完整历史（若已有缓存则增量）
        hist_df = data_bus.load_asset_history(sym, start_date, end_date)
        if hist_df is None or hist_df.empty:
            logger.debug(f"标的 {sym} 无历史数据（可能退市且数据不可得）")
            continue
        
        if 'log_return' not in hist_df.columns:
            logger.warning(f"标的 {sym} 历史数据缺少 'log_return' 列，跳过")
            continue
        
        # 判断该标的是否为退市股（便于日志）
        is_delisted = sym in delisted
        
        for idx, row in hist_df.iterrows():
            dt = idx.to_pydatetime().replace(tzinfo=data_bus._tz)
            price = float(row['close'])
            log_ret = float(row['log_return']) if pd.notna(row['log_return']) else 0.0
            
            # 流式查询退市残值（若已存入则使用，否则默认0.0）
            residual = data_bus.query_by_pit(sym, dt, "delisting_residual")
            if residual is None:
                residual = 0.0
                # 每个资产只记录一次缺失日志（防止日志爆炸）
                if sym not in residual_logged:
                    audit_logger.log_event("DATA_MISSING_DEFAULT_RESIDUAL", {
                        "symbol": sym,
                        "is_delisted": is_delisted,
                        "msg": "交易所官方残值公告采集缺失，触发 0.0 刚性降级"
                    })
                    residual_logged.add(sym)
            else:
                residual = float(residual)
            
            # 原子化写入总线状态空间
            data_bus.append_atom(sym, dt, price, "total_return_price", dt)
            data_bus.append_atom(sym, dt, log_ret, "log_return", dt)
            data_bus.append_atom(sym, dt, residual, "delisting_residual", dt)
            
            all_price_records.append({
                'symbol': sym,
                'date': dt,
                'price': price,
                'log_return': log_ret,
                'actual_log_return': log_ret,   # 兼容旧缓存
                'delisting_residual': residual,
                'is_delisted': is_delisted       # 可选，便于追踪
            })
        
        # 记录该资产对齐完成（每资产一条，不爆炸）
        audit_logger.log_event("TOTAL_RETURN_ALIGN", {
            "symbol": sym,
            "is_delisted": is_delisted,
            "msg": f"全收益时序及退市残值资产包对齐完成，历史观测深度: {len(hist_df)} 交易日"
        })

    # ---------- 持久化本地中间缓存 ----------
    if all_price_records:
        df_new = pd.DataFrame(all_price_records)
        df_new.drop_duplicates(subset=['symbol', 'date'], inplace=True)
        df_new.to_parquet(cache_path, index=False)
        logger.info(f"💾 Step 1.2 成功，包含 {len(df_new)} 条记录（覆盖 {len(all_stocks)} 只股票）已同步至持久化缓存")
    else:
        logger.warning("未生成任何有效价格矩阵，请检查基础数据源配置")