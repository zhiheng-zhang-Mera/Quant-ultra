import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Optional

logger = logging.getLogger("USPipeline")

def fetch_us_historical(symbol: str, start_date: str, end_date: str, cache_dir: Path, offline_debug: bool, has_yf: bool, yf_obj, ak_obj) -> Optional[pd.DataFrame]:
    c_path = cache_dir / f"us_{symbol}_history.parquet"
    if c_path.exists():
        try:
            full = pd.read_parquet(c_path)
            full["date"] = pd.to_datetime(full["date"])
            mask = (full["date"] >= pd.to_datetime(start_date)) & (full["date"] <= pd.to_datetime(end_date))
            df_res = full.loc[mask].copy()
            logger.info("[OP] Slice US Offline Cache Store | [SOURCE] Local Verified Parquet Block | [RESULT] Sub-matrix row count: %s | [SIGNIFICANCE] Yields pure historically true price traces", len(df_res))
            logger.info("[操作] 切片美股离线缓存区 | [来源] 本地经验证的Parquet块 | [结果] 子矩阵提取行数: %s | [意义] 交付纯净的、具备历史真实性的价格轨迹")
            return df_res
        except Exception: c_path.unlink(missing_ok=True)
        
    if offline_debug:
        logger.critical("[OP] Evaluate Network Strategy | [SOURCE] Sandboxed Environment Controller | [RESULT] Halt! Missing physical local data asset for %s | [SIGNIFICANCE] Strict prevention of random-walk vector injection during network blackouts", symbol)
        logger.critical("[操作] 评估网络访问策略 | [来源] 沙箱环境控制器 | [结果] 阻断！缺少 %s 的本地物理数据资产 | [意义] 严格禁止在断网时注入任何随机游走随机伪向量")
        raise RuntimeError(f"[A-5 Breach] Sandbox block triggered for {symbol}.")
        
    df = None
    tk = symbol.replace(".US", "").replace(".us", "")
    if tk in [".INX", "SPX"]: tk = "^GSPC"
    
    if has_yf and yf_obj:
        try:
            logger.info("[OP] Execute Public API Request | [SOURCE] yfinance International Node | [RESULT] Fetching historical price for symbol: %s | [SIGNIFICANCE] Targets premium untainted market index data directly", tk)
            logger.info("[操作] 执行公网API请求 | [来源] yfinance国际直连节点 | [结果] 正在拉取历史价格，标的代码: %s | [意义] 直接定位未受污染的高保真大盘核心交易指数")
            raw = yf_obj.Ticker(tk).history(start="2005-01-01", end=datetime.now().strftime("%Y-%m-%d"))
            if not raw.empty:
                df = raw.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
                df["amount"] = df["volume"] * df["close"]
                df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
                df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
        except Exception as e: 
            logger.warning("[OP] Request Premium Channel Failed | [SOURCE] yfinance Cloud Interface | [RESULT] Request timeout or network unreachable | [SIGNIFICANCE] Triggers secondary mirroring route alignment", exc_info=True)
            logger.warning("[操作] 请求高规通道失败 | [来源] yfinance云端接口 | [结果] 请求超时或网络不可达 | [意义] 触发二级镜像备用路由机制进行灾备对齐")

    if (df is None or df.empty) and ak_obj:
        try:
            if tk == "^GSPC":
                df_ak = ak_obj.index_us_stock_sina(symbol=".INX")
                if df_ak is not None and not df_ak.empty:
                    df = df_ak.copy().rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
                    df["amount"] = df["volume"] * df["close"]
                    df["date"] = pd.to_datetime(df["date"])
                    df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
        except Exception as e: logger.error(f"AkShare backup link down: {e}")
        
    if df is None or df.empty:
        raise RuntimeError(f"Halt: Failure to secure verified market vector for US target: {symbol}")
        
    df.sort_values("date", ascending=True, inplace=True)
    df.to_parquet(c_path, index=False)
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    return df.loc[mask].copy()

def fetch_us_trading_calendar(cache_dir: Path, offline_debug: bool, has_yf: bool, yf_obj, ak_obj, start_year: int = 2010, end_year: int = 2026) -> pd.DatetimeIndex:
    c_path = cache_dir / f"us_cal_{start_year}_{end_year}.parquet"
    if c_path.exists() and (datetime.now() - datetime.fromtimestamp(c_path.stat().st_mtime)).days < 7:
        return pd.DatetimeIndex(pd.read_parquet(c_path)["date"])
    if offline_debug: raise RuntimeError("Halt: Missing physical US market calendar.")
    if has_yf and yf_obj:
        try:
            df_us = yf_obj.Ticker("^GSPC").history(start=f"{start_year}-01-01", end=f"{end_year}-12-31")
            if not df_us.empty:
                d = df_us.index.tz_localize(None).sort_values().tolist()
                pd.DataFrame({"date": d}).to_parquet(c_path, index=False)
                return pd.DatetimeIndex(d)
        except: pass
    raise RuntimeError("US trading track dark. Critical safety trip activated.")