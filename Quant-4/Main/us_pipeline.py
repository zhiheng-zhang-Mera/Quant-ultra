import logging
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from typing import Optional

# [调整4] 引入指数退避重试，防止被云服务商封禁或抛出 429
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
except ImportError:
    # 兼容处理：建议环境中 pip install tenacity
    logging.warning("Missing 'tenacity' library. Recommended for production retry mechanisms.")
    def retry(*args, **kwargs):
        def decorator(func): return func
        return decorator
    stop_after_attempt = lambda x: None
    wait_exponential = lambda multiplier, min, max: None
    retry_if_exception_type = lambda x: None

logger = logging.getLogger("USPipeline")

# [调整4] 使用 tenacity 包装公网请求，遭遇限流时自动指数退避重试 (最多尝试 5 次)
@retry(
    stop=stop_after_attempt(5), 
    wait=wait_exponential(multiplier=1.5, min=2, max=20),
    reraise=True
)
def _fetch_yf_history_with_retry(yf_obj, tk: str, start: str, end: str):
    logger.info("[NETWORK] yfinance history request: %s (%s -> %s)", tk, start, end)
    raw = yf_obj.Ticker(tk).history(start=start, end=end)
    if raw.empty:
        raise ValueError(f"yfinance 返回空 DataFrame，可能是请求被拦截或标的错误: {tk}")
    return raw

def fetch_us_historical(symbol: str, start_date: str, end_date: str, cache_dir: Path, offline_debug: bool, has_yf: bool, yf_obj, ak_obj) -> Optional[pd.DataFrame]:
    c_path = cache_dir / f"us_{symbol}_history.parquet"
    if c_path.exists():
        try:
            full = pd.read_parquet(c_path)
            full["date"] = pd.to_datetime(full["date"])
            mask = (full["date"] >= pd.to_datetime(start_date)) & (full["date"] <= pd.to_datetime(end_date))
            df_res = full.loc[mask].copy()
            logger.info("[OP] Slice US Offline Cache Store | [RESULT] Yields pure historically true price traces")
            return df_res
        except Exception:
            c_path.unlink(missing_ok=True)
            
    if offline_debug:
        logger.critical(f"[OP] Evaluate Network Strategy | [RESULT] Halt! Missing physical local data asset for {symbol}")
        raise RuntimeError(f"[A-5 Breach] Sandbox block triggered for {symbol}.")
        
    df = None
    tk = symbol.replace(".US", "").replace(".us", "")
    if tk in [".INX", "SPX"]:
        tk = "^GSPC"
        
    if has_yf and yf_obj:
        try:
            # 采用包含防封禁重试的安全网络请求函数
            current_end = datetime.now().strftime("%Y-%m-%d")
            raw = _fetch_yf_history_with_retry(yf_obj, tk, start="2005-01-01", end=current_end)
            
            df = raw.reset_index().rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
            df["amount"] = df["volume"] * df["close"]
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
        except Exception as e:
            logger.warning(f"[OP] Request Premium Channel Failed | [RESULT] yfinance 彻底失败，转入灾备镜像 | {e}")
            
    if (df is None or df.empty) and ak_obj:
        try:
            if tk == "^GSPC":
                df_ak = ak_obj.index_us_stock_sina(symbol=".INX")
                if df_ak is not None and not df_ak.empty:
                    df = df_ak.copy().rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
                    df["amount"] = df["volume"] * df["close"]
                    df["date"] = pd.to_datetime(df["date"])
                    df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
        except Exception as e:
            logger.error(f"AkShare backup link down: {e}")
            
    if df is None or df.empty:
        raise RuntimeError(f"Halt: Failure to secure verified market vector for US target: {symbol}")
        
    df.sort_values("date", ascending=True, inplace=True)
    df.to_parquet(c_path, index=False)
    
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    return df.loc[mask].copy()

# [调整1] 拆除美股2026年日历上限炸弹
def fetch_us_trading_calendar(cache_dir: Path, offline_debug: bool, has_yf: bool, yf_obj, ak_obj, start_year: int = 2010, end_year: int = None) -> pd.DatetimeIndex:
    if end_year is None:
        end_year = datetime.now().year + 1  # 动态往后推延1年，确保跨年可用
        
    c_path = cache_dir / f"us_cal_{start_year}_{end_year}.parquet"
    if c_path.exists() and (datetime.now() - datetime.fromtimestamp(c_path.stat().st_mtime)).days < 7:
        try:
            cached = pd.read_parquet(c_path)
            if cached is not None and "date" in cached and not cached.empty:
                return pd.DatetimeIndex(cached["date"])
        except Exception as exc:
            logger.warning("Unreadable US calendar cache %s (%s); removing for refetch", c_path, exc)
            try:
                c_path.unlink(missing_ok=True)
            except OSError:
                pass
        
    if offline_debug:
        raise RuntimeError("Halt: Missing physical US market calendar.")
        
    if has_yf and yf_obj:
        try:
            df_us = _fetch_yf_history_with_retry(yf_obj, "^GSPC", start=f"{start_year}-01-01", end=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
            if not df_us.empty:
                d = df_us.index.tz_localize(None).sort_values().tolist()
                pd.DataFrame({"date": d}).to_parquet(c_path, index=False)
                return pd.DatetimeIndex(d)
        except Exception as e:
            logger.warning(f"Calendar fetching failure logic: {e}")
            pass
            
    raise RuntimeError("US trading track dark. Critical safety trip activated.")
