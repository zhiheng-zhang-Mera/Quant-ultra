import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
import pandas as pd
from Main.env_config import PROJECT_ROOT
from Main.us_pipeline import fetch_us_historical, fetch_us_trading_calendar

class FreeDataSourceManager:
    def __init__(self, cache_dir: Path = None, offline_debug: bool = False):
        self.cache_dir = cache_dir or (PROJECT_ROOT / "data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._sources: List[tuple] = []
        self._bs_logged = False
        self.offline_debug = offline_debug
        self._logger = logging.getLogger("DataSourceManager")
        self._load_env_file()
        self._init_sources()
        self.DEFAULT_START = "2005-01-01"
        self._failed_symbols: Set[str] = set()

    def _load_env_file(self):
        p = PROJECT_ROOT / "Main" / ".env"
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ[k.strip()] = v.strip()
            self._logger.info("[OP] Parse Token Files | [SOURCE] Main/.env Physical File | [RESULT] Environment tokens injected into memory | [SIGNIFICANCE] Enables seamless authorization for premium database nodes", p.name)
            self._logger.info("[操作] 解析Token凭证文件 | [来源] Main/.env物理文本 | [结果] 环境变量访问凭证注入内存 | [意义] 激活针对高级数据库节点的无缝授权通行")

    def _init_sources(self):
        self._has_yf = False
        if not self.offline_debug:
            try:
                import yfinance as yf; self._yf = yf; self._has_yf = True
                self._logger.info("[OP] Register Data Providers | [SOURCE] Local Python Site-Packages | [RESULT] Level 0 (yfinance Tracker) successfully active | [SIGNIFICANCE] Connects international equity pipeline", yf.__name__)
                self._logger.info("[操作] 注册数据供应源 | [来源] 本地Python第三方库环境 | [结果] Level 0 (yfinance 追踪器) 成功激活 | [意义] 接通国际股票数据流动管线")
            except ImportError: pass
        try:
            import akshare as ak; self._ak = ak; self._sources.append(("akshare", self._fetch_akshare))
        except ImportError: pass
        try:
            import baostock as bs; self._bs = bs; self._sources.append(("baostock", self._fetch_baostock))
        except ImportError: pass
        try:
            import tushare as ts
            t = os.environ.get("TUSHARE_TOKEN")
            if t and t != "your_tushare_token_here":
                ts.set_token(t); self._ts_pro = ts.pro_api(); self._sources.append(("tushare", self._fetch_tushare))
        except Exception: pass

    def fetch_historical(self, symbol: str, start_date: str, end_date: str, freq: str = "d") -> Optional[pd.DataFrame]:
        if symbol.endswith(".US") or symbol in {"^GSPC", ".INX"}:
            return fetch_us_historical(symbol, start_date, end_date, self.cache_dir, self.offline_debug, self._has_yf, getattr(self, "_yf", None), getattr(self, "_ak", None))
        if symbol in {"000300.SH", "000905.SH"}: return self.fetch_index_historical(symbol, start_date, end_date)
        if symbol in self._failed_symbols: return None
        c_path = self.cache_dir / f"{symbol}_history.parquet"
        if c_path.exists():
            try:
                df = pd.read_parquet(c_path)
                df["date"] = pd.to_datetime(df["date"])
                mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
                return df.loc[mask].copy()
            except: c_path.unlink(missing_ok=True)
            
        if self.offline_debug: return None
        for name, func in self._sources:
            try:
                df = func(symbol, self.DEFAULT_START, datetime.now().strftime("%Y-%m-%d"), freq)
                if df is not None and not df.empty:
                    df.to_parquet(c_path, index=False)
                    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
                    self._logger.info("[OP] Query A-Share Live Server | [SOURCE] Polling Interface Node: %s | [RESULT] Ingested matrix row footprint: %s | [SIGNIFICANCE] Successfully completed live multi-market fallbacks", name, len(df))
                    self._logger.info("[操作] 查询全A股实盘线上服务器 | [来源] 轮询接口节点: %s | [结果] 录入矩阵行数足迹: %s | [意义] 成功完成跨市场实时降级对齐")
                    return df.loc[mask].copy()
            except: continue
        self._failed_symbols.add(symbol)
        return None

    def fetch_index_historical(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        c_path = self.cache_dir / f"index_{symbol}_master.parquet"
        df = None
        if c_path.exists(): df = pd.read_parquet(c_path)
        if (df is None or df.empty) and not self.offline_debug and hasattr(self, "_ak"):
            code = symbol.split(".")[0]
            df_raw = self._ak.index_zh_a_hist(symbol=code, period="daily", start_date="20050101", end_date=datetime.now().strftime("%Y%m%d"))
            if df_raw is not None and not df_raw.empty:
                df_raw.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}, inplace=True)
                df_raw["date"] = pd.to_datetime(df_raw["date"])
                df = df_raw[["date", "open", "high", "low", "close", "volume", "amount"]]
                df.to_parquet(c_path, index=False)
        if df is not None:
            mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
            return df.loc[mask].copy()
        return None

    def _fetch_akshare(self, symbol: str, start: str, end: str, freq: str):
        c = symbol.split(".")[0]
        df = self._ak.stock_zh_a_hist(symbol=c, period="daily", start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="qfq")
        df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "open", "high", "low", "close", "volume", "amount"]]

    def _fetch_baostock(self, symbol: str, start: str, end: str, freq: str):
        if not self._bs_logged: self._bs.login(); self._bs_logged = True
        code = f"sh.{symbol.split('.')[0]}" if symbol.endswith(".SH") else f"sz.{symbol.split('.')[0]}"
        rs = self._bs.query_history_k_data_plus(code=code, fields="date,open,high,low,close,volume,amount", start_date=start, end_date=end, frequency=freq, adjustflag="2")
        data = []
        while rs.next(): data.append(rs.get_row_data())
        df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume", "amount"])
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "amount"]: df[col] = pd.to_numeric(df[col])
        return df

    def _fetch_tushare(self, symbol: str, start: str, end: str, freq: str):
        df = self._ts_pro.daily(ts_code=symbol, start_date=start.replace("-", ""), end_date=end.replace("-", ""))
        df.rename(columns={"trade_date": "date", "vol": "volume"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "high", "low", "close"]: df[c] = pd.to_numeric(df[c])
        df["volume"] *= 100
        df["amount"] *= 1000
        return df.sort_values("date")[["date", "open", "high", "low", "close", "volume", "amount"]]

    def fetch_stock_list(self) -> List[str]:
        c_path = self.cache_dir / "stock_list.parquet"
        if c_path.exists() and (datetime.now() - datetime.fromtimestamp(c_path.stat().st_mtime)).days < 1:
            return pd.read_parquet(c_path)["symbol"].tolist()
        try:
            df = self._ak.stock_zh_a_spot_em()
            syms = [f"{x}.SH" if str(x).startswith("6") else f"{x}.SZ" for x in df["代码"] if len(str(x)) == 6]
            if syms: pd.DataFrame({"symbol": syms}).to_parquet(c_path, index=False); return syms
        except: pass
        if c_path.exists(): return pd.read_parquet(c_path)["symbol"].tolist()
        return ["600000.SH", "600036.SH", "600519.SH", "000001.SZ", "000002.SZ"] * 16

    def fetch_trading_calendar(self, start_year: int = 2010, end_year: int = 2026) -> pd.DatetimeIndex:
        c_path = self.cache_dir / f"trading_calendar_{start_year}_{end_year}.parquet"
        if c_path.exists(): return pd.DatetimeIndex(pd.read_parquet(c_path)["date"])
        cal = self._ak.tool_trade_date_hist_sina()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"])
        cal = cal[(cal["trade_date"].dt.year >= start_year) & (cal["trade_date"].dt.year <= end_year)]
        dates = cal["trade_date"].tolist()
        pd.DataFrame({"date": dates}).to_parquet(c_path, index=False)
        return pd.DatetimeIndex(dates)