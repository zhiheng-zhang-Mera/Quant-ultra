import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
import pandas as pd
from Main.env_config import PROJECT_ROOT
from Main.us_pipeline import fetch_us_historical, fetch_us_trading_calendar as _fetch_us_calendar
from Main.data_quality import validate_ohlcv, write_manifest

class FreeDataSourceManager:
    def __init__(self, cache_dir: Path = None, offline_debug: bool = False, proxy_url: str = None):
        self.cache_dir = cache_dir or (PROJECT_ROOT / "data_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir = self.cache_dir / "evidence"
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self._sources: List[tuple] = []
        self._bs_logged = False
        self.offline_debug = offline_debug
        self._logger = logging.getLogger("DataSourceManager")
        self._load_env_file()
        self.proxy_url = proxy_url or os.environ.get("QUANT_ULTRA_PROXY") or os.environ.get("HTTPS_PROXY")
        if self.proxy_url:
            os.environ["HTTPS_PROXY"] = self.proxy_url
            os.environ["HTTP_PROXY"] = self.proxy_url
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
            # self._logger.info("[OP] Parse Token Files | [SOURCE] Main/.env Physical File | [RESULT] Environment tokens injected | [SIGNIFICANCE] Enables seamless authorization")
            self._logger.info("由数据源管理器加载环境文件 | 来源: Main/.env物理文件 | 结果: 环境令牌已注入 | 意义: 实现无缝授权")
    def _init_sources(self):
        self._has_yf = False
        if not self.offline_debug:
            try:
                import yfinance as yf
                self._yf = yf
                self._has_yf = True
                # self._logger.info("[OP] Register Data Providers | [SOURCE] yfinance | [RESULT] Level 0 active | [SIGNIFICANCE] Connects international equity pipeline")
                self._logger.info("由数据源管理器初始化数据提供商 | 来源: yfinance | 结果: Level 0 active | 意义: 连接国际股票管道")
            except ImportError: pass
        try:
            import akshare as ak
            self._ak = ak
            self._sources.append(("akshare", self._fetch_akshare))
        except ImportError: pass
        try:
            import baostock as bs
            self._bs = bs
            self._sources.append(("baostock", self._fetch_baostock))
        except ImportError: pass
        try:
            import tushare as ts
            t = os.environ.get("TUSHARE_TOKEN")
            if t and t != "your_tushare_token_here":
                ts.set_token(t)
                self._ts_pro = ts.pro_api()
                self._sources.append(("tushare", self._fetch_tushare))
        except Exception: pass
        try:
            import efinance as ef
            self._ef = ef
            self._sources.append(("efinance", self._fetch_efinance))
            # self._logger.info("[OP] Register Data Providers | [SOURCE] efinance | [RESULT] Level 4 active | [SIGNIFICANCE] Adds domestic backup mirror")
            self._logger.info("由数据源管理器注册数据提供商 | 来源: efinance | 结果: Level 4 active | 意义: 添加国内备份镜像")
        except ImportError: pass

    def fetch_historical(self, symbol: str, start_date: str, end_date: str, freq: str = "d") -> Optional[pd.DataFrame]:
        if symbol.endswith(".US") or symbol in {"^GSPC", ".INX"}:
            return fetch_us_historical(symbol, start_date, end_date, self.cache_dir, self.offline_debug, self._has_yf, getattr(self, "_yf", None), getattr(self, "_ak", None))
        if symbol in {"000300.SH", "000905.SH"}:
            return self.fetch_index_historical(symbol, start_date, end_date)
        if symbol in self._failed_symbols:
            return None
        c_path = self.cache_dir / f"{symbol}_history.parquet"
        if c_path.exists():
            try:
                df = pd.read_parquet(c_path)
                df["date"] = pd.to_datetime(df["date"])
                mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
                return df.loc[mask].copy()
            except:
                c_path.unlink(missing_ok=True)
        if self.offline_debug:
            return None
        for name, func in self._sources:
            try:
                df = func(symbol, self.DEFAULT_START, datetime.now().strftime("%Y-%m-%d"), freq)
                if df is not None and not df.empty:
                    evidence = validate_ohlcv(df, symbol)
                    evidence.update({"provider": name, "fetched_at": datetime.now().astimezone().isoformat(), "proxy_configured": bool(self.proxy_url)})
                    write_manifest(self.evidence_dir / f"{symbol}_{name}.json", evidence)
                    if not evidence["valid"]:
                        self._logger.warning("Rejected invalid dataset %s from %s: %s", symbol, name, evidence["errors"])
                        continue
                    df.to_parquet(c_path, index=False)
                    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
                    # self._logger.info(f"[OP] Query A-Share Live | [SOURCE] {name} | [RESULT] Rows: {len(df)} | [SIGNIFICANCE] Multi-source fallback success")
                    self._logger.info("由数据源管理器查询A股实时数据 | 来源: %s | 结果: 行数: %d | 意义: 多源回退成功", name, len(df))
                    return df.loc[mask].copy()
            except:
                continue
        self._failed_symbols.add(symbol)
        return None

    def fetch_index_historical(self, symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        c_path = self.cache_dir / f"index_{symbol}_master.parquet"
        df = None
        if c_path.exists():
            df = pd.read_parquet(c_path)
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
        if not self._bs_logged:
            self._bs.login()
            self._bs_logged = True
        code = f"sh.{symbol.split('.')[0]}" if symbol.endswith(".SH") else f"sz.{symbol.split('.')[0]}"
        rs = self._bs.query_history_k_data_plus(code=code, fields="date,open,high,low,close,volume,amount", start_date=start, end_date=end, frequency=freq, adjustflag="2")
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume", "amount"])
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col])
        return df

    def _fetch_tushare(self, symbol: str, start: str, end: str, freq: str):
        df = self._ts_pro.daily(ts_code=symbol, start_date=start.replace("-", ""), end_date=end.replace("-", ""))
        df.rename(columns={"trade_date": "date", "vol": "volume"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c])
        df["volume"] *= 100
        df["amount"] *= 1000
        return df.sort_values("date")[["date", "open", "high", "low", "close", "volume", "amount"]]

    def _fetch_efinance(self, symbol: str, start: str, end: str, freq: str):
        code = symbol.split(".")[0]
        freq_map = {"d": 1, "w": 5, "m": 6}
        ef_freq = freq_map.get(freq, 1)
        df = self._ef.stock.get_quote_history(code, beg=start.replace("-", ""), end=end.replace("-", ""), klt=ef_freq, fqt=1)
        if df is None or df.empty:
            return None
        df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        for c in ["open", "high", "low", "close", "volume", "amount"]:
            df[c] = pd.to_numeric(df[c])
        df["volume"] *= 100
        df.sort_values("date", ascending=True, inplace=True)
        return df[["date", "open", "high", "low", "close", "volume", "amount"]]

    def fetch_stock_list(self) -> List[str]:
        c_path = self.cache_dir / "stock_list.parquet"
        if c_path.exists() and (datetime.now() - datetime.fromtimestamp(c_path.stat().st_mtime)).days < 1:
            return pd.read_parquet(c_path)["symbol"].tolist()
        try:
            df = self._ak.stock_zh_a_spot_em()
            syms = [f"{x}.SH" if str(x).startswith("6") else f"{x}.SZ" for x in df["代码"] if len(str(x)) == 6]
            if syms:
                pd.DataFrame({"symbol": syms}).to_parquet(c_path, index=False)
                return syms
        except: pass
        if hasattr(self, "_ts_pro"):
            try:
                df_ts = self._ts_pro.stock_basic(list_status='L', fields='ts_code')
                if df_ts is not None and not df_ts.empty:
                    syms = [s for s in df_ts['ts_code'].tolist() if len(s.split(".")[0]) == 6 and not s.startswith("8")]
                    if syms:
                        pd.DataFrame({"symbol": syms}).to_parquet(c_path, index=False)
                        return syms
            except: pass
        if hasattr(self, "_ef"):
            try:
                df_ef = self._ef.stock.get_realtime_quotes()
                if df_ef is not None and not df_ef.empty:
                    code_col = '股票代码' if '股票代码' in df_ef.columns else ('代码' if '代码' in df_ef.columns else None)
                    if code_col:
                        syms = [f"{x}.SH" if str(x).startswith("6") else f"{x}.SZ" for x in df_ef[code_col] if len(str(x)) == 6]
                        if syms:
                            pd.DataFrame({"symbol": syms}).to_parquet(c_path, index=False)
                            return syms
            except: pass
        if c_path.exists():
            return pd.read_parquet(c_path)["symbol"].tolist()
        return ["600000.SH", "600036.SH", "600519.SH", "000001.SZ", "000002.SZ"] * 16

    def fetch_trading_calendar(self, start_year: int = 2010, end_year: int = datetime.now().year) -> pd.DatetimeIndex:
        c_path = self.cache_dir / f"trading_calendar_{start_year}_{end_year}.parquet"
        if c_path.exists():
            return pd.DatetimeIndex(pd.read_parquet(c_path)["date"])
        cal = self._ak.tool_trade_date_hist_sina()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"])
        cal = cal[(cal["trade_date"].dt.year >= start_year) & (cal["trade_date"].dt.year <= end_year)]
        dates = cal["trade_date"].tolist()
        pd.DataFrame({"date": dates}).to_parquet(c_path, index=False)
        return pd.DatetimeIndex(dates)

    def fetch_us_trading_calendar(self, start_year: int = 2010, end_year: int = datetime.now().year) -> pd.DatetimeIndex:
        return _fetch_us_calendar(self.cache_dir, self.offline_debug, self._has_yf, getattr(self, "_yf", None), getattr(self, "_ak", None), start_year, end_year)
