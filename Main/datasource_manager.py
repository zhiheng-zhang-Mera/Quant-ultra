"""
Quant-Ultra Flow - Free Data Source Manager (Expanded 4-Sources Edition)
"""
import os
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
import pandas as pd
from Main.env_config import PROJECT_ROOT

class FreeDataSourceManager:
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or (PROJECT_ROOT / "data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._sources: List[tuple] = []
        self._bs_logged = False
        self._logger = logging.getLogger("FreeDataSourceManager")
        
        # 优先加载 Main 文件夹下的环境变量文件
        self._load_env_file()
        # 初始化所有已安装的数据源
        self._init_sources()
        
        self.DEFAULT_START = "2005-01-01"
        self._failed_symbols: Set[str] = set()

    def _load_env_file(self):
        """自主解析 .env 文件并注入系统环境变量，避免生产环境缺失第三方包"""
        env_path = PROJECT_ROOT / "Main" / ".env"
        if env_path.exists():
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            os.environ[k.strip()] = v.strip()
                self._logger.info("✅ 成功加载 Main/.env 环境变量配置文件")
            except Exception as e:
                self._logger.warning(f"⚠️ 解析 Main/.env 文件失败: {e}")
        else:
            self._logger.warning("⚠️ 未找到 Main/.env 文件，系统将尝试从现有系统环境变量中读取 Token")

    def _init_sources(self):
        # 1. AkShare (主数据源)
        try:
            import akshare as ak
            self._ak = ak
            self._sources.append(("akshare", self._fetch_akshare))
            self._logger.info("✅ AkShare 注册成功 (Level 1)")
        except ImportError:
            self._logger.warning("⚠️ AkShare 未安装")

        # 2. BaoStock (备用数据源 1)
        try:
            import baostock as bs
            self._bs = bs
            self._sources.append(("baostock", self._fetch_baostock))
            self._logger.info("✅ BaoStock 注册成功 (Level 2)")
        except ImportError:
            self._logger.warning("⚠️ BaoStock 未安装")

        # 3. Tushare (备用数据源 2 - 读取环境变量)
        try:
            import tushare as ts
            token = os.environ.get("TUSHARE_TOKEN")
            if token and token != "your_tushare_token_here":
                ts.set_token(token)
                self._ts_pro = ts.pro_api()
                self._sources.append(("tushare", self._fetch_tushare))
                self._logger.info("✅ Tushare Pro 注册成功 (Level 3)")
            else:
                self._logger.warning("⚠️ Tushare 注册跳过: 环境变量中未检测到有效的 TUSHARE_TOKEN")
        except Exception as e:
            self._logger.warning(f"⚠️ Tushare 加载失败: {e}")

        # 4. efinance (备用数据源 3)
        try:
            import efinance as ef
            self._ef = ef
            self._sources.append(("efinance", self._fetch_efinance))
            self._logger.info("✅ efinance 注册成功 (Level 4)")
        except ImportError:
            self._logger.warning("⚠️ efinance 未安装")

        if not self._sources:
            raise RuntimeError("❌ 未检测到任何可用数据源，请检查基础环境安装情况。")

    def fetch_index_historical(self, symbol: str, start_date: str, end_date: str, freq: str = "d") -> Optional[pd.DataFrame]:
        cache_key = f"index_{symbol}_{start_date}_{end_date}_{freq}.parquet"
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            try:
                if (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days < 7:
                    df = pd.read_parquet(cache_path)
                    if not df.empty: return df
            except: pass
        try:
            code = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
            df = self._ak.stock_zh_index_hist(symbol=code, period="daily",
                                              start_date=start_date.replace("-", ""),
                                              end_date=end_date.replace("-", ""))
            if not df.empty:
                df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
                                   "收盘": "close", "成交量": "volume", "成交额": "amount"}, inplace=True, errors="ignore")
                df["date"] = pd.to_datetime(df["date"])
                df = df[["date", "open", "high", "low", "close", "volume", "amount"]]
                df.to_parquet(cache_path, index=False)
                return df
        except Exception as e:
            self._logger.warning(f"AkShare 指数获取失败 {symbol}: {e}")
        return None

    def fetch_historical(self, symbol: str, start_date: str, end_date: str, freq: str = "d", max_retries: int = 3) -> Optional[pd.DataFrame]:
        index_symbols = {"000300.SH", "000905.SH", "000016.SH", "399001.SZ", "399006.SZ"}
        if symbol in index_symbols:
            return self.fetch_index_historical(symbol, start_date, end_date, freq)

        if symbol in self._failed_symbols: return None
        full_cache_path = self.cache_dir / f"{symbol}_history.parquet"

        if full_cache_path.exists():
            try:
                full_df = pd.read_parquet(full_cache_path)
                if not full_df.empty:
                    full_df["date"] = pd.to_datetime(full_df["date"])
                    mask = (full_df["date"] >= pd.to_datetime(start_date)) & (full_df["date"] <= pd.to_datetime(end_date))
                    return full_df.loc[mask].copy()
                else:
                    full_cache_path.unlink(missing_ok=True)
            except:
                full_cache_path.unlink(missing_ok=True)

        current_date = datetime.now().strftime("%Y-%m-%d")
        download_end = end_date if pd.to_datetime(end_date) < datetime.now() else current_date
        download_start = self.DEFAULT_START

        # 核心解耦：在这里依次对 4 个注册的数据源进行轮询尝试
        last_error = None
        for name, fetch_func in self._sources:
            for attempt in range(1, max_retries + 1):
                try:
                    df = fetch_func(symbol, download_start, download_end, freq)
                    if df is not None and not df.empty:
                        df.to_parquet(full_cache_path, index=False)
                        self._failed_symbols.discard(symbol)
                        mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
                        return df.loc[mask].copy()
                except Exception as e:
                    last_error = e
                    time.sleep(0.5)
        
        if last_error is None: last_error = "四源轮询均返回空数据"
        self._logger.error(f"❌ 所有数据源获取 {symbol} 均失败: {last_error}")
        self._failed_symbols.add(symbol)
        return None

    def _fetch_akshare(self, symbol: str, start_date: str, end_date: str, freq: str = "d"):
        code = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
        period_map = {"d": "daily", "w": "weekly", "m": "monthly"}
        df = self._ak.stock_zh_a_hist(symbol=code, period=period_map.get(freq, "daily"),
                                      start_date=start_date.replace("-", ""),
                                      end_date=end_date.replace("-", ""), adjust="qfq")
        if df.empty: return None
        df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
                           "收盘": "close", "成交量": "volume", "成交额": "amount"}, inplace=True, errors="ignore")
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "open", "high", "low", "close", "volume", "amount"]]

    def _fetch_baostock(self, symbol: str, start_date: str, end_date: str, freq: str = "d"):
        if not self._bs_logged:
            self._bs.login(); self._bs_logged = True
        code = f"sh.{symbol.split('.')[0]}" if symbol.endswith(".SH") else f"sz.{symbol.split('.')[0]}"
        start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        rs = self._bs.query_history_k_data_plus(code=code, fields="date,open,high,low,close,volume,amount",
                                                start_date=start, end_date=end, frequency=freq, adjustflag="2")
        if rs is None or rs.error_code != "0": return None
        data = []
        while rs.next(): data.append(rs.get_row_data())
        if not data: return None
        df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume", "amount"])
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col])
        return df

    def _fetch_tushare(self, symbol: str, start_date: str, end_date: str, freq: str = "d"):
        """Tushare Pro 数据拉取器"""
        # Tushare 频度映射：日线 daily, 周线 weekly, 月线 monthly
        freq_map = {"d": "daily", "w": "weekly", "m": "monthly"}
        api_name = freq_map.get(freq, "daily")
        
        start = pd.to_datetime(start_date).strftime("%Y%m%d")
        end = pd.to_datetime(end_date).strftime("%Y%m%d")
        
        # 动态调用对应的每日/每周接口
        func = getattr(self._ts_pro, api_name)
        df = func(ts_code=symbol, start_date=start, end_date=end)
        
        if df is None or df.empty: return None
        
        # Tushare 列名：trade_date, open, high, low, close, vol, amount
        df.rename(columns={"trade_date": "date", "vol": "volume"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"])
        
        # 数值清洗转换：Tushare日线中的 vol 单位是“手”，amount单位是“千元”
        # 为了与 Akshare/BaoStock (股/元) 尺度对齐，需要进行乘数修正
        df["open"] = pd.to_numeric(df["open"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["close"] = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["volume"]) * 100       # 手 -> 股
        df["amount"] = pd.to_numeric(df["amount"]) * 1000     # 千元 -> 元
        
        df.sort_values("date", ascending=True, inplace=True)
        return df[["date", "open", "high", "low", "close", "volume", "amount"]]

    def _fetch_efinance(self, symbol: str, start_date: str, end_date: str, freq: str = "d"):
        """efinance 数据拉取器"""
        code = symbol.split(".")[0]
        # efinance 频度映射：1代表日线，5代表周线，6代表月线
        freq_map = {"d": 1, "w": 5, "m": 6}
        ef_freq = freq_map.get(freq, 1)
        
        start = pd.to_datetime(start_date).strftime("%Y%m%d")
        end = pd.to_datetime(end_date).strftime("%Y%m%d")
        
        # 获取复权历史数据 (qfq=1, hfq=2, bfq=0)
        df = self._ef.stock.get_quote_history(code, beg=start, end=end, klt=ef_freq, fqt=1)
        if df is None or df.empty: return None
        
        # efinance 经典东方财富列名清洗
        df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high", 
            "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "amount"
        }, inplace=True)
        
        df["date"] = pd.to_datetime(df["date"])
        df["open"] = pd.to_numeric(df["open"])
        df["high"] = pd.to_numeric(df["high"])
        df["low"] = pd.to_numeric(df["low"])
        df["close"] = pd.to_numeric(df["close"])
        df["volume"] = pd.to_numeric(df["volume"]) * 100   # 手 -> 股
        df["amount"] = pd.to_numeric(df["amount"])         # 元尺度本身是对齐的
        
        df.sort_values("date", ascending=True, inplace=True)
        return df[["date", "open", "high", "low", "close", "volume", "amount"]]

    def fetch_stock_list(self, max_retries: int = 3) -> List[str]:
        cache_path = self.cache_dir / "stock_list.parquet"
        if cache_path.exists():
            try:
                if (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days < 1:
                    df = pd.read_parquet(cache_path)
                    if not df.empty: return df["symbol"].tolist()
            except: pass
        for attempt in range(1, max_retries + 1):
            try:
                if hasattr(self, "_ak"):
                    df = self._ak.stock_zh_a_spot_em()
                    if not df.empty:
                        symbols = df["代码"].apply(lambda x: f"{x}.SH" if x.startswith("6") else f"{x}.SZ").tolist()
                        symbols = [s for s in symbols if not s.startswith("8") and len(s.split(".")[0]) == 6]
                        if symbols:
                            pd.DataFrame({"symbol": symbols}).to_parquet(cache_path, index=False)
                            return symbols
            except: time.sleep(1)
        raise RuntimeError("❌ 获取全 A 股票列表失败。")

    def fetch_trading_calendar(self, start_year: int = 2010, end_year: int = 2026) -> pd.DatetimeIndex:
        cache_path = self.cache_dir / f"trading_calendar_{start_year}_{end_year}.parquet"
        if cache_path.exists():
            try:
                if (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days < 7:
                    return pd.DatetimeIndex(pd.read_parquet(cache_path)["date"])
            except: pass
        try:
            if hasattr(self, "_ak"):
                cal = self._ak.tool_trade_date_hist_sina()
                if cal is not None and not cal.empty:
                    cal["trade_date"] = pd.to_datetime(cal["trade_date"])
                    cal = cal[(cal["trade_date"].dt.year >= start_year) & (cal["trade_date"].dt.year <= end_year)]
                    dates = cal["trade_date"].tolist()
                    if dates:
                        pd.DataFrame({"date": dates}).to_parquet(cache_path, index=False)
                        return pd.DatetimeIndex(dates)
        except Exception as e:
            self._logger.warning(f"获取日历异常: {e}")
        raise RuntimeError(f"❌ 获取交易日历失败。")