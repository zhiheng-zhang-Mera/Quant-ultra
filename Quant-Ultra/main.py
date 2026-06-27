"""
Quant-Ultra + Conformal-BL Investment Workflow Engine
Main Orchestrator [2026 Production Release]
Fully compliant with Final-Flow.md (Logging, Audit, Git Hash Binding, Hard Kill)
"""
import sys
import os
import logging
import argparse
import subprocess
import json
import hashlib
import traceback
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Callable

import pandas as pd
import numpy as np
import pytz

warnings.filterwarnings("ignore")

# ============================
# 0. 全局常量与路径
# ============================
PROJECT_ROOT = Path(__file__).parent.resolve()
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

RUN_TIMESTAMP = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f")[:-3]
MAIN_LOG_FILE = LOG_DIR / f"orchestrator_{RUN_TIMESTAMP}.log"

PHASE_MODULES = [
    "step1_data_foundation",
    "step2_data_slicing",
    "step3_pit_setup",
    "step4_labeling_weighting",
    "step5_model_training_calibration",
    "step6_position_sizing",
    "step7_fsm_backtest",
    "step8_audit_stress_test",
    "step9_live_mlops",
]

PHASE_DEPENDENCIES: Dict[str, Set[str]] = {
    "step1_data_foundation": set(),
    "step2_data_slicing": {"step1_data_foundation"},
    "step3_pit_setup": {"step1_data_foundation", "step2_data_slicing"},
    "step4_labeling_weighting": {"step1_data_foundation", "step2_data_slicing", "step3_pit_setup"},
    "step5_model_training_calibration": {"step1_data_foundation", "step2_data_slicing", "step3_pit_setup",
                                         "step4_labeling_weighting"},
    "step6_position_sizing": {"step1_data_foundation", "step2_data_slicing", "step3_pit_setup",
                              "step4_labeling_weighting", "step5_model_training_calibration"},
    "step7_fsm_backtest": {"step1_data_foundation", "step2_data_slicing", "step3_pit_setup",
                           "step4_labeling_weighting", "step5_model_training_calibration",
                           "step6_position_sizing"},
    "step8_audit_stress_test": {"step1_data_foundation", "step2_data_slicing", "step3_pit_setup",
                                "step4_labeling_weighting", "step5_model_training_calibration",
                                "step6_position_sizing", "step7_fsm_backtest"},
    "step9_live_mlops": {"step1_data_foundation", "step2_data_slicing", "step3_pit_setup",
                         "step4_labeling_weighting", "step5_model_training_calibration",
                         "step6_position_sizing", "step7_fsm_backtest", "step8_audit_stress_test"},
}

# ============================
# 1. 日志系统初始化
# ============================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(MAIN_LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("Orchestrator")

# ============================
# 2. 环境指纹与 Git 校验
# ============================
def get_git_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode('ascii').strip()
    except Exception:
        return "NO_GIT"

def get_git_status() -> str:
    try:
        status = subprocess.check_output(['git', 'status', '--porcelain'], stderr=subprocess.DEVNULL).decode('ascii').strip()
        return "CLEAN" if not status else "DIRTY"
    except Exception:
        return "UNKNOWN"

ENV_FINGERPRINT = {
    "git_commit_hash": get_git_hash(),
    "git_status": get_git_status(),
    "python_version": sys.version,
    "hostname": os.uname().nodename if hasattr(os, 'uname') else "UNKNOWN",
    "run_timestamp": RUN_TIMESTAMP,
    "working_dir": str(PROJECT_ROOT)
}

if ENV_FINGERPRINT["git_status"] == "DIRTY":
    logger.critical("=" * 80)
    logger.critical("FATAL: Git 工作区存在未提交修改，禁止启动生产级回测！")
    logger.critical("请先提交或暂存所有更改，确保工作区 CLEAN。")
    logger.critical("=" * 80)
    sys.exit(1)

# ============================
# 3. 免费开源数据源管理器（增强版）
# ============================
class FreeDataSourceManager:
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or (PROJECT_ROOT / "data_cache")
        self.cache_dir.mkdir(exist_ok=True)
        self._sources: List[tuple] = []
        self._bs_logged = False
        self._logger = logging.getLogger("FreeDataSourceManager")
        self._init_sources()
        
    def _init_sources(self):
        try:
            import akshare as ak
            self._ak = ak
            self._sources.append(("akshare", self._fetch_akshare))
            logger.info("✅ AkShare 加载成功（主数据源）")
        except ImportError as e:
            logger.warning(f"⚠️ AkShare 未安装: {e}")
        try:
            import baostock as bs
            self._bs = bs
            self._sources.append(("baostock", self._fetch_baostock))
            logger.info("✅ BaoStock 加载成功（备用数据源）")
        except ImportError as e:
            logger.warning(f"⚠️ BaoStock 未安装: {e}")
        if not self._sources:
            raise RuntimeError("❌ 未检测到任何免费数据源（AkShare / BaoStock）。请安装至少一个。")

    # ---------- 指数专用获取 ----------
    def fetch_index_historical(self, symbol: str, start_date: str, end_date: str, freq: str = "d") -> Optional[pd.DataFrame]:
        """专门获取指数历史数据，优先 AkShare 的 stock_zh_index_hist"""
        cache_key = f"index_{symbol}_{start_date}_{end_date}_{freq}.parquet"
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            try:
                if (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days < 7:
                    df = pd.read_parquet(cache_path)
                    if not df.empty:
                        return df
            except:
                pass

        # 1) AkShare 指数接口
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

        # 2) BaoStock 降级
        try:
            if not self._bs_logged:
                self._bs.login()
                self._bs_logged = True
            code = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
            bs_code = f"sh.{code}" if symbol.endswith(".SH") else f"sz.{code}"
            # 强制清洗日期
            start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
            end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
            rs = self._bs.query_history_k_data_plus(code=bs_code,
                                                    fields="date,open,high,low,close,volume,amount",
                                                    start_date=start, end_date=end,
                                                    frequency="d", adjustflag="2")
            if rs is not None and rs.error_code == "0":
                data = []
                while rs.next():
                    data.append(rs.get_row_data())
                if data:
                    df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume", "amount"])
                    df["date"] = pd.to_datetime(df["date"])
                    for col in ["open", "high", "low", "close", "volume", "amount"]:
                        df[col] = pd.to_numeric(df[col])
                    df.to_parquet(cache_path, index=False)
                    return df
        except Exception as e:
            self._logger.warning(f"BaoStock 指数获取失败 {symbol}: {e}")

        return None  # 失败返回 None

    def fetch_historical(self, symbol: str, start_date: str, end_date: str, freq: str = "d", max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        统一入口，自动判别指数或个股。
        所有失败返回 None，不再抛出 RuntimeError。
        """
        # 判断是否为常见指数
        index_symbols = {"000300.SH", "000905.SH", "000016.SH", "399001.SZ", "399006.SZ"}
        if symbol in index_symbols:
            return self.fetch_index_historical(symbol, start_date, end_date, freq)

        # 个股逻辑（原实现，但改为返回 None）
        cache_key = f"{symbol}_{start_date}_{end_date}_{freq}.parquet"
        cache_path = self.cache_dir / cache_key
        if cache_path.exists():
            try:
                mtime = datetime.fromtimestamp(cache_path.stat().st_mtime)
                if (datetime.now() - mtime).days < 7:
                    df = pd.read_parquet(cache_path)
                    if not df.empty:
                        return df
            except Exception:
                pass

        last_error = None
        for name, fetch_func in self._sources:
            for attempt in range(1, max_retries + 1):
                try:
                    df = fetch_func(symbol, start_date, end_date, freq)
                    if df is not None and not df.empty:
                        df.to_parquet(cache_path, index=False)
                        return df
                except Exception as e:
                    last_error = e
                    time.sleep(1)
        if last_error is None:
            last_error = "所有数据源返回空数据（未抛出异常）"
        self._logger.error(f"❌ 所有数据源获取 {symbol} 均失败，最后异常: {last_error}")
        return None  # 改为返回 None，不再抛异常

    def _fetch_akshare(self, symbol: str, start_date: str, end_date: str, freq: str = "d"):
        code = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
        period_map = {"d": "daily", "w": "weekly", "m": "monthly"}
        period = period_map.get(freq, "daily")
        df = self._ak.stock_zh_a_hist(symbol=code, period=period,
                                      start_date=start_date.replace("-", ""),
                                      end_date=end_date.replace("-", ""), adjust="qfq")
        if df.empty:
            return None
        df.rename(columns={"日期": "date", "开盘": "open", "最高": "high", "最低": "low",
                           "收盘": "close", "成交量": "volume", "成交额": "amount"}, inplace=True, errors="ignore")
        df["date"] = pd.to_datetime(df["date"])
        return df[["date", "open", "high", "low", "close", "volume", "amount"]]

    def _fetch_baostock(self, symbol: str, start_date: str, end_date: str, freq: str = "d"):
        if not self._bs_logged:
            self._bs.login()
            self._bs_logged = True
        code = symbol.replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
        bs_code = f"sh.{code}" if symbol.endswith(".SH") else f"sz.{code}"
        freq_map = {"d": "d", "w": "w", "m": "m"}
        bs_freq = freq_map.get(freq, "d")
        # 关键修复：强制清洗日期格式
        start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
        end = pd.to_datetime(end_date).strftime("%Y-%m-%d")
        rs = self._bs.query_history_k_data_plus(code=bs_code,
                                                fields="date,open,high,low,close,volume,amount",
                                                start_date=start, end_date=end,
                                                frequency=bs_freq, adjustflag="2")
        if rs is None:
            return None
        if rs.error_code != "0":
            return None
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        if not data:
            return None
        df = pd.DataFrame(data, columns=["date", "open", "high", "low", "close", "volume", "amount"])
        df["date"] = pd.to_datetime(df["date"])
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col])
        return df

    def fetch_stock_list(self, max_retries: int = 3) -> List[str]:
        cache_path = self.cache_dir / "stock_list.parquet"
        if cache_path.exists():
            try:
                if (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days < 1:
                    df = pd.read_parquet(cache_path)
                    if not df.empty:
                        return df["symbol"].tolist()
            except Exception:
                pass
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
            except Exception:
                time.sleep(1)
        for attempt in range(1, max_retries + 1):
            try:
                if not self._bs_logged:
                    self._bs.login()
                    self._bs_logged = True
                rs = self._bs.query_all_stock()
                if rs is None:
                    continue
                if rs.error_code == "0":
                    symbols = []
                    while rs.next():
                        row = rs.get_row_data()
                        code = row[0].split(".")[1]
                        if row[0].startswith("sh.6") or row[0].startswith("sz.0") or row[0].startswith("sz.3"):
                            symbols.append(f"{code}.SH" if row[0].startswith("sh") else f"{code}.SZ")
                    if symbols:
                        pd.DataFrame({"symbol": symbols}).to_parquet(cache_path, index=False)
                        return symbols
            except Exception:
                time.sleep(1)
        raise RuntimeError("❌ 所有数据源获取全 A 股票列表均失败。")

    def fetch_trading_calendar(self, start_year: int = 2010, end_year: int = 2026) -> pd.DatetimeIndex:
            cache_path = self.cache_dir / f"trading_calendar_{start_year}_{end_year}.parquet"
            if cache_path.exists():
                try:
                    if (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)).days < 7:
                        df = pd.read_parquet(cache_path)
                        if not df.empty:
                            return pd.DatetimeIndex(df["date"])
                except Exception as e:
                    self._logger.warning(f"读取交易日历缓存失败: {e}")

            # 1) AkShare
            try:
                if hasattr(self, "_ak"):
                    cal = self._ak.tool_trade_date_hist_sina()
                    if cal is not None and not cal.empty:
                        cal["trade_date"] = pd.to_datetime(cal["trade_date"])
                        cal = cal[(cal["trade_date"].dt.year >= start_year) & (cal["trade_date"].dt.year <= end_year)]
                        dates = cal["trade_date"].tolist()
                        if dates:
                            pd.DataFrame({"date": dates}).to_parquet(cache_path, index=False)
                            self._logger.info(f"AkShare 获取交易日历成功，共 {len(dates)} 天")
                            return pd.DatetimeIndex(dates)
                    else:
                        self._logger.warning("AkShare 返回空日历")
            except Exception as e:
                self._logger.warning(f"AkShare 获取交易日历异常: {e}")

            # 2) BaoStock
            try:
                if not self._bs_logged:
                    self._bs.login()
                    self._bs_logged = True
                all_dates = []
                for year in range(start_year, end_year + 1):
                    rs = self._bs.query_trade_dates(start_date=f"{year}-01-01", end_date=f"{year}-12-31")
                    if rs is None:
                        self._logger.warning(f"BaoStock 查询 {year} 年交易日返回 None")
                        continue
                    if rs.error_code != "0":
                        self._logger.warning(f"BaoStock 查询 {year} 年交易日失败，错误码: {rs.error_code}")
                        continue
                    while rs.next():
                        row = rs.get_row_data()
                        if row[1] == "1":   # 1 表示交易日
                            all_dates.append(pd.Timestamp(row[0]))
                if all_dates:
                    pd.DataFrame({"date": all_dates}).to_parquet(cache_path, index=False)
                    self._logger.info(f"BaoStock 获取交易日历成功，共 {len(all_dates)} 天")
                    return pd.DatetimeIndex(all_dates)
                else:
                    self._logger.warning("BaoStock 未返回任何交易日")
            except Exception as e:
                self._logger.warning(f"BaoStock 获取交易日历异常: {e}")

            # 3) pandas_market_calendars 降级
            try:
                import pandas_market_calendars as mcal
                sse = mcal.get_calendar('SSE')
                start = datetime(start_year, 1, 1)
                end = datetime(end_year, 12, 31)
                schedule = sse.schedule(start_date=start, end_date=end)
                dates = schedule.index.tz_localize(None).to_pydatetime().tolist()
                if dates:
                    pd.DataFrame({"date": dates}).to_parquet(cache_path, index=False)
                    self._logger.info(f"pandas_market_calendars 获取交易日历成功，共 {len(dates)} 天")
                    return pd.DatetimeIndex(dates)
            except ImportError:
                self._logger.warning("pandas_market_calendars 未安装，无法降级")
            except Exception as e:
                self._logger.warning(f"pandas_market_calendars 降级失败: {e}")

            raise RuntimeError(f"❌ 所有数据源获取交易日历 ({start_year}-{end_year}) 均失败。")

# ============================
# 4. 审计日志
# ============================
class AuditLogger:
    def __init__(self, log_dir: Path):
        self.log_dir = log_dir
        self.log_dir.mkdir(exist_ok=True)
        self.events: List[Dict] = []
        self._session_id = RUN_TIMESTAMP

    def log_event(self, event_type: str, details: Dict[str, Any]):
        entry = {
            "timestamp": datetime.now(pytz.timezone("Asia/Shanghai")).isoformat(timespec="milliseconds"),
            "session_id": self._session_id,
            "event_type": event_type,
            "details": details
        }
        self.events.append(entry)
        log_path = self.log_dir / f"audit_{self._session_id}.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def flush(self):
        full_path = self.log_dir / f"audit_full_{self._session_id}.json"
        with open(full_path, "w", encoding="utf-8") as f:
            json.dump(self.events, f, indent=2, ensure_ascii=False)

# ============================
# 5. PIT 数据总线（统一实现）
# ============================
class PITDataBus:
    """
    统一的数据总线，整合原子存储与历史数据加载，并提供 BL 融合所需的全部方法。
    """
    def __init__(self, data_manager: FreeDataSourceManager, audit_logger=None,
                 strict_mode: bool = True, tz=pytz.timezone("Asia/Shanghai")):
        self.manager = data_manager
        self._tz = tz
        self._cache: Dict[str, pd.DataFrame] = {}          # asset -> full history (indexed by date)
        self._atom_storage: Dict[str, List[Dict]] = {}     # event_type -> list of atom records
        self._universe: Optional[List[str]] = None
        self._logger = logging.getLogger("PITDataBus")
        self.audit_logger = audit_logger
        self._strict_mode = strict_mode
        # 用于 BL 融合的缓存
        self._mcap_cache: Dict[str, float] = {}
        self._sector_cache: Dict[str, str] = {}
        self._margin_cache: Dict[str, bool] = {}
        self._price_cache: Dict[tuple, float] = {}
        self._risk_aversion_cache: Dict[str, float] = {}

    def _log_audit(self, event_type: str, details: dict):
        if self.audit_logger:
            self.audit_logger.log_event(event_type, details)
        else:
            self._logger.warning(f"AUDIT: {event_type} -> {details}")

    def _handle_failure(self, method_name: str, asset: str, error: Exception,
                        fallback_value=None, raise_exception: bool = None):
        if raise_exception is None:
            raise_exception = self._strict_mode
        details = {"method": method_name, "asset": asset, "error": str(error), "fallback": fallback_value}
        self._log_audit("DATA_FETCH_FAILED", details)
        if raise_exception:
            raise RuntimeError(f"数据获取失败 [{method_name}] {asset}: {error}")
        else:
            self._logger.warning(f"数据获取失败，使用降级值 {fallback_value}，asset={asset}")
            return fallback_value

    # ---------- 原子存储接口 ----------
    def append_atom(self, asset: str, timestamp: datetime, value: Any,
                    event_type: str, announcement_date: datetime):
        if event_type not in self._atom_storage:
            self._atom_storage[event_type] = []
        rec = {
            "asset": asset,
            "timestamp": self._ensure_ms_precision(timestamp),
            "value": value,
            "announcement_date": self._ensure_ms_precision(announcement_date)
        }
        self._atom_storage[event_type].append(rec)

    def query_by_pit(self, asset: str, timestamp_T: datetime, field: str) -> Optional[Any]:
        """
        从原子存储中查询最新值。若 field 为 'total_return_price' 或 'close' 等，
        也可从历史缓存中直接获取，此处优先从原子存储查找，若没有则尝试从历史加载。
        """
        t_target = self._ensure_ms_precision(timestamp_T)
        # 先从原子存储中查找
        if field in self._atom_storage:
            records = self._atom_storage[field]
            valid = [r for r in records if r["asset"] == asset and r["announcement_date"] <= t_target]
            if valid:
                valid.sort(key=lambda x: x["timestamp"])
                return valid[-1]["value"]
        # 若未找到，尝试从历史缓存（load_asset_history）中获取
        # 注意：此处假设 field 为价格或成交量等，需明确映射
        # 简化：对于价格类字段，直接调用 load_asset_history 并取最近值
        if field in ["total_return_price", "close", "open", "high", "low", "volume", "amount", "adv"]:
            df = self.load_asset_history(asset, end_date=timestamp_T.strftime("%Y-%m-%d"))
            if df is not None and not df.empty:
                # 找到 <= timestamp_T 的最新行
                idx = df.index.searchsorted(pd.Timestamp(t_target).tz_localize(None), side='right') - 1
                if idx >= 0:
                    if field in df.columns:
                        return df.iloc[idx][field]
                    elif field == "total_return_price" and "close" in df.columns:
                        return df.iloc[idx]["close"]  # 前复权 close 即为全收益价格
        return None

    def _ensure_ms_precision(self, dt: datetime) -> datetime:
        dt = dt.replace(microsecond=(dt.microsecond // 1000) * 1000)
        return dt.astimezone(self._tz)

    # ---------- 历史数据加载 ----------
    def load_asset_history(self, asset: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        """
        加载并缓存资产历史数据，返回 DataFrame，索引为日期，包含 open, high, low, close, volume, amount,
        以及衍生列 log_return, total_return_price (即 close 前复权), adv (滚动20日平均成交额)。
        """
        if asset in self._cache:
            df = self._cache[asset]
            if end_date is not None:
                # 截取至 end_date
                df = df[df.index <= pd.Timestamp(end_date)]
                if df.empty:
                    return None
            if start_date is not None:
                df = df[df.index >= pd.Timestamp(start_date)]
            return df

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = "2010-01-01"
        raw = self.manager.fetch_historical(asset, start_date, end_date)
        if raw is None or raw.empty:
            self._logger.warning(f"资产 {asset} 历史数据为空")
            return None
        raw.set_index("date", inplace=True)
        raw.sort_index(inplace=True)
        # 计算衍生指标
        raw["log_return"] = np.log(raw["close"] / raw["close"].shift(1))
        raw["total_return_price"] = raw["close"]   # 前复权价格即全收益价格
        raw["adv"] = raw["amount"].rolling(20, min_periods=1).mean()
        self._cache[asset] = raw
        return raw

    # ---------- 基准数据 ----------
    def fetch_benchmark_prices(self, start_date: str, end_date: str) -> pd.Series:
        df = self.manager.fetch_historical("000300.SH", start_date, end_date)
        if df is None or df.empty:
            self._logger.warning("基准指数数据为空，返回空 Series")
            return pd.Series()
        df.set_index("date", inplace=True)
        return df["close"]

    def get_benchmark_code(self) -> str:
        return "000300.SH"

    def get_benchmark_returns(self, start_date: str, end_date: str, freq='W') -> pd.Series:
        try:
            df = self.manager.fetch_historical("000300.SH", start_date, end_date)
            if df is None or df.empty:
                return pd.Series()
            df.set_index("date", inplace=True)
            ret = np.log(df["close"] / df["close"].shift(1))
            if freq == 'W':
                ret = ret.resample('W').last()
            return ret
        except Exception as e:
            self._logger.warning(f"获取基准收益率失败: {e}")
            return pd.Series()

    # ---------- BL 融合所需方法 ----------
    def get_free_float_market_cap(self, asset: str, date: datetime) -> float:
        cache_key = f"{asset}_{date.strftime('%Y%m%d')}"
        if cache_key in self._mcap_cache:
            return self._mcap_cache[cache_key]
        try:
            code = self._to_bs_code(asset)
            rs = self.manager._bs.query_history_k_data_plus(
                code=code, fields="date,free_float,close",
                start_date=date.strftime("%Y-%m-%d"), end_date=date.strftime("%Y-%m-%d"),
                adjustflag="2"
            )
            if rs is None:
                raise Exception("BaoStock 返回 None")
            if rs.error_code == "0" and rs.next():
                row = rs.get_row_data()
                free_float = float(row[1])  # 万股
                price = float(row[2])
                mcap = (free_float * price) / 1e4  # 亿元
                self._mcap_cache[cache_key] = mcap
                return mcap
            else:
                raise Exception("BaoStock 返回空或错误")
        except Exception as e:
            return self._handle_failure("get_free_float_market_cap", asset, e, fallback_value=0.0)

    def get_sector(self, asset: str) -> str:
        if asset in self._sector_cache:
            return self._sector_cache[asset]
        try:
            df = self.manager._ak.stock_industry_sw()
            code = asset.split('.')[0]
            code_col = "股票代码" if "股票代码" in df.columns else "代码"
            row = df[df[code_col] == code]
            if row.empty:
                raise Exception("未找到代码")
            sector_col = None
            for col in ['申万行业', '行业', '申万一级行业']:
                if col in row.columns:
                    sector_col = col
                    break
            if sector_col is None:
                raise Exception("行业列不存在")
            sector = row.iloc[0][sector_col]
            self._sector_cache[asset] = sector
            return sector
        except Exception as e:
            try:
                rs = self.manager._bs.query_stock_industry(code=self._to_bs_code(asset))
                if rs is None:
                    raise Exception("BaoStock 返回 None")
                if rs.error_code == "0" and rs.next():
                    sector = rs.get_row_data()[1]
                    self._sector_cache[asset] = sector
                    return sector
            except Exception:
                pass
            return self._handle_failure("get_sector", asset, e, fallback_value="未知")

    def is_marginable(self, asset: str) -> bool:
        if asset in self._margin_cache:
            return self._margin_cache[asset]
        try:
            sse = self.manager._ak.stock_margin_sse(start_date="", end_date="")
            sz = self.manager._ak.stock_margin_sz(start_date="", end_date="")
            all_codes = set(sse['证券代码']) | set(sz['证券代码'])
            code = asset.split('.')[0]
            is_margin = code in all_codes
            self._margin_cache[asset] = is_margin
            return is_margin
        except Exception as e:
            return self._handle_failure("is_marginable", asset, e, fallback_value=False)

    def get_short_rate(self, asset: str) -> float:
        return 0.08 / 252

    def compute_market_risk_aversion(self, end_date: str, window_years=5) -> float:
        cache_key = f"{end_date}_{window_years}"
        if cache_key in self._risk_aversion_cache:
            return self._risk_aversion_cache[cache_key]
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d")
            start = end - timedelta(days=window_years*365)
            rets = self.get_benchmark_returns(start.strftime("%Y-%m-%d"), end_date, freq='W')
            if len(rets) < 20:
                raise Exception("有效周收益率不足")
            ann_mean = rets.mean() * 52
            ann_std = rets.std() * np.sqrt(52)
            rf = 0.025
            lambda_mkt = (ann_mean - rf) / (ann_std ** 2) if ann_std > 0 else 0.02
            self._risk_aversion_cache[cache_key] = lambda_mkt
            return lambda_mkt
        except Exception as e:
            return self._handle_failure("compute_market_risk_aversion", "market", e, fallback_value=0.02)

    def _to_bs_code(self, asset):
        code = asset.split('.')[0]
        if asset.endswith('.SH'):
            return f"sh.{code}"
        elif asset.endswith('.SZ'):
            return f"sz.{code}"
        else:
            return f"sh.{code}"

    def get_universe(self, refresh: bool = False) -> List[str]:
        if self._universe is None or refresh:
            self._universe = self.manager.fetch_stock_list()
        return self._universe

# ============================
# 6. 阶段加载与依赖校验
# ============================
def load_phase_module(phase_name: str):
    try:
        mod = __import__(phase_name)
        if hasattr(mod, "execute") and callable(mod.execute):
            return mod.execute
        else:
            logger.warning(f"模块 {phase_name} 缺少可调用 execute 函数")
            return None
    except ImportError as e:
        logger.warning(f"模块 {phase_name} 导入失败: {e}")
        return None

def check_dependencies(phase: str, completed_phases: Set[str]) -> bool:
    deps = PHASE_DEPENDENCIES.get(phase, set())
    missing = deps - completed_phases
    if missing:
        logger.error(f"阶段 {phase} 缺少前置依赖: {missing}")
        return False
    return True

def save_context_snapshot(context: Dict, phase_name: str):
    snapshot_path = LOG_DIR / f"context_{phase_name}_{RUN_TIMESTAMP}.json"
    try:
        safe_ctx = {}
        for k, v in context.items():
            if k in ["data_bus", "audit_logger", "data_manager"]:
                safe_ctx[k] = f"<{type(v).__name__} object>"
            elif isinstance(v, (pd.DataFrame, pd.Series, np.ndarray)):
                safe_ctx[k] = f"<{type(v).__name__} shape={v.shape if hasattr(v, 'shape') else 'N/A'}>"
            else:
                safe_ctx[k] = v
        with open(snapshot_path, "w", encoding="utf-8") as f:
            json.dump(safe_ctx, f, indent=2, default=str, ensure_ascii=False)
    except Exception as e:
        logger.warning(f"快照保存失败: {e}")

# ============================
# 7. 命令行参数解析
# ============================
def parse_args():
    parser = argparse.ArgumentParser(description="Quant-Ultra + Conformal-BL 量化投资工作流主控器")
    parser.add_argument("--config", type=str, default="./config.yaml", help="外部 YAML 配置文件路径")
    parser.add_argument("--skip-phases", type=str, default="", help="跳过指定阶段（逗号分隔）")
    parser.add_argument("--only-phase", type=str, default=None, help="仅执行指定阶段（及其前置依赖）")
    parser.add_argument("--resume-from", type=str, default=None, help="从指定阶段恢复执行")
    parser.add_argument("--no-git-check", action="store_true", help="【危险】跳过 Git 脏工作区检查")
    return parser.parse_args()

# ============================
# 8. 主调度函数
# ============================
def run_pipeline(args):
    logger.info("=" * 80)
    logger.info("🚀 LAUNCHING QUANT-ULTRA + CONFORMAL-BL PRODUCTION FLOW")
    logger.info("=" * 80)
    logger.info(f"Run Timestamp   : {RUN_TIMESTAMP}")
    logger.info(f"Git Commit Hash : {ENV_FINGERPRINT['git_commit_hash']}")
    logger.info(f"Git Status      : {ENV_FINGERPRINT['git_status']}")
    logger.info(f"Main Log File   : {MAIN_LOG_FILE}")
    logger.info(f"Command Line    : {' '.join(sys.argv)}")

    if args.no_git_check:
        logger.warning("⚠️ 已强制跳过 Git 脏工作区检查（仅用于调试）")

    skip_set = set(args.skip_phases.split(",")) if args.skip_phases else set()
    skip_set = {p.strip() for p in skip_set if p.strip()}

    if args.only_phase:
        target = args.only_phase
        if target not in PHASE_MODULES:
            logger.error(f"无效阶段名: {target}，可选: {PHASE_MODULES}")
            sys.exit(1)
        phases_to_run = []
        def collect_deps(p):
            for dep in PHASE_DEPENDENCIES.get(p, set()):
                if dep not in phases_to_run:
                    collect_deps(dep)
            if p not in phases_to_run:
                phases_to_run.append(p)
        collect_deps(target)
        phases_to_run = [p for p in PHASE_MODULES if p in phases_to_run and p not in skip_set]
    elif args.resume_from:
        if args.resume_from not in PHASE_MODULES:
            logger.error(f"无效阶段名: {args.resume_from}，可选: {PHASE_MODULES}")
            sys.exit(1)
        start_idx = PHASE_MODULES.index(args.resume_from)
        phases_to_run = [p for p in PHASE_MODULES[start_idx:] if p not in skip_set]
    else:
        phases_to_run = [p for p in PHASE_MODULES if p not in skip_set]

    if not phases_to_run:
        logger.warning("没有需要执行的阶段，请检查 --skip-phases 或 --only-phase 参数")
        return

    logger.info(f"📋 计划执行阶段 (共 {len(phases_to_run)} 个): {', '.join(phases_to_run)}")

    try:
        data_manager = FreeDataSourceManager()
        audit_logger = AuditLogger(LOG_DIR)
        data_bus = PITDataBus(data_manager, audit_logger=audit_logger, strict_mode=True)
    except Exception as e:
        logger.critical(f"❌ 基础设施初始化失败: {e}")
        sys.exit(1)

    config = {}
    if args.config and Path(args.config).exists():
        try:
            import yaml
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            logger.info(f"✅ 外部配置加载成功: {args.config}")
        except Exception as e:
            logger.warning(f"配置加载失败: {e}")

    np.random.seed(42)  # 固定随机种子，确保可复现

    # 生成默认完整配置（包含所有可调参数）
    default_config = {
        "adv_window": 20,
        "min_adv_threshold": 1e7,
        "ipo_safety_days": 20,
        "max_participation_rate": 0.05,
        "expected_turnover": 0.05,
        "max_single_stock_weight": 0.05,
        "default_residual_rate": 0.0,
        "impact_alpha": 0.5,
        "impact_kappa_base": 0.05,
        "spread_lookback_days": 60,
        "stock_cap_pct": 0.045,
        "total_shares_source": "free_float",
        "short_rate_default": 0.08/252,
        "short_rate_source": "fixed",
        "tau_BL": 0.02,
        "omega_min": 1e-8,
        "omega_max": 0.01,
        "gamma_risk_initial": 2.5,
        "sector_limit": 0.3,
        "epsilon": 0.001,
        "transaction_cost_coeff": 0.0003,
        "lambda_decay": 0.01,
        "vol_window": 20,
        "threshold_multiplier": 0.5,
        "min_vol_obs": 5,
        "error_threshold_window": 252,
        "embargo_min": 5,
        "holding_period": 5,
        "max_leverage": 2.0,
        "d_min_search": [0.1, 0.3, 0.5, 0.7, 0.9],
        "vif_threshold": 30,
        "cluster_select_ratio": 0.8,
        "lgb_params": {
            "n_estimators": 100,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "deterministic": True,
            "num_threads": 1,
            "random_state": 42,
            "verbosity": -1,
        },
        "train_b1_grid_gamma": np.linspace(0.3, 0.7, 9).tolist(),
        "error_min_samples": 50,
        "cv_folds": 3,
        "psi_lookback_days": 60,
        "volatility_window": 20,
        "crowded_corr_threshold": 0.95,
        "vol_compress_quantile": 0.1,
        "mae_threshold": 1e-5,
        "watchdog_timeout": 30,
        "psi_threshold": 0.25,
        "psi_window": 5,
        "max_incremental_trees": 2000,
        "max_model_size": 2e9,
        "smoothing_period": 25,
        }
    # 合并用户配置（若提供）
    for k, v in default_config.items():
        if k not in config:
            config[k] = v

    pipeline_context = {
        "run_metadata": {
            "timestamp": RUN_TIMESTAMP,
            "git_hash": ENV_FINGERPRINT["git_commit_hash"],
            "args": vars(args)
        },
        "config": config,
        "data_bus": data_bus,
        "data_manager": data_manager,
        "audit_logger": audit_logger,
        "trading_days_dt": None,
        "slices": {},
        "assets": [],
        "embargo_window": None,
        "holding_period": config.get("holding_period", 5),
        "_completed_phases": set(),
        "_phase_status": {},
        "_phase_timings": {},
    }

    try:
        logger.info("正在加载交易日历...")
        cal = data_manager.fetch_trading_calendar(2010, 2026)
        pipeline_context["trading_days_dt"] = cal.tz_localize(pytz.timezone("Asia/Shanghai")).tolist()
        logger.info(f"✅ 交易日历加载完成，共 {len(cal)} 个交易日")
    except Exception as e:
        logger.critical(f"❌ 交易日历加载失败，无法继续: {e}")
        sys.exit(1)

    for phase_name in phases_to_run:
        if not check_dependencies(phase_name, pipeline_context["_completed_phases"]):
            logger.critical(f"❌ 阶段 {phase_name} 前置依赖未满足，终止流程")
            audit_logger.log_event("PHASE_DEPENDENCY_FAILED", {"phase": phase_name})
            sys.exit(1)

        func = load_phase_module(phase_name)
        if func is None:
            logger.error(f"❌ 阶段 {phase_name} 模块未实现或缺少 execute，跳过")
            pipeline_context["_phase_status"][phase_name] = "skipped"
            audit_logger.log_event("PHASE_SKIPPED", {"phase": phase_name, "reason": "not_implemented"})
            continue

        logger.info("=" * 60)
        logger.info(f">>> 开始执行: {phase_name} <<<")
        start_time = time.time()
        try:
            pipeline_context["_current_phase"] = phase_name
            if phase_name == "step7_fsm_backtest" and "daily_weights" not in pipeline_context:
                logger.warning("上下文缺少 daily_weights，请确保 step6 已正确生成并注入。")
            result = func(pipeline_context)
            if not isinstance(result, dict):
                raise TypeError(f"阶段 {phase_name} 必须返回 dict 类型，实际返回 {type(result)}")
            pipeline_context.update(result)
            elapsed = time.time() - start_time
            pipeline_context["_completed_phases"].add(phase_name)
            pipeline_context["_phase_status"][phase_name] = "success"
            pipeline_context["_phase_timings"][phase_name] = elapsed
            logger.info(f">>> 成功完成: {phase_name} (耗时 {elapsed:.2f} 秒) <<<")
            audit_logger.log_event("PHASE_SUCCESS", {"phase": phase_name, "elapsed_seconds": elapsed})
        except Exception as e:
            elapsed = time.time() - start_time
            logger.critical(f"❌ 阶段 {phase_name} 致命异常 (耗时 {elapsed:.2f} 秒)")
            logger.critical(traceback.format_exc())
            pipeline_context["_phase_status"][phase_name] = "failed"
            audit_logger.log_event("PHASE_FATAL", {
                "phase": phase_name,
                "exception": str(e),
                "traceback": traceback.format_exc(),
                "elapsed_seconds": elapsed
            })
            audit_logger.flush()
            logger.critical("🔴 物理熔断触发，进程终止")
            sys.exit(1)
        save_context_snapshot(pipeline_context, phase_name)

    logger.info("=" * 80)
    logger.info("🏁 QUANT-ULTRA WORKFLOW COMPLETED")
    logger.info(f"成功阶段: {pipeline_context['_completed_phases']}")
    logger.info(f"阶段状态: {pipeline_context['_phase_status']}")
    logger.info(f"总审计事件: {len(audit_logger.events)}")
    audit_logger.flush()
    logger.info("=" * 80)

if __name__ == '__main__':
    args = parse_args()
    run_pipeline(args)