"""
Quant-Ultra Flow - PIT Data Bus
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import pytz
from Main.datasource_manager import FreeDataSourceManager

class PITDataBus:
    def __init__(self, data_manager: FreeDataSourceManager, audit_logger=None, strict_mode: bool = True, tz=pytz.timezone("Asia/Shanghai")):
        self.manager = data_manager
        self._tz = tz
        self._cache: Dict[str, pd.DataFrame] = {}
        self._atom_storage: Dict[str, List[Dict]] = {}
        self._universe: Optional[List[str]] = None
        self._logger = logging.getLogger("PITDataBus")
        self.audit_logger = audit_logger
        self._strict_mode = strict_mode
        self._mcap_cache, self._sector_cache, self._margin_cache, self._risk_aversion_cache = {}, {}, {}, {}

    def _log_audit(self, event_type: str, details: dict):
        if self.audit_logger: self.audit_logger.log_event(event_type, details)

    def _handle_failure(self, method_name: str, asset: str, error: Exception, fallback_value=None):
        details = {"method": method_name, "asset": asset, "error": str(error), "fallback": fallback_value}
        self._log_audit("DATA_FETCH_FAILED", details)
        if self._strict_mode: raise RuntimeError(f"数据获取失败 [{method_name}] {asset}: {error}")
        return fallback_value

    def append_atom(self, asset: str, timestamp: datetime, value: Any, event_type: str, announcement_date: datetime):
        if event_type not in self._atom_storage: self._atom_storage[event_type] = []
        self._atom_storage[event_type].append({
            "asset": asset, "timestamp": self._ensure_ms_precision(timestamp), "value": value,
            "announcement_date": self._ensure_ms_precision(announcement_date)
        })

    def query_by_pit(self, asset: str, timestamp_T: datetime, field: str) -> Optional[Any]:
        t_target = self._ensure_ms_precision(timestamp_T)
        if field in self._atom_storage:
            valid = [r for r in self._atom_storage[field] if r["asset"] == asset and r["announcement_date"] <= t_target]
            if valid:
                valid.sort(key=lambda x: x["timestamp"])
                return valid[-1]["value"]
        if field in ["total_return_price", "close", "open", "high", "low", "volume", "amount", "adv"]:
            df = self.load_asset_history(asset, end_date=timestamp_T.strftime("%Y-%m-%d"))
            if df is not None and not df.empty:
                idx = df.index.searchsorted(pd.Timestamp(t_target).tz_localize(None), side='right') - 1
                if idx >= 0: return df.iloc[idx][field] if field in df.columns else df.iloc[idx]["close"]
        return None

    def _ensure_ms_precision(self, dt: datetime) -> datetime:
        return dt.replace(microsecond=(dt.microsecond // 1000) * 1000).astimezone(self._tz)

    def load_asset_history(self, asset: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        if asset in self._cache:
            df = self._cache[asset]
            if end_date: df = df[df.index <= pd.Timestamp(end_date)]
            if start_date: df = df[df.index >= pd.Timestamp(start_date)]
            return df
        raw = self.manager.fetch_historical(asset, start_date or "2010-01-01", end_date or datetime.now().strftime("%Y-%m-%d"))
        if raw is None or raw.empty: return None
        raw.set_index("date", inplace=True); raw.sort_index(inplace=True)
        raw["log_return"] = np.log(raw["close"] / raw["close"].shift(1))
        raw["total_return_price"] = raw["close"]
        raw["adv"] = raw["amount"].rolling(20, min_periods=1).mean()
        self._cache[asset] = raw
        return raw

    def fetch_benchmark_prices(self, start_date: str, end_date: str) -> pd.Series:
        df = self.manager.fetch_historical("000300.SH", start_date, end_date)
        if df is None or df.empty: return pd.Series()
        df.set_index("date", inplace=True)
        return df["close"]

    def get_benchmark_code(self) -> str: return "000300.SH"

    def get_free_float_market_cap(self, asset: str, date: datetime) -> float:
        cache_key = f"{asset}_{date.strftime('%Y%m%d')}"
        if cache_key in self._mcap_cache: return self._mcap_cache[cache_key]
        try:
            code = f"sh.{asset.split('.')[0]}" if asset.endswith('.SH') else f"sz.{asset.split('.')[0]}"
            rs = self.manager._bs.query_history_k_data_plus(code=code, fields="date,free_float,close", start_date=date.strftime("%Y-%m-%d"), end_date=date.strftime("%Y-%m-%d"), adjustflag="2")
            if rs is None or rs.error_code != "0" or not rs.next(): raise Exception("BaoStock 错误")
            row = rs.get_row_data()
            mcap = (float(row[1]) * float(row[2])) / 1e4
            self._mcap_cache[cache_key] = mcap
            return mcap
        except Exception as e:
            return self._handle_failure("get_free_float_market_cap", asset, e, fallback_value=0.0)

    def get_sector(self, asset: str) -> str:
        if asset in self._sector_cache: return self._sector_cache[asset]
        try:
            df = self.manager._ak.stock_industry_sw()
            row = df[df["股票代码" if "股票代码" in df.columns else "代码"] == asset.split('.')[0]]
            for col in ['申万行业', '行业', '申万一级行业']:
                if col in row.columns:
                    self._sector_cache[asset] = row.iloc[0][col]
                    return row.iloc[0][col]
            raise Exception("未知行业")
        except Exception as e:
            return self._handle_failure("get_sector", asset, e, fallback_value="未知")

    def is_marginable(self, asset: str) -> bool:
        if asset in self._margin_cache: return self._margin_cache[asset]
        try:
            all_codes = set(self.manager._ak.stock_margin_sse(start_date="", end_date="")['证券代码']) | set(self.manager._ak.stock_margin_sz(start_date="", end_date="")['证券代码'])
            is_margin = asset.split('.')[0] in all_codes
            self._margin_cache[asset] = is_margin
            return is_margin
        except Exception as e:
            return self._handle_failure("is_marginable", asset, e, fallback_value=False)

    def get_short_rate(self, asset: str) -> float: return 0.08 / 252

    def compute_market_risk_aversion(self, end_date: str, window_years=5) -> float:
        cache_key = f"{end_date}_{window_years}"
        if cache_key in self._risk_aversion_cache: return self._risk_aversion_cache[cache_key]
        try:
            end = datetime.strptime(end_date, "%Y-%m-%d")
            df = self.manager.fetch_historical("000300.SH", (end - timedelta(days=window_years*365)).strftime("%Y-%m-%d"), end_date)
            df.set_index("date", inplace=True)
            rets = np.log(df["close"] / df["close"].shift(1)).resample('W').last()
            lambda_mkt = (rets.mean() * 52 - 0.025) / ((rets.std() * np.sqrt(52)) ** 2)
            self._risk_aversion_cache[cache_key] = lambda_mkt
            return lambda_mkt
        except Exception as e:
            return self._handle_failure("compute_market_risk_aversion", "market", e, fallback_value=0.02)

    def get_universe(self, refresh: bool = False) -> List[str]:
        if self._universe is None or refresh: self._universe = self.manager.fetch_stock_list()
        return self._universe