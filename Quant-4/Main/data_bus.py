import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import pytz
from Main.datasource_manager import FreeDataSourceManager
from Main.market_attributes import get_free_float_market_cap, get_sector, is_marginable, compute_market_risk_aversion

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
        if self.audit_logger:
            self.audit_logger.log_event(event_type, details)

    def _handle_failure(self, method_name: str, asset: str, error: Exception, fallback_value=None):
        details = {"method": method_name, "asset": asset, "error": str(error), "fallback": fallback_value}
        self._log_audit("DATA_FETCH_FAILED", details)
        if self._strict_mode:
            raise RuntimeError(f"Data fault [{method_name}] for {asset}: {error}")
        return fallback_value

    def get_node_by_asset(self, asset: str) -> str:
        return "US_share_node" if asset.endswith(".US") else "A_share_node"

    def append_atom(self, asset: str, timestamp: datetime, value: Any, event_type: str, announcement_date: datetime):
        if event_type not in self._atom_storage:
            self._atom_storage[event_type] = []
        self._atom_storage[event_type].append({
            "asset": asset,
            "timestamp": self._ensure_ms_precision(timestamp),
            "value": value,
            "announcement_date": self._ensure_ms_precision(announcement_date)
        })

    def query_by_pit(self, asset: str, timestamp_T: datetime, field: str) -> Optional[Any]:
        t_target = self._ensure_ms_precision(timestamp_T)
        if t_target.time() < datetime.strptime("15:00:00", "%H:%M:%S").time() and asset.endswith(('.SH', '.SZ', '.BJ')):
            self._logger.warning("[OP] Validate PIT Coordinates | [SOURCE] Query Gatekeeper | [RESULT] Intraday alert for %s | [SIGNIFICANCE] Guard against look-ahead leakage", asset)
        if field == "delisting_residual":
            return self.get_delisting_residual(asset, timestamp_T)
        if field in self._atom_storage:
            valid = [r for r in self._atom_storage[field] if r["asset"] == asset and r["announcement_date"] <= t_target]
            if valid:
                valid.sort(key=lambda x: x["timestamp"])
                return valid[-1]["value"]
        if field in ["total_return_price", "close", "open", "high", "low", "volume", "amount", "adv", "actual_log_return"]:
            df = self.load_asset_history(asset, end_date=timestamp_T.strftime("%Y-%m-%d"))
            if df is not None and not df.empty:
                idx = df.index.searchsorted(pd.Timestamp(t_target).tz_localize(None), side='right') - 1
                if idx >= 0:
                    return df.iloc[idx][field] if field in df.columns else df.iloc[idx]["close"]
        return None

    def get_delisting_residual(self, asset: str, date: datetime) -> float:
        self._log_audit("TSDB_ERROR_BUS", {"event": "default_residual", "asset": asset, "date": date.strftime("%Y-%m-%d"), "fallback_value": 0.0})
        return 0.0

    def _ensure_ms_precision(self, dt: datetime) -> datetime:
        return dt.replace(microsecond=(dt.microsecond // 1000) * 1000).astimezone(self._tz)

    def load_asset_history(self, asset: str, start_date: str = None, end_date: str = None) -> Optional[pd.DataFrame]:
        if asset in self._cache:
            return self._cache[asset]
        raw = self.manager.fetch_historical(asset, start_date or "2010-01-01", end_date or datetime.now().strftime("%Y-%m-%d"))
        if raw is None or raw.empty:
            return None
        raw.set_index("date", inplace=True)
        raw.sort_index(inplace=True)
        raw["log_return"] = np.log(raw["close"] / raw["close"].shift(1))
        raw["actual_log_return"] = raw["log_return"].fillna(0.0)
        raw["total_return_price"] = raw["close"]
        raw["adv"] = raw["amount"].rolling(20, min_periods=1).mean()
        self._cache[asset] = raw
        return raw

    def fetch_benchmark_prices(self, start_date: str, end_date: str) -> pd.Series:
        df = self.manager.fetch_historical(self.get_benchmark_code(), start_date, end_date)
        if df is None or df.empty:
            return pd.Series()
        df.set_index("date", inplace=True)
        return df["close"]

    def get_benchmark_code(self) -> str:
        return "000300.SH"

    def get_free_float_market_cap(self, asset: str, date: datetime) -> float:
        return get_free_float_market_cap(self, asset, date)

    def get_sector(self, asset: str) -> str:
        return get_sector(self, asset)

    def is_marginable(self, asset: str) -> bool:
        return is_marginable(self, asset)

    def compute_market_risk_aversion(self, end_date: str, window_years=5) -> float:
        return compute_market_risk_aversion(self, end_date, window_years)

    def get_universe(self, refresh: bool = False) -> List[str]:
        if self._universe is None or refresh:
            a_universe = self.manager.fetch_stock_list()
            us_universe = ["AAPL.US", "MSFT.US", "AMZN.US", "GOOG.US", "NVDA.US"]
            self._universe = a_universe + us_universe
        return self._universe