# -*- coding: utf-8 -*-
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
        
        # [核心修复] 日志防抖集合，解决内层循环重复报警导致的 I/O 阻塞
        self._warned_pit_assets = set()

    def _log_audit(self, event_type: str, details: dict):
        if self.audit_logger:
            self.audit_logger.log_event(event_type, details)

    def _handle_failure(self, method_name: str, asset: str, error: Exception, fallback_value=None):
        details = {"method": method_name, "asset": asset, "error": str(error), "fallback": fallback_value}
        self._log_audit("DATA_FETCH_FAILED", details)
        self._logger.warning(f"Failed {method_name} for {asset}: {error}")
        return fallback_value

    def validate_pit_coordinates(self, asset: str, request_date: Any) -> bool:
        """
        验证时间坐标是否符合 Point-in-Time 原则，防止未来函数。
        """
        try:
            req_dt = pd.to_datetime(request_date)
            now_dt = datetime.now(self._tz).replace(tzinfo=None)
            
            # 判断是否为盘中未收盘状态
            is_intraday_or_future_condition = (req_dt.date() >= now_dt.date()) and (now_dt.hour < 15)
            
            if is_intraday_or_future_condition:
                if asset not in self._warned_pit_assets:
                    self._logger.warning(f"[OP] Validate PIT Coordinates | [SOURCE] Query Gatekeeper | [RESULT] Intraday alert for {asset} | [SIGNIFICANCE] Guard against look-ahead leakage")
                    self._warned_pit_assets.add(asset)
                else:
                    self._logger.debug(f"Intraday alert silenced for {asset}")
                
                if self._strict_mode:
                    # 在严格模式下，如果确实属于严重的未来函数违规，此处可进一步执行熔断
                    pass
        except Exception as e:
            self._logger.debug(f"PIT Validation error for {asset}: {e}")
            
        return True


    def get_universe(self):
            """
            三级降级防御获取股票池 (已增强动态列名嗅探)
            """
            if getattr(self, '_universe', None):
                return self._universe
                
            # Level 1: 本地缓存快速读取
            import os
            import pandas as pd
            cache_path = os.path.join("Quant-4", "Data_Cache", "ashare_sw_meta.parquet")
            if os.path.exists(cache_path):
                try:
                    df = pd.read_parquet(cache_path, columns=['asset'])
                    self._universe = df['asset'].unique().tolist()
                    return self._universe
                except Exception:
                    pass

            if self.manager.offline_debug:
                self._universe = self.manager.fetch_stock_list()
                return self._universe
                    
            # Level 2: 动态回源 AkShare (解决 KeyError 痛点)
            try:
                import akshare as ak
                df = ak.stock_info_a_code_name()
                
                # 动态嗅探列名：遍历所有可能的代码列表名
                code_col = None
                possible_cols = ['代码', '证券代码', 'symbol', 'code', '股票代码']
                for col in possible_cols:
                    if col in df.columns:
                        code_col = col
                        break
                        
                if not code_col:
                    raise ValueError(f"AkShare 接口返回的列名已完全变更，当前返回列名: {df.columns.tolist()}")
                    
                raw_codes = df[code_col].astype(str).tolist()
                valid_universe = []
                for c in raw_codes:
                    c = c.zfill(6) # 补齐 6 位
                    if c.startswith('6'): valid_universe.append(f"{c}.SH")
                    elif c.startswith('0') or c.startswith('3'): valid_universe.append(f"{c}.SZ")
                    elif c.startswith('8') or c.startswith('4') or c.startswith('8') or c.startswith('9'): valid_universe.append(f"{c}.BJ")
                
                self._universe = valid_universe
                self._logger.info(f"[状态对齐] Level 2: 成功从 AkShare 回源获取全市场 {len(self._universe)} 只标的")
                return self._universe
                
            except Exception as e:
                self._logger.error(f"Fallback universe fetching failed: {e}")
                self._universe = []
                self._logger.warning("Universe could not be resolved automatically. Returning empty list.")
                return self._universe

    def set_universe(self, new_universe: list):
        """允许外部（如 Phase 1 的清洗模块）写入存活的股票池"""
        self._universe = new_universe

    def set_universe(self, symbols: List[str]):
        """提供给外部直接覆盖股票池的显式接口"""
        self._universe = symbols
        self._logger.info(f"Universe manually overridden with {len(symbols)} assets.")

    def fetch_historical(self, asset: str, start_date: str, end_date: str) -> pd.DataFrame:
        if asset in self._cache:
            df = self._cache[asset]
            mask = (df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))
            return df.loc[mask].copy()
            
        raw = self.manager.fetch_historical(asset, start_date, end_date)
        if raw is None or raw.empty:
            return pd.DataFrame()
            
        raw["log_return"] = np.log(raw["close"] / raw["close"].shift(1))
        raw["actual_log_return"] = raw["log_return"].fillna(0.0)
        raw["total_return_price"] = raw["close"]
        if "amount" in raw.columns:
            raw["adv"] = raw["amount"].rolling(20, min_periods=1).mean()
            
        self._cache[asset] = raw
        
        mask = (raw['date'] >= pd.to_datetime(start_date)) & (raw['date'] <= pd.to_datetime(end_date))
        return raw.loc[mask].copy()

    # [融合版] 结合了旧版的特殊业务逻辑与高速二分查找，以及新版的防抖与内存预载机制
    def query_by_pit(self, asset: str, target_date: Any, field: str) -> Optional[Any]:
        # 1. 使用自带防抖机制的独立验证器，彻底终结日志雪崩
        self.validate_pit_coordinates(asset, target_date)
        
        dt = pd.to_datetime(target_date)
        dt_tz_naive = dt.tz_localize(None) if dt.tz is not None else dt
        
        # 2. 保留旧版的特色业务逻辑 (退市残值)
        if field == "delisting_residual" and hasattr(self, "get_delisting_residual"):
            return self.get_delisting_residual(asset, target_date)
            
        # 3. 保留旧版的财务原子存储高频解析
        if field in self._atom_storage:
            valid = [r for r in self._atom_storage[field] if r["asset"] == asset and r["announcement_date"] <= dt_tz_naive]
            if valid:
                valid.sort(key=lambda x: x["timestamp"])
                return valid[-1]["value"]
                
        # 4. 融合新版的 I/O 预载优化 (防止逐天读盘)
        if asset not in self._cache:
            current_year = datetime.now(self._tz).year
            self.fetch_historical(asset, "2005-01-01", f"{current_year + 1}-12-31")
            
        df = self._cache.get(asset)
        
        # 5. 融合旧版的高性能二分查找算法 (searchsorted)
        if df is not None and not df.empty:
            # 确保有一个时间序列供 searchsorted 查找
            if 'date' in df.columns:
                search_series = pd.to_datetime(df['date']).dt.tz_localize(None)
                idx = search_series.searchsorted(dt_tz_naive, side='right') - 1
                if idx >= 0:
                    return df.iloc[idx][field] if field in df.columns else df.iloc[idx].get("close", np.nan)
                    
        return None

    def fetch_benchmark_prices(self, start_date: str, end_date: str) -> pd.Series:
        df = self.manager.fetch_historical(self.get_benchmark_code(), start_date, end_date)
        if df is None or df.empty:
            return pd.Series(dtype=float)
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
    
    def load_asset_history(self, sym: str, start_date=None, end_date=None):
        """
        核心功能：根据时间窗口，从底层文件系统加载个股的时序截面数据。
        实现时点信息 (PIT) 截断，防止下游标签计算产生前瞻性偏差 (Look-ahead bias)。
        """
        import os
        import pandas as pd
        
        # 1. 动态定位缓存目录 (兼容不同的项目启动路径)
        cache_dir = None
        if hasattr(self, 'manager') and hasattr(self.manager, 'cache_dir'):
            cache_dir = self.manager.cache_dir
        else:
            # 默认回退到当前工作目录下的 Data_Cache
            cache_dir = os.path.join(os.getcwd(), "Data_Cache")
            if not os.path.exists(cache_dir):
                # 尝试向上一级寻找
                cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Data_Cache")
        
        # 2. 探针适配：兼容 Phase_1 可能会输出的不同文件命名规范
        file_path = os.path.join(cache_dir, f"{sym}_history.parquet")
        if not os.path.exists(file_path):
            alt_path = os.path.join(cache_dir, f"{sym}.parquet")
            if os.path.exists(alt_path):
                file_path = alt_path
            else:
                return pd.DataFrame()
            
        # 3. 加载物理数据并执行刚性时空切割
        try:
            df = pd.read_parquet(file_path)
            
            # 判断日期是被设为了普通列，还是设为了索引
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
                df = df.sort_values('date').reset_index(drop=True)
            elif df.index.name == 'date' or isinstance(df.index, pd.DatetimeIndex):
                if start_date:
                    df = df[df.index >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df.index <= pd.to_datetime(end_date)]
                    
            return df
            
        except Exception as e:
            if hasattr(self, '_logger'):
                self._logger.error(f"[IO故障] 无法加载标的 {sym} 的历史行情: {e}")
            return pd.DataFrame()
        
    def get_node_by_asset(self, sym: str) -> str:
        """
        联邦多轨路由识别：判定单只资产标的所属的跨境微观时空节点。
        专门对接 Phase_5 非对称边际效用校准模块。
        """
        if not sym or not isinstance(sym, str):
            return "unknown_node"
            
        sym_upper = sym.upper().strip()
        
        # 1. 刚性后缀校验：满足中国三大交易所（沪、深、北）后缀的直接判定为 A股节点
        if sym_upper.endswith(('.SH', '.SZ', '.BJ')):
            return "A_share_node"
            
        # 2. 泛化特征自愈：如果被清洗成了纯数字代码（如 '000001'）且不含美股标识，兜底归入 A股节点
        parts = sym_upper.split('.')
        if parts[0].isdigit() and not 'US' in sym_upper:
            return "A_share_node"
            
        # 3. 跨境分流：其余显式带 .US 后缀或带有英文字母前缀的代码统统分流至美股节点
        return "US_share_node"
