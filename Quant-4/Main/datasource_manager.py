import os
import time
import logging
import queue
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set
import pandas as pd
from Main.env_config import PROJECT_ROOT
from Main.us_pipeline import fetch_us_historical, fetch_us_trading_calendar as _fetch_us_calendar
from Main.data_quality import validate_ohlcv, write_manifest

class CircuitOpenError(RuntimeError):
    pass

class FreeDataSourceManager:
    def __init__(self, cache_dir: Path = None, offline_debug: bool = False, proxy_url: str = None, source_timeout_seconds: float = 30.0, source_cooldown_seconds: float = 60.0, source_max_cooldown_seconds: float = 600.0):
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
        self._failed_symbol_retry_at: dict[str, float] = {}
        self.source_timeout_seconds = max(float(source_timeout_seconds), 0.01)
        self._disabled_sources: Set[str] = set()
        self.source_cooldown_seconds = max(float(source_cooldown_seconds), 0.01)
        self.source_max_cooldown_seconds = max(float(source_max_cooldown_seconds), self.source_cooldown_seconds)
        self._source_retry_at: dict[str, float] = {}
        self._source_timeout_counts: dict[str, int] = {}
        self._circuit_lock = threading.Lock()

    def _source_retry_remaining(self, name: str) -> float:
        with self._circuit_lock:
            remaining = self._source_retry_at.get(name, 0.0) - time.monotonic()
            if remaining <= 0:
                self._disabled_sources.discard(name)
                self._source_retry_at.pop(name, None)
                return 0.0
            return remaining

    def _bounded_source_call(self, name: str, func, *args, **kwargs):
        """Run one provider within a hard wall-clock budget and circuit-break timeouts."""
        remaining = self._source_retry_remaining(name)
        if remaining > 0:
            raise CircuitOpenError(f"{name} cooling down; retry in {remaining:.2f}s")
        results: queue.Queue = queue.Queue(maxsize=1)
        def invoke():
            try:
                results.put((True, func(*args, **kwargs)))
            except BaseException as exc:
                results.put((False, exc))
        worker = threading.Thread(target=invoke, name=f"market-source-{name}", daemon=True)
        worker.start()
        worker.join(self.source_timeout_seconds)
        if worker.is_alive():
            with self._circuit_lock:
                count = self._source_timeout_counts.get(name, 0) + 1
                self._source_timeout_counts[name] = count
                cooldown = min(self.source_cooldown_seconds * (2 ** (count - 1)), self.source_max_cooldown_seconds)
                self._disabled_sources.add(name)
                self._source_retry_at[name] = time.monotonic() + cooldown
            raise TimeoutError(f"{name} exceeded {self.source_timeout_seconds:.2f}s; retry after {cooldown:.2f}s")
        ok, value = results.get_nowait()
        if not ok:
            raise value
        with self._circuit_lock:
            self._disabled_sources.discard(name)
            self._source_retry_at.pop(name, None)
            self._source_timeout_counts.pop(name, None)
        return value

    def _write_download_audit(self, symbol: str, attempts: list, status: str) -> None:
        payload = {"symbol": symbol, "status": status, "attempts": attempts, "generated_at": datetime.now().astimezone().isoformat()}
        (self.evidence_dir / f"{symbol}_download_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
        if self.offline_debug:
            return
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
            if time.monotonic() < self._failed_symbol_retry_at.get(symbol, 0.0):
                return None
            self._failed_symbols.discard(symbol)
            self._failed_symbol_retry_at.pop(symbol, None)
        c_path = self.cache_dir / f"{symbol}_history.parquet"
        if c_path.exists():
            try:
                df = pd.read_parquet(c_path)
                df["date"] = pd.to_datetime(df["date"])
                validation = validate_ohlcv(df, symbol)
                manifests = list(self.evidence_dir.glob(f"{symbol}_*.json"))
                matching = False
                for manifest in manifests:
                    try:
                        evidence = json.loads(manifest.read_text(encoding="utf-8"))
                        matching |= evidence.get("valid") is True and evidence.get("sha256") == validation["sha256"] and int(evidence.get("rows", -1)) == len(df)
                    except (OSError, json.JSONDecodeError, ValueError, TypeError):
                        continue
                if validation["valid"] and matching:
                    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
                    cached = df.loc[mask].copy()
                    if not cached.empty:
                        return cached
                self._logger.warning("Rejected cache without matching valid evidence: %s", symbol)
            except Exception:
                c_path.unlink(missing_ok=True)
        if self.offline_debug:
            return None
        attempts = []
        for name, func in self._sources:
            retry_remaining = self._source_retry_remaining(name)
            if retry_remaining > 0:
                attempts.append({"provider": name, "status": "CIRCUIT_OPEN", "retry_after_seconds": round(retry_remaining, 3)})
                continue
            started = time.monotonic()
            try:
                df = self._bounded_source_call(name, func, symbol, self.DEFAULT_START, datetime.now().strftime("%Y-%m-%d"), freq)
                elapsed = round(time.monotonic() - started, 3)
                if df is None or df.empty:
                    attempts.append({"provider": name, "status": "EMPTY", "elapsed_seconds": elapsed})
                    continue
                if df is not None and not df.empty:
                    evidence = validate_ohlcv(df, symbol)
                    requested = df[(pd.to_datetime(df["date"]) >= pd.to_datetime(start_date)) & (pd.to_datetime(df["date"]) <= pd.to_datetime(end_date))] if "date" in df else pd.DataFrame()
                    if evidence["valid"] and requested.empty:
                        evidence["valid"] = False
                        evidence["errors"].append("no rows in requested date range")
                    evidence.update({"provider": name, "fetched_at": datetime.now().astimezone().isoformat(), "proxy_configured": bool(self.proxy_url), "elapsed_seconds": elapsed})
                    write_manifest(self.evidence_dir / f"{symbol}_{name}.json", evidence)
                    if not evidence["valid"]:
                        attempts.append({"provider": name, "status": "INCOMPLETE_OR_INVALID", "elapsed_seconds": elapsed, "errors": evidence["errors"]})
                        self._logger.warning("Rejected invalid dataset %s from %s: %s", symbol, name, evidence["errors"])
                        continue
                    df.to_parquet(c_path, index=False)
                    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
                    attempts.append({"provider": name, "status": "ACCEPTED", "elapsed_seconds": elapsed, "rows": len(df), "sha256": evidence["sha256"]})
                    self._write_download_audit(symbol, attempts, "COMPLETE")
                    self._failed_symbols.discard(symbol)
                    self._failed_symbol_retry_at.pop(symbol, None)
                    # self._logger.info(f"[OP] Query A-Share Live | [SOURCE] {name} | [RESULT] Rows: {len(df)} | [SIGNIFICANCE] Multi-source fallback success")
                    self._logger.info("由数据源管理器查询A股实时数据 | 来源: %s | 结果: 行数: %d | 意义: 多源回退成功", name, len(df))
                    return df.loc[mask].copy()
            except TimeoutError as exc:
                attempts.append({"provider": name, "status": "TIMEOUT_CIRCUIT_OPEN", "elapsed_seconds": round(time.monotonic() - started, 3), "error": str(exc)})
                self._logger.warning("Provider timeout for %s via %s; switching channel", symbol, name)
                continue
            except Exception as exc:
                attempts.append({"provider": name, "status": "ERROR", "elapsed_seconds": round(time.monotonic() - started, 3), "error": str(exc)})
                continue
        self._failed_symbols.add(symbol)
        self._failed_symbol_retry_at[symbol] = time.monotonic() + self.source_cooldown_seconds
        self._write_download_audit(symbol, attempts, "FAILED_ALL_CHANNELS")
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
            cached = pd.read_parquet(c_path)["symbol"].dropna().astype(str).unique().tolist()
            if len(cached) >= 500:
                return cached
            self._logger.warning("Rejected undersized stock-list cache: %s symbols", len(cached))
        if self.offline_debug:
            return ["600000.SH", "600036.SH", "600519.SH", "000001.SZ", "000002.SZ"]
        try:
            df = self._bounded_source_call("akshare_spot_list", self._ak.stock_zh_a_spot_em)
            syms = [f"{x}.SH" if str(x).startswith("6") else f"{x}.SZ" for x in df["代码"] if len(str(x)) == 6]
            if syms:
                pd.DataFrame({"symbol": syms}).to_parquet(c_path, index=False)
                return syms
        except: pass
        if hasattr(self, "_ak"):
            try:
                frame = self._bounded_source_call("akshare_static_list", self._ak.stock_info_a_code_name)
                code_col = next((c for c in frame.columns if "代码" in str(c) or str(c).lower() in {"code", "symbol"}), None)
                if code_col is not None:
                    syms = []
                    for raw in frame[code_col].astype(str):
                        code = raw.strip().split(".")[0].zfill(6)
                        if len(code) == 6 and code.isdigit():
                            syms.append(f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ")
                    syms = sorted(set(syms))
                    if len(syms) >= 500:
                        pd.DataFrame({"symbol": syms}).to_parquet(c_path, index=False)
                        return syms
            except Exception:
                pass
        if hasattr(self, "_bs"):
            try:
                def baostock_codes():
                    self._bs.login()
                    result = self._bs.query_all_stock(day=datetime.now().strftime("%Y-%m-%d"))
                    values = []
                    while result.next(): values.append(result.get_row_data()[0])
                    return values
                codes = []
                for raw in self._bounded_source_call("baostock_stock_list", baostock_codes):
                    exchange, code = raw.split(".", 1)
                    if len(code) == 6 and code.isdigit(): codes.append(f"{code}.{exchange.upper()}")
                syms = sorted(set(codes))
                if len(syms) >= 500:
                    pd.DataFrame({"symbol": syms}).to_parquet(c_path, index=False)
                    return syms
            except Exception:
                pass
        if hasattr(self, "_ts_pro"):
            try:
                df_ts = self._bounded_source_call("tushare_stock_list", self._ts_pro.stock_basic, list_status='L', fields='ts_code')
                if df_ts is not None and not df_ts.empty:
                    syms = [s for s in df_ts['ts_code'].tolist() if len(s.split(".")[0]) == 6 and not s.startswith("8")]
                    if syms:
                        pd.DataFrame({"symbol": syms}).to_parquet(c_path, index=False)
                        return syms
            except: pass
        if hasattr(self, "_ef"):
            try:
                df_ef = self._bounded_source_call("efinance_stock_list", self._ef.stock.get_realtime_quotes)
                if df_ef is not None and not df_ef.empty:
                    code_col = '股票代码' if '股票代码' in df_ef.columns else ('代码' if '代码' in df_ef.columns else None)
                    if code_col:
                        syms = [f"{x}.SH" if str(x).startswith("6") else f"{x}.SZ" for x in df_ef[code_col] if len(str(x)) == 6]
                        if syms:
                            pd.DataFrame({"symbol": syms}).to_parquet(c_path, index=False)
                            return syms
            except: pass
        if c_path.exists():
            cached = pd.read_parquet(c_path)["symbol"].dropna().astype(str).unique().tolist()
            if len(cached) >= 500:
                return cached
        return []

    def fetch_full_market_list(self, include_delisted: bool = True) -> List[str]:
        """Return the broad A-share research universe, including delisted codes when available."""
        active = set(self.fetch_stock_list())
        if not self.offline_debug and hasattr(self, "_ak"):
            try:
                etfs = self._bounded_source_call("akshare_etf_list", self._ak.fund_etf_spot_em)
                code_col = next((c for c in etfs.columns if "代码" in str(c) or str(c).lower() in {"code", "symbol"}), None)
                if code_col is not None:
                    for raw in etfs[code_col].astype(str):
                        code = raw.strip().split(".")[0].zfill(6)
                        if len(code) == 6 and code.isdigit():
                            active.add(f"{code}.SH" if code.startswith(("50", "51", "56", "58")) else f"{code}.SZ")
            except Exception as exc:
                self._logger.warning("Unable to extend universe with ETFs: %s", exc)
        if include_delisted and not self.offline_debug and hasattr(self, "_ak"):
            for method_name in ("stock_info_sh_delist", "stock_info_sz_delist"):
                try:
                    frame = self._bounded_source_call(f"akshare_{method_name}", getattr(self._ak, method_name))
                    code_col = next((c for c in frame.columns if "代码" in str(c) or str(c).lower() in {"code", "symbol"}), None)
                    if code_col is None:
                        continue
                    for raw in frame[code_col].astype(str):
                        code = raw.strip().split(".")[0].zfill(6)
                        if len(code) == 6 and code.isdigit():
                            active.add(f"{code}.SH" if code.startswith(("6", "9")) else f"{code}.SZ")
                except Exception as exc:
                    self._logger.warning("Unable to extend universe from %s: %s", method_name, exc)
        symbols = sorted(active)
        if symbols:
            pd.DataFrame({"symbol": symbols}).to_parquet(self.cache_dir / "full_market_universe.parquet", index=False)
        return symbols

    def fetch_asset_names(self, symbols: List[str]) -> dict[str, str]:
        """Resolve stock/ETF names from the same market-data sources, never a hard-coded list."""
        requested = {str(symbol).upper() for symbol in symbols}
        if not requested:
            return {}
        cache_path = self.cache_dir / "security_master.parquet"
        cached = pd.DataFrame(columns=["symbol", "asset_name"])
        if cache_path.exists():
            try:
                cached = pd.read_parquet(cache_path)
            except Exception:
                pass
        result = dict(zip(cached.get("symbol", pd.Series(dtype=str)).astype(str).str.upper(), cached.get("asset_name", pd.Series(dtype=str)).astype(str)))
        missing = requested - set(result)
        if missing and not self.offline_debug and hasattr(self, "_ak"):
            frames = []
            for source, method in (("stock_name_master", "stock_zh_a_spot_em"), ("etf_name_master", "fund_etf_spot_em")):
                try:
                    frames.append(self._bounded_source_call(source, getattr(self._ak, method)))
                except Exception:
                    continue
            master_rows = []
            for frame in frames:
                if frame is None or frame.empty:
                    continue
                code_col = next((c for c in frame if frame[c].astype(str).str.fullmatch(r"\\d{6}").mean() > .5), None)
                name_col = next((c for c in frame if c != code_col and frame[c].dtype == object and frame[c].astype(str).str.len().between(2, 40).mean() > .5), None)
                if code_col is None or name_col is None:
                    continue
                for code, name in zip(frame[code_col].astype(str), frame[name_col].astype(str)):
                    symbol = f"{code}.SH" if code.startswith(("5", "6", "9")) else f"{code}.SZ"
                    master_rows.append({"symbol": symbol, "asset_name": name})
            if master_rows:
                master = pd.concat([cached, pd.DataFrame(master_rows)], ignore_index=True).drop_duplicates("symbol", keep="last")
                master.to_parquet(cache_path, index=False)
                result.update(dict(zip(master["symbol"].astype(str).str.upper(), master["asset_name"].astype(str))))
        return {symbol: result.get(symbol, "名称数据不可用 / Name data unavailable") for symbol in requested}

    def fetch_trading_calendar(self, start_year: int = 2010, end_year: int = datetime.now().year) -> pd.DatetimeIndex:
        c_path = self.cache_dir / f"trading_calendar_{start_year}_{end_year}.parquet"
        if c_path.exists():
            return pd.DatetimeIndex(pd.read_parquet(c_path)["date"])
        if self.offline_debug:
            raise RuntimeError(f"Offline trading calendar cache missing: {c_path}")
        cal = self._ak.tool_trade_date_hist_sina()
        cal["trade_date"] = pd.to_datetime(cal["trade_date"])
        cal = cal[(cal["trade_date"].dt.year >= start_year) & (cal["trade_date"].dt.year <= end_year)]
        dates = cal["trade_date"].tolist()
        pd.DataFrame({"date": dates}).to_parquet(c_path, index=False)
        return pd.DatetimeIndex(dates)

    def fetch_us_trading_calendar(self, start_year: int = 2010, end_year: int = datetime.now().year) -> pd.DatetimeIndex:
        return _fetch_us_calendar(self.cache_dir, self.offline_debug, self._has_yf, getattr(self, "_yf", None), getattr(self, "_ak", None), start_year, end_year)
