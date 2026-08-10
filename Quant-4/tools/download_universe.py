"""Incremental, resumable bulk download of the survivorship-free A-share
universe daily bars into ``Data_Cache``.

Universe: every A-share stock ever listed inside the backtest window (from the
baostock master list, including delisted names) plus the audited ETF universe.

Sources, in order of preference:
  1. akshare/Eastmoney ``stock_zh_a_hist`` (qfq) - fastest for live names;
  2. akshare/Tencent ``stock_zh_a_hist_tx`` (qfq) - fallback for live names;
  3. baostock ``query_history_k_data_plus`` (qfq) - delisted names, slow.

The script is resume-safe: symbols with a complete-enough cache are skipped and
progress/failures are persisted, so a later run continues exactly where the
previous one stopped. Run repeatedly (or as a background job) until the
coverage ratio reported by ``universe_coverage`` reaches 1.0.
"""
from __future__ import annotations

import argparse
import json
import logging
import msvcrt
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = PROJECT_ROOT / "Data_Cache"

sys.path.insert(0, str(PROJECT_ROOT))
from Main.pit_universe import (  # noqa: E402
    MASTER_REL,
    ETF_UNIVERSE,
    ever_alive_stocks,
    fetch_master_list,
    load_master_list,
    stock_out_date,
)

logger = logging.getLogger("download_universe")

_OUT_COLUMNS = ["date", "open", "high", "low", "close", "volume", "amount"]


def _to_engine_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize any provider frame to the engine cache schema (date index)."""
    rename = {
        "日期": "date", "开盘": "open", "最高": "high", "最低": "low",
        "收盘": "close", "成交量": "volume", "成交额": "amount",
    }
    df = df.rename(columns=rename)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    out = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    for col in ("open", "high", "low", "close", "volume", "amount"):
        if col not in out.columns:
            out[col] = np.nan
    return out[["date", "open", "high", "low", "close", "volume", "amount"]]


def fetch_eastmoney(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    import akshare as ak

    code = symbol.split(".")[0]
    df = ak.stock_zh_a_hist(
        symbol=code, period="daily",
        start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="qfq",
    )
    if df is None or df.empty:
        return None
    return _to_engine_frame(df)


def fetch_tencent(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    import akshare as ak

    code, exchange = symbol.split(".")
    df = ak.stock_zh_a_hist_tx(
        symbol=f"{exchange.lower()}{code}",
        start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="qfq",
    )
    if df is None or df.empty:
        return None
    frame = _to_engine_frame(df)
    # Tencent reports volume in lots (100 shares) and omits amount; approximate
    # amount = volume * 100 * close (used only for the liquidity filter).
    if frame["amount"].isna().all():
        frame["amount"] = frame["volume"] * 100.0 * frame["close"]
    return frame


def fetch_sina(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    import akshare as ak

    code, exchange = symbol.split(".")
    df = ak.stock_zh_a_daily(
        symbol=f"{exchange.lower()}{code}",
        start_date=start.replace("-", ""), end_date=end.replace("-", ""), adjust="qfq",
    )
    if df is None or df.empty:
        return None
    return _to_engine_frame(df)


def fetch_baostock(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    import baostock as bs

    code, exchange = symbol.split(".")
    login = bs.login()
    if login.error_code != "0":
        return None
    try:
        rs = bs.query_history_k_data_plus(
            code=f"{exchange.lower()}.{code}",
            fields="date,open,high,low,close,volume,amount",
            start_date=start, end_date=end, frequency="d", adjustflag="2",
        )
        rows = []
        while (rs.error_code == "0") and rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        frame = pd.DataFrame(rows, columns=_OUT_COLUMNS)
        return _to_engine_frame(frame)
    finally:
        bs.logout()


class _SourceBreaker:
    """Trip a flaky source for the rest of the process after repeated failures,
    so a blocked provider (e.g. Eastmoney under rate limit) stops eating the
    per-symbol timeout budget."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.failures = 0
        self.disabled = False

    def note_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.disabled = True

    def note_success(self) -> None:
        self.failures = 0


def download_one(
    symbol: str, start: str, end: str, timeout: float = 60.0,
    allow_tencent: bool = True, breaker: Optional[_SourceBreaker] = None,
) -> Optional[pd.DataFrame]:
    """Try sources in order with a hard wall-clock budget per symbol."""

    def _run(fn):
        import threading as _t

        box: dict = {}

        def worker():
            try:
                box["df"] = fn()
            except BaseException as exc:  # noqa: BLE001 - recorded, not raised
                box["exc"] = exc

        t = _t.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise TimeoutError(f"{symbol} exceeded {timeout}s")
        if "exc" in box:
            raise box["exc"]
        return box.get("df")

    if breaker is None or not breaker.disabled:
        try:
            df = _run(lambda: fetch_eastmoney(symbol, start, end))
            if df is not None and not df.empty:
                if breaker is not None:
                    breaker.note_success()
                return df
        except Exception:
            if breaker is not None:
                breaker.note_failure()
            time.sleep(0.3)
    try:
        df = _run(lambda: fetch_sina(symbol, start, end))
        if df is not None and not df.empty:
            return df
    except Exception:
        time.sleep(0.5)
    if allow_tencent:
        try:
            df = _run(lambda: fetch_tencent(symbol, start, end))
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
    return None


def cached_complete(
    cache_dir: Path, symbol: str, start: str, end: str, ipo_date: Optional[pd.Timestamp],
    tolerance_days: int = 20, lookback_days: int = 400,
) -> bool:
    """True when the cached parquet covers what the backtest actually needs.

    The backtest starts 2016-01-01 and its factors need at most ~250 trading
    days of lookback, so history from ``start + lookback_days`` onward is
    sufficient. Several free sources (e.g. Sina's kline endpoint) only return
    ~12 years of history even for names listed much earlier; requiring the
    full ``start`` window would make those files permanently 'incomplete' and
    trigger an endless re-download of identical data.
    """
    path = cache_dir / f"{symbol}_history.parquet"
    if not path.exists():
        return False
    try:
        df = pd.read_parquet(path, columns=["date"])
        if df.empty:
            return False
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        first, last = dates.min(), dates.max()
    except Exception:
        return False
    target_end = pd.Timestamp(end)
    if last < target_end - pd.Timedelta(days=tolerance_days):
        return False
    if ipo_date is not None and pd.notna(ipo_date) and ipo_date > pd.Timestamp(start):
        return first <= ipo_date + pd.Timedelta(days=tolerance_days)
    required_first = pd.Timestamp(start) + pd.Timedelta(days=lookback_days)
    return first <= required_first


def _save(cache_dir: Path, symbol: str, df: pd.DataFrame) -> None:
    path = cache_dir / f"{symbol}_history.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.parquet")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class _DownloadLock:
    """Single-instance guard so repeated launches never duplicate the job.

    Uses an atomic Windows byte-range lock (msvcrt) instead of a pid file, so
    the launcher/interpreter pair race cannot let two instances run at once."""

    def __init__(self, cache_dir: Path):
        self.path = cache_dir / "universe" / "download.lock"
        self._fh = None

    def acquire(self) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self.path, "a+")
            msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            self._fh.seek(0)
            self._fh.truncate()
            self._fh.write(str(os.getpid()))
            self._fh.flush()
            return True
        except OSError:
            if self._fh is not None:
                try:
                    self._fh.close()
                except OSError:
                    pass
            return False

    def owns(self) -> bool:
        """True while this process still owns the lock."""
        try:
            return self.path.exists() and self.path.read_text().strip() == str(os.getpid())
        except Exception:
            return False

    def release(self) -> None:
        if self._fh is not None:
            try:
                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None
        try:
            self.path.unlink(missing_ok=True)
        except OSError:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DATA_CACHE)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=0, help="0 = all targets")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--only-delisted", action="store_true")
    parser.add_argument("--allow-tencent", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-master", action="store_true")
    args = parser.parse_args()

    if args.end is None:
        args.end = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    lock = _DownloadLock(args.cache_dir)
    if not lock.acquire():
        print("another download_universe instance is already running; exiting")
        return 3
    try:
        return _run_main(args, lock)
    finally:
        lock.release()


def _run_main(args, lock) -> int:
    if not (args.cache_dir / MASTER_REL).exists() or args.force_master:
        fetch_master_list(args.cache_dir)
    master = load_master_list(args.cache_dir)
    if master.empty:
        logger.error("Empty master list; cannot build universe")
        return 2

    backtest_start = "2016-01-01"
    stocks = ever_alive_stocks(master, backtest_start, args.end)
    out_dates = stock_out_date(master)
    targets: List[str]
    if args.only_delisted:
        targets = [s for s in stocks if out_dates.get(s) is not None]
    else:
        targets = stocks + list(ETF_UNIVERSE)

    to_do: List[str] = []
    skipped = 0
    out_dates_map = {s: (out_dates.get(s) or pd.Timestamp(args.end)) for s in targets}
    for symbol in targets:
        symbol_end = out_dates_map[symbol].strftime("%Y-%m-%d")
        tolerance = 200 if out_dates.get(symbol) is not None else 20
        if cached_complete(
            args.cache_dir, symbol, args.start, symbol_end,
            master.loc[master["symbol"] == symbol, "ipo_date"].iloc[0]
            if (master["symbol"] == symbol).any() else None,
            tolerance_days=tolerance,
        ):
            skipped += 1
            continue
        to_do.append(symbol)
    if args.limit > 0:
        to_do = to_do[: args.limit]

    print(
        f"targets={len(targets)} skipped_cached={skipped} to_download={len(to_do)} "
        f"(delisted_only={args.only_delisted}, workers={args.workers})"
    )
    if args.dry_run:
        print("dry-run: no downloads started")
        return 0

    failures: Dict[str, str] = {}
    done = 0
    lock = threading.Lock()
    breaker = _SourceBreaker(threshold=3)
    t0 = time.time()
    progress_path = args.cache_dir / "universe" / "download_progress.json"
    failures_path = args.cache_dir / "universe" / "download_failures.json"

    def _progress():
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (len(to_do) - done) / rate / 60 if rate > 0 else float("nan")
        _write_json(progress_path, {
            "updated_at": datetime.now().astimezone().isoformat(),
            "to_download": len(to_do), "done": done, "failed": len(failures),
            "rate_per_s": round(rate, 3), "eta_minutes": round(eta, 1) if eta == eta else None,
        })

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                download_one, symbol, args.start,
                out_dates_map[symbol].strftime("%Y-%m-%d"),
                args.timeout, args.allow_tencent, breaker,
            ): symbol
            for symbol in to_do
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
            except Exception as exc:  # noqa: BLE001
                df = None
                failures[symbol] = f"{type(exc).__name__}: {exc}"
            if df is not None and len(df):
                try:
                    _save(args.cache_dir, symbol, df)
                except Exception as exc:  # noqa: BLE001
                    failures[symbol] = f"save: {exc}"
            elif symbol not in failures:
                failures[symbol] = "empty"
            with lock:
                done += 1
            if done % 25 == 0 or done == len(to_do):
                _progress()
                print(
                    f"[{done}/{len(to_do)}] ok={done - len(failures)} failed={len(failures)} "
                    f"elapsed={time.time() - t0:.0f}s", flush=True,
                )

    _progress()
    _write_json(failures_path, failures)
    print(f"finished: ok={len(to_do) - len(failures)} failed={len(failures)}")
    print(f"failures: {failures_path}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
