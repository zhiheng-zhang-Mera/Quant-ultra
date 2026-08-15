"""Fast, resumable bulk downloader for the PIT A-share universe.

Optimized for networks where the Eastmoney endpoint is blocked (the default
``download_universe.py`` stalls on it): the primary channel is the Tencent
forward-adjusted kline API via direct HTTP paging (~1.1s per 800-row page,
~5s per 12-year symbol), with baostock as the fallback for names Tencent no
longer serves. Everything else (resume-safety, progress persistence,
single-instance lock, engine cache schema) reuses ``download_universe``.

Usage:
    python tools/download_universe_fast.py [--workers 10] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
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
import requests

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
from tools.download_universe import (  # noqa: E402
    _DownloadLock,
    _OUT_COLUMNS,
    _save,
    cached_complete,
    fetch_baostock,
)

logger = logging.getLogger("download_universe_fast")
_HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_SINA_HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Referer": "https://finance.sina.com.cn"}
_PAGE = 800
_SINA_PAGE = 1023
_BS_LOCK = threading.Lock()  # baostock's global socket session is not thread-safe


class ChannelBreaker:
    """Trip a channel after repeated failures; auto-recover after cooldown."""

    def __init__(self, threshold: int = 4, cooldown_s: float = 600.0):
        self.threshold = threshold
        self.cooldown = cooldown_s
        self.failures = 0
        self.blocked_until = 0.0

    def ok(self) -> bool:
        if self.failures == 0:
            return True
        if time.time() >= self.blocked_until:
            self.failures = 0
            return True
        return False

    def note_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.threshold:
            self.blocked_until = time.time() + self.cooldown

    def note_success(self) -> None:
        self.failures = 0


def _is_anti_bot(resp) -> bool:
    """True when a provider returns its anti-crawler challenge (non-JSON/501)."""
    try:
        resp.json()
        return False
    except Exception:
        return True


def fetch_sina_paged(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Sina CN_MarketDataService daily kline, back-walking 1023-row pages.

    Volume is in shares and no amount is given; approximate amount =
    volume * close (liquidity filter only).
    """
    code, exchange = symbol.split(".")
    px = f"{exchange.lower()}{code}"
    start_ts = pd.Timestamp(start)
    rows: List[list] = []
    page_end = pd.Timestamp(end)
    seen = set()
    for _ in range(20):
        url = (
            "https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData"
            f"?symbol={px}&scale=240&ma=no&datalen={_SINA_PAGE}"
        )
        try:
            resp = requests.get(url, headers=_SINA_HDR, timeout=20)
            if resp.status_code != 200 or _is_anti_bot(resp):
                return None
            page = resp.json()
        except Exception:
            return None
        if not isinstance(page, list) or not page:
            break
        # filter rows strictly before page_end (the API returns the most recent
        # N rows relative to the request moment; paging walks back by date)
        page = [r for r in page if pd.Timestamp(r.get("day")) < page_end]
        page = sorted(page, key=lambda r: r.get("day"))
        new_rows = [r for r in page if r.get("day") not in seen]
        if not new_rows:
            break
        rows = new_rows + rows
        seen.update(r.get("day") for r in new_rows)
        first = pd.Timestamp(new_rows[0]["day"])
        if len(page) < _SINA_PAGE or first <= start_ts:
            break
        page_end = first - pd.Timedelta(days=1)
        time.sleep(0.15)
    if not rows:
        return None
    frame = pd.DataFrame(
        {"date": [r["day"] for r in rows], "open": [r["open"] for r in rows],
         "high": [r["high"] for r in rows], "low": [r["low"] for r in rows],
         "close": [r["close"] for r in rows], "volume": [r["volume"] for r in rows]}
    )
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    for col in ("open", "high", "low", "close", "volume"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["amount"] = frame["volume"] * frame["close"]
    return frame[["date", "open", "high", "low", "close", "volume", "amount"]]


def fetch_tencent_paged(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Full qfq daily history via Tencent fqkline with back-walking paging.

    Names whose forward-adjusted (qfq) series goes negative (huge historical
    dividends, e.g. Vanke) are refetched backward-adjusted (hfq): hfq is always
    positive and, like qfq, preserves period returns - the engine only consumes
    per-name returns, so either adjustment is internally consistent.
    """
    code, exchange = symbol.split(".")
    px = f"{exchange.lower()}{code}"
    start_ts = pd.Timestamp(start)

    def _fetch(adjust: str) -> Optional[pd.DataFrame]:
        rows: List[list] = []
        page_end = pd.Timestamp(end)
        seen = set()
        for _ in range(30):  # hard cap on pages per symbol
            url = (
                f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
                f"?param={px},day,2010-01-01,{page_end.strftime('%Y-%m-%d')},{_PAGE},{adjust}"
            )
            try:
                resp = requests.get(url, headers=_HDR, timeout=20)
                data = resp.json().get("data", {}).get(px, {})
            except Exception:
                return None
            page = data.get(f"{adjust}day") or data.get("day") or []
            if not page:
                break
            # Tencent attaches a dividend-info dict as a 7th cell on ex-dividend
            # days; the kline itself is the first 6 cells (date,o,c,h,l,volume).
            page = [r[:6] for r in page]
            new_rows = [r for r in page if r[0] not in seen]
            rows = new_rows + rows
            seen.update(r[0] for r in new_rows)
            first = pd.Timestamp(page[0][0])
            if len(page) < _PAGE or first <= start_ts:
                break
            page_end = first - pd.Timedelta(days=1)
            time.sleep(0.05)
        if not rows:
            return None
        frame = pd.DataFrame(rows, columns=["date", "open", "close", "high", "low", "volume"])
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
        for col in ("open", "high", "low", "close", "volume"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        # Tencent reports volume in lots (100 shares) and omits amount; approximate
        # amount = volume * 100 * close (used only for the liquidity filter).
        frame["amount"] = frame["volume"] * 100.0 * frame["close"]
        if (frame["close"] <= 0).any() or (frame["open"] <= 0).any():
            return None  # negative adjusted prices -> caller retries hfq
        return frame[["date", "open", "high", "low", "close", "volume", "amount"]]

    frame = _fetch("qfq")
    if frame is None:
        logger.debug("%s qfq negative/invalid; retrying hfq", symbol)
        frame = _fetch("hfq")
    return frame


def download_one_fast(symbol: str, start: str, end: str, timeout: float = 45.0) -> Optional[pd.DataFrame]:
    """Tencent first, baostock fallback (Eastmoney is blocked on this network)."""
    df = None
    try:
        box: dict = {}

        def _run(fn):
            import threading as _t

            res: dict = {}

            def worker():
                try:
                    res["df"] = fn()
                except BaseException as exc:  # noqa: BLE001
                    res["exc"] = exc

            t = _t.Thread(target=worker, daemon=True)
            t.start()
            t.join(timeout)
            if t.is_alive():
                raise TimeoutError(f"{symbol} exceeded {timeout}s")
            if "exc" in res:
                raise res["exc"]
            return res.get("df")

        df = _run(lambda: fetch_tencent_paged(symbol, start, end))
    except Exception as exc:  # noqa: BLE001
        logger.debug("tencent failed for %s: %s", symbol, exc)
    if df is None or df.empty:
        try:
            df = fetch_baostock(symbol, start, end)
        except Exception as exc:  # noqa: BLE001
            logger.debug("baostock failed for %s: %s", symbol, exc)
    return df


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
    targets: List[str] = stocks + list(ETF_UNIVERSE)
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
        f"(workers={args.workers})",
        flush=True,
    )
    if args.dry_run:
        return 0

    failures: Dict[str, str] = {}
    done = 0
    lock = threading.Lock()
    t0 = time.time()
    progress_path = args.cache_dir / "universe" / "download_progress_fast.json"
    failures_path = args.cache_dir / "universe" / "download_failures_fast.json"

    def _progress():
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (len(to_do) - done) / rate / 60 if rate > 0 else float("nan")
        (args.cache_dir / "universe").mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps({
                "updated_at": datetime.now().astimezone().isoformat(),
                "to_download": len(to_do), "done": done, "failed": len(failures),
                "rate_per_s": round(rate, 3),
                "eta_minutes": round(eta, 1) if eta == eta else None,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                download_one_fast, symbol, args.start,
                out_dates_map[symbol].strftime("%Y-%m-%d"), args.timeout,
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
            if done % 50 == 0 or done == len(to_do):
                _progress()
                print(
                    f"[{done}/{len(to_do)}] ok={done - len(failures)} failed={len(failures)} "
                    f"elapsed={time.time() - t0:.0f}s", flush=True,
                )
    _progress()
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"finished: ok={len(to_do) - len(failures)} failed={len(failures)}")
    print(f"failures: {failures_path}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DATA_CACHE)
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 = all targets")
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force-master", action="store_true")
    args = parser.parse_args()
    if args.end is None:
        args.end = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    lock = _DownloadLock(args.cache_dir)
    if not lock.acquire():
        print("another download instance is already running; exiting")
        return 3
    try:
        return _run_main(args, lock)
    finally:
        lock.release()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
