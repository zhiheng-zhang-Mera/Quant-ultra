"""Resumable full-history downloader via the sina klc_kl (encrypted) channel.

On this network the fast providers are blocked: Eastmoney (connection abort),
Tencent (anti-bot HTTP 501) and baostock (login rejected after rate-limit).
The sina ``hisdata_klc2/klc_kl.js`` endpoint still serves the FULL history for
live AND delisted names, with the payload encrypted by a JS decoder that
akshare bundles (``hk_js_decode``). We decode it with ONE py_mini_racer isolate
whose calls are serialized by a lock (V8 isolates crash if initialized
concurrently; a single locked isolate is thread-safe - verified 8x10 calls).

The sina series is raw; the qfq factors (``qfq.js``, plain JSON) are applied by
division (``adjusted = raw / factor``, factor ffill'd by ex-date), exactly like
akshare's ``stock_zh_a_daily``. Volume/amount come directly from the payload.

Channel order: sina_klc -> tencent (breaker) -> baostock (breaker, locked).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
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
from tools.download_universe import _DownloadLock, _save, cached_complete  # noqa: E402

logger = logging.getLogger("download_universe_sina")
_HDR = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

_JS = None
_JS_LOCK = threading.Lock()


def _decoder():
    """Lazily build the single locked py_mini_racer isolate."""
    global _JS
    if _JS is None:
        from akshare.stock.cons import hk_js_decode

        import py_mini_racer

        js = py_mini_racer.MiniRacer()
        js.eval(hk_js_decode)
        _JS = js
    return _JS


def _decode(payload: str) -> list:
    with _JS_LOCK:
        return _decoder().call("d", payload)


def fetch_sina_klc(symbol: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """Full qfq-adjusted daily history for a symbol via sina klc_kl + qfq.js."""
    code, exchange = symbol.split(".")
    px = f"{exchange.lower()}{code}"
    try:
        r1 = requests.get(
            f"https://finance.sina.com.cn/realstock/company/{px}/hisdata_klc2/klc_kl.js",
            headers=_HDR, timeout=25,
        )
        if r1.status_code != 200:
            return None
        payload = r1.text.split("=")[1].split(";")[0].replace('"', "")
        rows = _decode(payload)
        if not rows:
            return None
        frame = pd.DataFrame(rows)
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        if frame["date"].dt.tz is not None:
            frame["date"] = frame["date"].dt.tz_localize(None)
        frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
        for col in ("open", "high", "low", "close", "volume", "amount"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        # qfq factors (plain JSON after 'var xxqfq=')
        r2 = requests.get(
            f"https://finance.sina.com.cn/realstock/company/{px}/qfq.js",
            headers=_HDR, timeout=20,
        )
        if r2.status_code == 200:
            m = re.search(r"=(\{.*\})", r2.text, re.S)
            if m:
                factors = json.loads(m.group(1)).get("data") or []
                if factors:
                    fdf = pd.DataFrame(factors)
                    fdf["d"] = pd.to_datetime(fdf["d"], errors="coerce")
                    fdf["f"] = pd.to_numeric(fdf["f"], errors="coerce")
                    fdf = fdf.dropna(subset=["d", "f"]).sort_values("d")
                    merged = pd.merge_asof(
                        frame.sort_values("date"), fdf, left_on="date", right_on="d", direction="backward"
                    )
                    factor = merged["f"].fillna(1.0).to_numpy(dtype=float)
                    for col in ("open", "high", "low", "close"):
                        frame[col] = (frame[col].to_numpy(dtype=float) / factor).round(2)
        # keep rows in the requested window
        start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
        frame = frame[(frame["date"] >= start_ts) & (frame["date"] <= end_ts)]
        if frame.empty:
            return None
        return frame[["date", "open", "high", "low", "close", "volume", "amount"]]
    except Exception as exc:  # noqa: BLE001
        logger.debug("sina klc failed for %s: %s", symbol, exc)
        return None


def download_one_sina(symbol: str, start: str, end: str, timeout: float = 60.0) -> Optional[pd.DataFrame]:
    """sina_klc primary; tencent/baostock fallbacks are guarded by their own
    availability probes (both are service-blocked on this network today)."""
    df = None

    def _run(fn):
        box: dict = {}

        def worker():
            try:
                box["df"] = fn()
            except BaseException as exc:  # noqa: BLE001
                box["exc"] = exc

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            raise TimeoutError(f"{symbol} exceeded {timeout}s")
        if "exc" in box:
            raise box["exc"]
        return box.get("df")

    try:
        df = _run(lambda: fetch_sina_klc(symbol, start, end))
    except Exception:
        df = None
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
        f"(workers={args.workers}, channel=sina_klc)",
        flush=True,
    )
    if args.dry_run:
        return 0

    failures: Dict[str, str] = {}
    done = 0
    lock = threading.Lock()
    t0 = time.time()
    progress_path = args.cache_dir / "universe" / "download_progress_sina.json"
    failures_path = args.cache_dir / "universe" / "download_failures_sina.json"

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
            pool.submit(download_one_sina, symbol, args.start, out_dates_map[symbol].strftime("%Y-%m-%d"), args.timeout): symbol
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
            if done % 100 == 0 or done == len(to_do):
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
    parser.add_argument("--timeout", type=float, default=60.0)
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
