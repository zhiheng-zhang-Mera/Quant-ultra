"""Parallel, resumable PIT dividend downloader from cninfo (巨潮资讯).

Baostock's dividend API is rate-limited on this network (~7s/symbol and
throttled), so the primary channel is akshare ``stock_dividend_cninfo``
(authoritative disclosure source, ~4s/symbol, HTTP - safe to parallelize).
Rows are cached per symbol as ``Data_Cache/dividends/<symbol>_dividends.parquet``
with the exact schema ``pit_dividends.load_dividend_cash`` consumes
(symbol, ex_date, cash_ps), so the engine and the honest backtest are unchanged.

PIT rule: a dividend enters the trailing yield only after its ex-dividend date
(除息日, falling back to 除权日 / 除权登记日). Cash per share = 每10股派息 / 10.
"""
from __future__ import annotations

import argparse
import json
import logging
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

from Main.pit_universe import ETF_UNIVERSE, ever_alive_stocks, load_master_list  # noqa: E402

logger = logging.getLogger("download_dividends")
_CASH_COL = 4    # 股息(税前) per 10 shares
_EX_DIV = 7      # 除息日
_EX_RIGHT = 6    # 除权日
_EX_REG = 5      # 除权登记日


def _to_engine_schema(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Extract (ex_date, cash_ps) from the cninfo frame."""
    cash10 = pd.to_numeric(df.iloc[:, _CASH_COL], errors="coerce")
    out = pd.DataFrame(index=df.index)
    out["symbol"] = symbol
    out["ex_date"] = pd.to_datetime(df.iloc[:, _EX_DIV], errors="coerce")
    out["ex_date"] = out["ex_date"].fillna(pd.to_datetime(df.iloc[:, _EX_RIGHT], errors="coerce"))
    out["ex_date"] = out["ex_date"].fillna(pd.to_datetime(df.iloc[:, _EX_REG], errors="coerce"))
    out["cash_ps"] = cash10 / 10.0
    out = out.dropna(subset=["ex_date", "cash_ps"])
    out = out[out["cash_ps"] > 0].drop_duplicates(subset=["ex_date"], keep="last")
    return out[["symbol", "ex_date", "cash_ps"]].sort_values("ex_date")


_CNINFO_URL = "https://webapi.cninfo.com.cn/api/sysapi/p_sysapi1139"


def fetch_cninfo(symbol: str) -> Optional[pd.DataFrame]:
    """Direct cninfo request with the pure-Python AES signature (parallel-safe,
    no py_mini_racer/V8 - akshare's V8 isolate crashes under concurrent use)."""
    import requests

    from tools.cninfo_aes import cninfo_mcode

    headers = {
        "Accept": "*/*",
        "Accept-Enckey": cninfo_mcode(),
        "Accept-Encoding": "gzip, deflate",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        "Origin": "http://webapi.cninfo.com.cn",
        "Referer": "http://webapi.cninfo.com.cn/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "X-Requested-With": "XMLHttpRequest",
    }
    code = symbol.split(".")[0]
    r = requests.post(_CNINFO_URL, params={"scode": code}, headers=headers, timeout=25)
    j = r.json()
    recs = j.get("records") or []
    if not recs:
        return None
    frame = pd.DataFrame(recs)
    # map the API field codes to the same positional layout akshare exposes:
    # [0]=F006D 实施公告日期 [1]=F044V 分红类型 [2]=F010N 送股比例 [3]=F011N 转增比例
    # [4]=F012N 股息(税前)/10股 [5]=F018D 除权登记日 [6]=F020D 除权日 [7]=F023D 除息日
    cols = ["F006D", "F044V", "F010N", "F011N", "F012N", "F018D", "F020D", "F023D", "F025D", "F007V", "F001V"]
    frame = frame[[c for c in cols if c in frame.columns]]
    return _to_engine_schema(frame, symbol)


def _cache_path(cache_dir: Path, symbol: str) -> Path:
    return cache_dir / f"{symbol.replace('.', '_')}_dividends.parquet"


def fetch_one(symbol: str, cache_dir: Path, timeout: float = 30.0) -> str:
    path = _cache_path(cache_dir, symbol)
    if path.exists():
        try:
            frame = pd.read_parquet(path)
            if len(frame) and not frame["ex_date"].isna().all():
                return "cached"
        except Exception:
            pass

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
        df = _run(lambda: fetch_cninfo(symbol))
    except Exception as exc:  # noqa: BLE001
        return f"error:{type(exc).__name__}:{exc}"[:160]
    if df is None or df.empty:
        # symbol without cash dividends in the window: cache an empty frame so
        # it is not refetched every run
        empty = pd.DataFrame(columns=["symbol", "ex_date", "cash_ps"])
        empty.to_parquet(path, index=False)
        return "no-data"
    df.to_parquet(path, index=False)
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DATA_CACHE / "dividends")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 = all stocks")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    master = load_master_list(DATA_CACHE)
    if master.empty:
        print("empty master list; run download_universe first", file=sys.stderr)
        return 2
    end = (pd.Timestamp.now() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    stocks = ever_alive_stocks(master, "2014-01-01", end)
    targets: List[str] = stocks  # ETFs need no cash-dividend factor
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    to_do = [s for s in targets if not _cache_path(args.cache_dir, s).exists()]
    if args.limit > 0:
        to_do = to_do[: args.limit]
    print(f"targets={len(targets)} cached={len(targets) - len(to_do)} to_fetch={len(to_do)} workers={args.workers}", flush=True)
    if args.dry_run or not to_do:
        return 0

    status: Dict[str, str] = {}
    done = 0
    lock = threading.Lock()
    t0 = time.time()

    def _progress():
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        eta = (len(to_do) - done) / rate / 60 if rate > 0 else float("nan")
        (args.cache_dir / "..").resolve()
        (DATA_CACHE / "universe").mkdir(parents=True, exist_ok=True)
        (DATA_CACHE / "universe" / "dividend_progress.json").write_text(
            json.dumps({
                "updated_at": datetime.now().astimezone().isoformat(),
                "to_fetch": len(to_do), "done": done,
                "rate_per_s": round(rate, 3),
                "eta_minutes": round(eta, 1) if eta == eta else None,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(fetch_one, s, args.cache_dir): s for s in to_do}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                status[symbol] = future.result()
            except Exception as exc:  # noqa: BLE001
                status[symbol] = f"error:{exc}"[:160]
            with lock:
                done += 1
            if done % 100 == 0 or done == len(to_do):
                _progress()
                print(f"[{done}/{len(to_do)}] elapsed={time.time() - t0:.0f}s", flush=True)
    _progress()
    from collections import Counter

    counts = Counter(v.split(":")[0] for v in status.values())
    print(f"finished: {dict(counts)}")
    (DATA_CACHE / "universe" / "dividend_failures.json").write_text(
        json.dumps({k: v for k, v in status.items() if not v.startswith("ok")}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(main())
