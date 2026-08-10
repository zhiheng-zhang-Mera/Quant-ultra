"""Backfill PIT dividend history for the survivorship-free universe.

The weekly-rotation dividend factor needs per-symbol cash-per-share history
keyed by ex-date. baostock's dividend endpoint is currently unreliable in this
environment, so this tool uses akshare's ``stock_history_dividend_detail``
(Eastmoney) which returns the full history including the ex-date and cash per
share, works for delisted names, and is stable at 4 concurrent workers.

Output schema matches ``Main/pit_dividends.py`` so ``load_dividend_cash`` can
consume it unchanged: ``Data_Cache/dividends/<SYMBOL>_dividends.parquet`` with
columns ``symbol, ex_date, cash_ps``. Resume-safe: symbols with a cached file
that already reaches the target year are skipped.
"""
from __future__ import annotations

import argparse
import glob
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
CACHE_DIR = PROJECT_ROOT / "Data_Cache"
DIV_DIR = CACHE_DIR / "dividends"


def _fetch_one(code: str) -> Optional[pd.DataFrame]:
    import akshare as ak

    df = ak.stock_history_dividend_detail(symbol=code, indicator="分红", date="")
    if df is None or df.empty:
        return None
    out = pd.DataFrame(
        {
            "ex_date": pd.to_datetime(df["除权除息日"].replace("", np.nan), errors="coerce"),
            "cash_ps": pd.to_numeric(df["派息"], errors="coerce"),
        }
    )
    out = out.dropna(subset=["ex_date", "cash_ps"])
    out = out[out["cash_ps"] > 0].drop_duplicates(subset=["ex_date"], keep="last")
    if out.empty:
        return None
    return out.sort_values("ex_date")[["ex_date", "cash_ps"]]


def _save(symbol: str, frame: pd.DataFrame) -> None:
    frame = frame.copy()
    frame.insert(0, "symbol", symbol)
    path = DIV_DIR / f"{symbol.replace('.', '_')}_dividends.parquet"
    tmp = path.with_name(f"{path.stem}.{threading.get_ident()}.tmp.parquet")
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def _cached_ok(symbol: str, target_year: int) -> bool:
    path = DIV_DIR / f"{symbol.replace('.', '_')}_dividends.parquet"
    if not path.exists():
        return False
    try:
        frame = pd.read_parquet(path, columns=["ex_date"])
        dates = pd.to_datetime(frame["ex_date"], errors="coerce").dropna()
        return bool(len(dates)) and dates.max().year >= target_year - 1
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target-year", type=int, default=int(pd.Timestamp.now().year))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    DIV_DIR.mkdir(parents=True, exist_ok=True)
    symbols = sorted(
        Path(p).stem.replace("_history", "")
        for p in glob.glob(str(CACHE_DIR / "*_history.parquet"))
    )
    # Only A-share stocks: ETF/fund codes carry dividends inside NAV already
    # and are not part of the per-share cash factor.
    stocks = [
        s for s in symbols
        if not s.split(".")[0].startswith(("15", "16", "50", "51", "56", "58", "159"))
        and not s.startswith("us_")
    ]
    to_do = [s for s in stocks if not _cached_ok(s, args.target_year)]
    if args.limit > 0:
        to_do = to_do[: args.limit]
    print(f"stock symbols={len(stocks)} cached_ok={len(stocks) - len(to_do)} to_fetch={len(to_do)}")
    if args.dry_run:
        return 0

    failures: Dict[str, str] = {}
    done = 0
    lock = threading.Lock()
    t0 = time.time()
    progress_path = DIV_DIR.parent / "universe" / "dividend_progress.json"

    def _progress():
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0.0
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(
            json.dumps(
                {
                    "updated_at": datetime.now().astimezone().isoformat(),
                    "to_fetch": len(to_do), "done": done, "failed": len(failures),
                    "rate_per_s": round(rate, 3),
                    "eta_minutes": round((len(to_do) - done) / rate / 60, 1) if rate > 0 else None,
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_fetch_one, s.split(".")[0]): s for s in to_do}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                frame = future.result()
            except Exception as exc:  # noqa: BLE001
                frame = None
                failures[symbol] = f"{type(exc).__name__}: {exc}"
            if frame is not None and len(frame):
                try:
                    _save(symbol, frame)
                except Exception as exc:  # noqa: BLE001
                    failures[symbol] = f"save: {exc}"
            with lock:
                done += 1
            if done % 200 == 0 or done == len(to_do):
                _progress()
                print(f"[{done}/{len(to_do)}] ok={done - len(failures)} failed={len(failures)}", flush=True)
    _progress()
    (DIV_DIR.parent / "universe" / "dividend_failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"finished: ok={len(to_do) - len(failures)} failed={len(failures)}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    raise SystemExit(main())
