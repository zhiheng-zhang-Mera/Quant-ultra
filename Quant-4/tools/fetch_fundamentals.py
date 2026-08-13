"""Fetch annual fundamentals for the most liquid PIT names from baostock.

baostock ``query_profit_data`` / ``query_growth_data`` are free and
rate-limited. Fetching the full 5500-name universe quarterly would take 30+
hours, so this tool covers the top ``--top-n`` names by average traded amount
(the engine's own liquidity filters keep picks inside this set anyway) with
ANNUAL (Q4) reports from ``--start-year`` to the latest year. Every record
carries ``pubDate`` (announcement date) so downstream factors can be aligned
point-in-time (a value enters the factor only after its publication).

Output (JSON cache, resumable):
    {"600519.SH": [{"pub_date": "2024-04-03", "stat_date": "2023-12-31",
                    "roeAvg": 0.36, "gpMargin": 0.92, "npMargin": 0.52,
                    "yoyNI": 0.19, "yoyPNI": 0.19, "epsTTM": 59.5}, ...]}
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = PROJECT_ROOT / "Data_Cache"
DEFAULT_OUT = DATA_CACHE / "fundamentals_annual.json"
MASTER_REL = Path("universe") / "ashare_master.parquet"
EXCLUDED_PREFIXES = ("4", "8", "92")


def to_baostock_code(symbol: str) -> str:
    number, exchange = symbol.split(".")
    return f"{exchange.lower()}.{number}"


def top_liquid_symbols(limit: int) -> list:
    """Rank the PIT universe by average daily amount and take the top ``limit``."""
    master = pd.read_parquet(DATA_CACHE / MASTER_REL)
    symbols = [
        row.symbol
        for row in master.itertuples()
        if row.type == "1" and not row.symbol.startswith(EXCLUDED_PREFIXES)
    ]
    amounts = {}
    n = 0
    for sym in symbols:
        path = DATA_CACHE / f"{sym}_history.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["amount"])
            amounts[sym] = float(df["amount"].tail(250).mean())
        except Exception:
            continue
        n += 1
        if n % 1000 == 0:
            print(f"  scanned {n} symbols", flush=True)
    ranked = sorted(amounts, key=amounts.get, reverse=True)
    print(f"ranked {len(ranked)} symbols by amount")
    return ranked[:limit]


def fetch_annual(bs, code: str, start_year: int, end_year: int) -> list:
    records = []
    for year in range(start_year, end_year + 1):
        rs = bs.query_profit_data(code=code, year=year, quarter=4)
        profit = None
        while rs.error_code == "0" and rs.next():
            profit = rs.get_row_data()
        if profit:
            rs2 = bs.query_growth_data(code=code, year=year, quarter=4)
            growth = None
            while rs2.error_code == "0" and rs2.next():
                growth = rs2.get_row_data()
            records.append({
                "pub_date": profit[1],
                "stat_date": profit[2],
                "roeAvg": float(profit[3]) if profit[3] else None,
                "npMargin": float(profit[4]) if profit[4] else None,
                "gpMargin": float(profit[5]) if profit[5] else None,
                "epsTTM": float(profit[7]) if profit[7] else None,
                "yoyNI": float(growth[3]) if growth and growth[3] else None,
                "yoyPNI": float(growth[5]) if growth and growth[5] else None,
            })
    return records


def fetch_quarterly(bs, code: str, start_year: int, end_year: int) -> list:
    """Profit/growth for every quarter (1-4) of each year, PIT pubDate kept."""
    records = []
    for year in range(start_year, end_year + 1):
        for quarter in (1, 2, 3, 4):
            rs = bs.query_profit_data(code=code, year=year, quarter=quarter)
            profit = None
            while rs.error_code == "0" and rs.next():
                profit = rs.get_row_data()
            if profit:
                rs2 = bs.query_growth_data(code=code, year=year, quarter=quarter)
                growth = None
                while rs2.error_code == "0" and rs2.next():
                    growth = rs2.get_row_data()
                records.append({
                    "pub_date": profit[1],
                    "stat_date": profit[2],
                    "roeAvg": float(profit[3]) if profit[3] else None,
                    "npMargin": float(profit[4]) if profit[4] else None,
                    "gpMargin": float(profit[5]) if profit[5] else None,
                    "epsTTM": float(profit[7]) if profit[7] else None,
                    "yoyNI": float(growth[3]) if growth and growth[3] else None,
                    "yoyPNI": float(growth[5]) if growth and growth[5] else None,
                })
    return records


def _fetch_symbol(bs, sym: str, fetch, start_year: int, end_year: int, timeout: float = 60.0) -> list:
    """Fetch one symbol with a watchdog thread so a hung baostock query
    (``rs.next()`` blocking forever) cannot stall the whole download."""
    box: dict = {}

    def worker():
        box["records"] = fetch(bs, sym, start_year, end_year)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print(f"  WARN: {sym} timed out after {timeout:.0f}s, skipped", flush=True)
        return []
    return box.get("records", [])


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch annual fundamentals for liquid PIT names")
    parser.add_argument("--top-n", type=int, default=600)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0, help="fetch only this many symbols (test)")
    parser.add_argument("--quarterly", action="store_true",
                        help="fetch all four quarters (fresh timeliness) into "
                             "fundamentals_quarterly.json")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    symbols = top_liquid_symbols(args.top_n)
    if args.limit:
        symbols = symbols[: args.limit]
    if args.quarterly:
        args.out = args.out.parent / "fundamentals_quarterly.json"
    print(f"target symbols: {len(symbols)}, years {args.start_year}..2025, "
          f"{'quarterly' if args.quarterly else 'annual Q4'}")

    cache: dict = {}
    if args.out.exists():
        cache = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"resuming: {len(cache)} symbols cached")

    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        print(f"baostock login failed: {lg.error_msg}", file=sys.stderr)
        return 1
    try:
        t0 = time.time()
        done = 0
        for i, sym in enumerate(symbols):
            if sym in cache and cache[sym]:
                continue
            if i % 10 == 0:
                print(f"  progress: {i}/{len(symbols)} symbols, cache={len(cache)}", flush=True)
            fetch = fetch_quarterly if args.quarterly else fetch_annual
            records = _fetch_symbol(bs, to_baostock_code(sym), fetch, args.start_year, 2025)
            if records:
                cache[sym] = records
                done += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
            if done % 10 == 0 and done:
                args.out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(symbols) - i - 1) / max(rate, 1e-6) / 60
                print(f"  fetched {done} symbols, rate {rate:.2f}/s, eta ~{eta:.0f} min", flush=True)
    finally:
        args.out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        bs.logout()
    print(f"done: {len(cache)} symbols with fundamentals -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
