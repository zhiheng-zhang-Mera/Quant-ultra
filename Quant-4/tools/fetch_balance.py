"""Fetch annual balance-sheet + operation data for the most liquid PIT names.

baostock ``query_balance_data`` / ``query_operation_data`` (annual Q4) for the
top ``--top-n`` liquid names, stored separately from the profit/growth cache
(``Data_Cache/fundamentals_balance.json``) so the two downloads never race.
Every record carries ``pubDate`` for PIT alignment.

Fields (see Main.fundamental_factors.BALANCE_FIELDS):
- debt_ratio: liabilityToAsset (safety/leverage, negative edge expected)
- current_ratio: currentRatio (liquidity)
- asset_turn: AssetTurnRatio (efficiency)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = PROJECT_ROOT / "Data_Cache"
DEFAULT_OUT = DATA_CACHE / "fundamentals_balance.json"
MASTER_REL = Path("universe") / "ashare_master.parquet"
EXCLUDED_PREFIXES = ("4", "8", "92")


def to_baostock_code(symbol: str) -> str:
    number, exchange = symbol.split(".")
    return f"{exchange.lower()}.{number}"


def top_liquid_symbols(limit: int) -> list:
    master = pd.read_parquet(DATA_CACHE / MASTER_REL)
    symbols = [
        row.symbol
        for row in master.itertuples()
        if row.type == "1" and not row.symbol.startswith(EXCLUDED_PREFIXES)
    ]
    amounts = {}
    for sym in symbols:
        path = DATA_CACHE / f"{sym}_history.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["amount"])
            amounts[sym] = float(df["amount"].tail(250).mean())
        except Exception:
            continue
    ranked = sorted(amounts, key=amounts.get, reverse=True)
    return ranked[:limit]


def fetch_annual(bs, code: str, start_year: int, end_year: int) -> list:
    records = []
    for year in range(start_year, end_year + 1):
        rs = bs.query_balance_data(code=code, year=year, quarter=4)
        bal = None
        while rs.error_code == "0" and rs.next():
            bal = rs.get_row_data()
        if bal:
            rs2 = bs.query_operation_data(code=code, year=year, quarter=4)
            op = None
            while rs2.error_code == "0" and rs2.next():
                op = rs2.get_row_data()
            records.append({
                "pub_date": bal[1],
                "stat_date": bal[2],
                "debt_ratio": float(bal[5]) if bal[5] else None,      # liabilityToAsset
                "current_ratio": float(bal[3]) if bal[3] else None,
                "asset_turn": float(op[5]) if op and op[5] else None,  # AssetTurnRatio
            })
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch annual balance/operation data for liquid PIT names")
    parser.add_argument("--top-n", type=int, default=600)
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--sleep", type=float, default=0.05)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    symbols = top_liquid_symbols(args.top_n)
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"target symbols: {len(symbols)}, years {args.start_year}..2025, annual Q4")

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
            records = fetch_annual(bs, to_baostock_code(sym), args.start_year, 2025)
            if records:
                cache[sym] = records
                done += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
            if done % 50 == 0 and done:
                args.out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(symbols) - i - 1) / max(rate, 1e-6) / 60
                print(f"  fetched {done} symbols, rate {rate:.2f}/s, eta ~{eta:.0f} min", flush=True)
    finally:
        args.out.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        bs.logout()
    print(f"done: {len(cache)} symbols -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
