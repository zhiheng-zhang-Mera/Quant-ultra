"""Fetch the CSRC industry map for the full PIT universe from baostock.

baostock ``query_stock_industry`` is free and rate-limited. The tool is
resumable: progress is written incrementally to a JSON cache
(``Data_Cache/sector_map_full.json``), and already-fetched symbols are
skipped on restart, so an interrupted run can continue where it left off.

Usage:
    python tools/fetch_sector_map.py [--limit 200] [--sleep 0.05]

Output format: {"600519.SH": "C15酒、饮料和精制茶制造业", ...}
ETFs and instruments without an industry come back empty and are stored as
"综合" so downstream factors can group them.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_CACHE = PROJECT_ROOT / "Data_Cache"
DEFAULT_OUT = DATA_CACHE / "sector_map_full.json"
MASTER_REL = Path("universe") / "ashare_master.parquet"
EXCLUDED_PREFIXES = ("4", "8", "92")


def to_baostock_code(symbol: str) -> str:
    number, exchange = symbol.split(".")
    return f"{exchange.lower()}.{number}"


def fetch_industry(code: str, retries: int = 3) -> str:
    import baostock as bs

    for attempt in range(retries):
        rs = bs.query_stock_industry(code=code)
        rows = []
        while rs.error_code == "0" and rs.next():
            rows.append(rs.get_row_data())
        if rs.error_code == "0":
            # row = [update_date, code, code_name, industry, industry_class]
            if rows:
                return rows[0][3] or "综合"
            return "综合"
        if attempt < retries - 1:
            time.sleep(0.5)
    return "综合"


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch CSRC industry map for the PIT universe")
    parser.add_argument("--limit", type=int, default=0, help="fetch only the first N symbols (test mode)")
    parser.add_argument("--sleep", type=float, default=0.05, help="seconds between queries (rate limit)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    master_path = DATA_CACHE / MASTER_REL
    if not master_path.exists():
        print(f"master list not found: {master_path}", file=sys.stderr)
        return 1
    master = pd.read_parquet(master_path)
    symbols = [
        row.symbol
        for row in master.itertuples()
        if row.type == "1" and not row.symbol.startswith(EXCLUDED_PREFIXES)
    ]
    if args.limit:
        symbols = symbols[: args.limit]
    print(f"target symbols: {len(symbols)}")

    sector_map: dict = {}
    if args.out.exists():
        sector_map = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"resuming from cache: {len(sector_map)} already fetched")

    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        print(f"baostock login failed: {lg.error_msg}", file=sys.stderr)
        return 1
    try:
        done = 0
        t0 = time.time()
        for i, symbol in enumerate(symbols):
            if symbol in sector_map:
                continue
            sector_map[symbol] = fetch_industry(to_baostock_code(symbol))
            done += 1
            if args.sleep > 0:
                time.sleep(args.sleep)
            if done % 200 == 0:
                args.out.write_text(json.dumps(sector_map, ensure_ascii=False), encoding="utf-8")
                rate = done / max(time.time() - t0, 1e-6)
                eta = (len(symbols) - i - 1) / max(rate, 1e-6) / 60
                print(f"  fetched {done} (total {len(sector_map)}), rate {rate:.1f}/s, eta ~{eta:.0f} min", flush=True)
    finally:
        args.out.write_text(json.dumps(sector_map, ensure_ascii=False), encoding="utf-8")
        bs.logout()
    print(f"done: {len(sector_map)} symbols -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
