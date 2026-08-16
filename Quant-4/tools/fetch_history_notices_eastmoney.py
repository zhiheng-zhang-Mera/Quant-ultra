"""Fetch FULL-HISTORY per-stock announcements from eastmoney (data.eastmoney.com/notices).

Round-19.6 priority 1: the free Sina news source only covers ~5 months, which
blocked the full-history revalidation (R16/R18). The eastmoney notices API
(through akshare stock_individual_notice_report) exposes the COMPLETE per-stock
announcement history (measured: 600519 -> 1074 rows, 2001-07-26..2026-08-15).
Announcements are point-in-time company disclosures (reports, board
resolutions, shareholder changes, contracts...) - a company-level text signal
with exact dates, distinct from news but suitable for the historical
revalidation harness (contract columns + dictionary sentiment + optional LLM).

Output rows match Phase_3.alternative_data_contract.RAW_REQUIRED_COLUMNS:
  record_id, source, source_type, license, published_at, ingested_at,
  symbol, text, is_synthetic=False

Usage:
  python tools/fetch_history_notices_eastmoney.py --codes 600519.SH 000858.SZ
      [--out Quant-4/Data_Cache/alternative_raw/eastmoney_notice_real_38.jsonl]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PRODUCTION_38 = [
    "600519.SH", "000858.SZ", "601318.SH", "600036.SH", "000001.SZ", "000333.SH",
    "600900.SH", "601899.SH", "002594.SZ", "300750.SZ", "600887.SH", "601012.SH",
    "600028.SH", "600309.SH", "601857.SH", "600941.SH", "601398.SH", "601628.SH",
    "600585.SH", "601088.SH", "600690.SH", "600048.SH", "600030.SH", "601166.SH",
    "600276.SH", "000651.SZ", "000725.SZ", "002415.SZ", "002714.SZ", "300059.SZ",
    "300760.SZ", "002475.SZ", "000002.SZ", "000568.SZ", "002304.SZ", "300124.SZ",
    "688981.SH", "688111.SH",
]


def code6(symbol: str) -> str:
    return symbol.split(".")[0]


def make_rows(code: str, frame, ingested: datetime) -> List[dict]:
    rows = []
    seen = set()
    for _, it in frame.iterrows():
        title = str(it.get("公告标题", "")).strip()
        day = str(it.get("公告日期", "")).strip()
        url = str(it.get("网址", "")).strip()
        if not title or not day:
            continue
        # announcement id from the detail URL (AN...html) - stable unique key
        m = re.search(r"AN([0-9]+)", url)
        ann_id = m.group(1) if m else f"{day}:{len(seen)}"
        key = f"{code}:eastmoney_notice:{ann_id}"
        if key in seen:
            continue
        seen.add(key)
        try:
            published = datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if published > ingested:
            continue  # never record future items as already ingested
        rows.append({
            "record_id": key,
            "source": "eastmoney_notices",
            "source_type": "notice",
            "license": "public_display",
            "published_at": published.isoformat(),
            "ingested_at": ingested.isoformat(),
            "symbol": code,
            "text": title,
            "is_synthetic": False,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="*", default=None)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "Data_Cache" / "alternative_raw" / "eastmoney_notice_real_38.jsonl")
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    codes = args.codes or PRODUCTION_38
    try:
        import akshare as ak
    except ImportError as exc:
        print(f"FATAL: akshare required ({exc})", file=sys.stderr)
        return 2

    ingested = datetime.now(timezone.utc)
    all_rows: List[dict] = []
    for code in codes:
        c6 = code6(code)
        try:
            t0 = time.time()
            frame = ak.stock_individual_notice_report(security=c6)
            rows = make_rows(code, frame, ingested)
            all_rows.extend(rows)
            span = (frame["公告日期"].min(), frame["公告日期"].max()) if len(frame) else ("-", "-")
            print(f"{code}: {len(frame)} raw -> {len(rows)} contract rows | "
                  f"{span[0]}..{span[1]} | {time.time() - t0:.0f}s", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"{code}: FAIL {type(exc).__name__} {str(exc)[:120]}", file=sys.stderr, flush=True)
        time.sleep(args.sleep)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"total: {len(all_rows)} rows -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
