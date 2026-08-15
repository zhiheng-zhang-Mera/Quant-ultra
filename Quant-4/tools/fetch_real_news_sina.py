"""Fetch REAL per-stock news from Sina vCB_AllNewsStock into the contract format.

Produces a JSONL matching Phase_3.alternative_data_contract.RAW_REQUIRED_COLUMNS:
  record_id, source, source_type, license, published_at, ingested_at,
  symbol, text, is_synthetic=False

The Sina AllNewsStock page lists per-stock news (news + announcements) with
exact date/time, symbol-mapped. We fetch a bounded number of pages per symbol
(recent window) and write one record per article. Text is the article title
(we do not fetch each article body to keep the corpus small and bounded).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"


def fetch(url: str, timeout: float = 15.0) -> Tuple[int, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://finance.sina.com.cn/"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def parse_datelist(txt: str) -> List[dict]:
    m = re.search(r'datelist"><ul>(.*?)</ul>', txt, re.S)
    if not m:
        return []
    body = m.group(1)
    rows = re.findall(
        r"(\d{4}-\d{2}-\d{2})&nbsp;(\d{2}:\d{2})&nbsp;&nbsp;<a target='_blank' href='([^']+)'>([^<]+)</a>",
        body,
    )
    return [{"date": d, "time": t, "url": u, "title": title.strip()} for d, t, u, title in rows]


def sina_symbol(code: str) -> str:
    code = code.split(".")[0]
    if code.startswith(("6", "9", "5")):
        return "sh" + code
    return "sz" + code


def fetch_symbol_news(code: str, max_pages: int = 3, sleep_s: float = 0.2) -> List[dict]:
    sym = sina_symbol(code)
    out: List[dict] = []
    for page in range(1, max_pages + 1):
        url = (
            f"https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllNewsStock.php"
            f"?symbol={sym}&Page={page}"
        )
        try:
            _, b = fetch(url)
            txt = b.decode("gbk", errors="replace")
            items = parse_datelist(txt)
            if not items:
                break
            out.extend(items)
        except Exception as exc:
            print(f"  [{code}] page {page} error: {type(exc).__name__} {str(exc)[:80]}", file=sys.stderr)
            break
        time.sleep(sleep_s)
    return out


def to_contract_rows(code: str, items: List[dict], source: str, source_type: str,
                     license_: str, ingested: datetime) -> List[dict]:
    rows = []
    for i, it in enumerate(items):
        published = datetime.strptime(it["date"] + " " + it["time"], "%Y-%m-%d %H:%M")
        published = published.replace(tzinfo=timezone.utc)
        if published > ingested:
            continue  # never record future-published items as already ingested
        rows.append({
            "record_id": f"{code}:sina:{it['date']}:{it['time']}:{i}",
            "source": source,
            "source_type": source_type,
            "license": license_,
            "published_at": published.isoformat(),
            "ingested_at": ingested.isoformat(),
            "symbol": code,
            "text": it["title"],
            "is_synthetic": False,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="*", default=None, help="stock codes (e.g. 600519.SH); default: sample")
    parser.add_argument("--max-pages", type=int, default=3, help="pages per symbol (~40 items/page)")
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "Data_Cache" / "alternative_raw" / "sina_news_real.jsonl")
    args = parser.parse_args()

    codes = args.codes or ["600519.SH", "000858.SZ", "601318.SH", "600036.SH", "000001.SZ",
                           "000333.SZ", "600900.SH", "601899.SH", "002594.SZ", "300750.SZ"]
    ingested = datetime.now(timezone.utc)
    all_rows = []
    for code in codes:
        items = fetch_symbol_news(code, max_pages=args.max_pages, sleep_s=args.sleep)
        rows = to_contract_rows(code, items, "sina_finance", "news", "public_display", ingested)
        all_rows.extend(rows)
        print(f"{code}: {len(items)} items -> {len(rows)} contract rows (newest={items[0]['date'] if items else '-'})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"total: {len(all_rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
