"""Fetch REAL per-stock forum posts from guba.eastmoney.com (股吧) into the contract format.

Round-19 priority 3: the 股吧 JSON API (gbapi.eastmoney.com) returns 403, but
the HTML list page (https://guba.eastmoney.com/list,{code},f.html) is fully
reachable (HTTP 200) and exposes per-stock community posts with read/reply
counts and timestamps. This tool parses those pages and writes contract rows
matching Phase_3.alternative_data_contract.RAW_REQUIRED_COLUMNS:
  record_id, source, source_type, license, published_at, ingested_at,
  symbol, text, is_synthetic=False

Honest limitations (recorded in the file metadata comment):
  - the list page shows "最后更新" (last-update) time, used as the
    published_at proxy (not the original post creation time),
  - only MM-DD HH:MM is exposed; the year is inferred as the current year
    when the date is not in the future (free-source recent-window constraint,
    same class as the Sina source),
  - rows mix community posts and official-account news reposts within the
    stock's 股吧 page (source_type kept as "forum" for all).

Usage:
  python tools/fetch_forum_guba.py --codes 600519.SH 000858.SZ --max-pages 2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")

_ROW_RE = re.compile(
    r'<tr class="listitem">.*?'
    r'<div class="read">([^<]*)</div>.*?'
    r'<div class="reply">([^<]*)</div>.*?'
    r'<div class="title"><a data-postid="(\d+)" data-posttype="(\d+)" href="([^"]*)">(.*?)</a></div>.*?'
    r'<div class="author"><a[^>]*>(.*?)</a></div>.*?'
    r'<div class="update">([^<]*)</div>',
    re.S,
)


def fetch(url: str, timeout: float = 15.0) -> Tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Referer": "https://guba.eastmoney.com/"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", errors="replace")


def parse_rows(txt: str) -> List[dict]:
    rows = []
    for m in _ROW_RE.finditer(txt):
        rows.append({
            "read": m.group(1).strip(),
            "reply": m.group(2).strip(),
            "postid": m.group(3).strip(),
            "posttype": m.group(4).strip(),
            "href": m.group(5).strip(),
            "title": re.sub(r"<[^>]+>", "", m.group(6)).strip(),
            "author": re.sub(r"<[^>]+>", "", m.group(7)).strip(),
            "update": m.group(8).strip(),
        })
    return rows


def to_contract_rows(code: str, items: List[dict], ingested: datetime) -> List[dict]:
    rows = []
    seen = set()
    for it in items:
        postid = it["postid"]
        if postid in seen or not it["title"]:
            continue
        seen.add(postid)
        m = re.match(r"(\d{2})-(\d{2}) (\d{2}):(\d{2})", it["update"])
        if not m:
            continue
        mm, dd, hh, mi = (int(x) for x in m.groups())
        # The 股吧 list page shows China Standard Time (UTC+8); the year is
        # inferred as the current year unless the local datetime is in the
        # future (new-year boundary), then the previous year.
        local_tz = timezone(timedelta(hours=8))
        now_local = ingested.astimezone(local_tz)
        year = now_local.year
        published_local = datetime(year, mm, dd, hh, mi, tzinfo=local_tz)
        if published_local > now_local + timedelta(days=1):
            published_local = datetime(year - 1, mm, dd, hh, mi, tzinfo=local_tz)
        published = published_local.astimezone(timezone.utc)
        if published > ingested:
            continue  # never record future items as already ingested
        rows.append({
            "record_id": f"{code}:guba:{postid}",
            "source": "guba_eastmoney",
            "source_type": "forum",
            "license": "public_display",
            "published_at": published.isoformat(),
            "ingested_at": ingested.isoformat(),
            "symbol": code,
            "text": it["title"],
            "is_synthetic": False,
        })
    return rows


def fetch_symbol_posts(code: str, max_pages: int = 1, sleep_s: float = 0.3) -> List[dict]:
    digits = code.split(".")[0]
    out: List[dict] = []
    for page in range(1, max_pages + 1):
        url = f"https://guba.eastmoney.com/list,{digits},f.html" if page == 1 \
            else f"https://guba.eastmoney.com/list,{digits},f_{page}.html"
        try:
            st, txt = fetch(url)
            if st != 200:
                print(f"  [{code}] page {page} HTTP {st}", file=sys.stderr)
                break
            items = parse_rows(txt)
            if not items:
                break
            out.extend(items)
        except Exception as exc:
            print(f"  [{code}] page {page} error: {type(exc).__name__} {str(exc)[:80]}", file=sys.stderr)
            break
        time.sleep(sleep_s)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--codes", nargs="*", default=None)
    parser.add_argument("--max-pages", type=int, default=1)
    parser.add_argument("--sleep", type=float, default=0.3)
    parser.add_argument("--out", type=Path,
                        default=PROJECT_ROOT / "Data_Cache" / "alternative_raw" / "guba_forum_real.jsonl")
    args = parser.parse_args()

    codes = args.codes or ["600519.SH", "000858.SZ", "601318.SH", "600036.SH", "000001.SZ"]
    all_rows: List[dict] = []
    for code in codes:
        items = fetch_symbol_posts(code, max_pages=args.max_pages, sleep_s=args.sleep)
        # snapshot ingest AFTER the fetch so the fetched page is by definition
        # fully in the past (PIT-consistent; the Sina tool has the same guard)
        ingested = datetime.now(timezone.utc)
        rows = to_contract_rows(code, items, ingested)
        all_rows.extend(rows)
        print(f"{code}: {len(items)} rows parsed -> {len(rows)} contract rows "
              f"(newest={rows[0]['published_at'] if rows else '-'})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"total: {len(all_rows)} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
