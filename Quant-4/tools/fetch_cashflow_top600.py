"""Fast parallel cash-flow fetch for the most-liquid ~600 names (R19.7).

Replaces the akshare-based fetch (70s/symbol, ~11h for 583 names) with a
direct emweb API implementation: resolve companyType from the finance page
(hidctype), list report dates (2015+ filter), fetch details in 5-date batches,
8 parallel workers -> ~2-5 min total. Incremental save + resume.

Cache: Data_Cache/fundamentals_cashflow.json in the shared format
  {symbol: [{pub_date (NOTICE_DATE), stat_date, ocf_np}]}
with ocf_np = NETCASH_OPERATE / NETPROFIT (NP>0 only; NaN otherwise).

Usage: python reports/_iter/_fetch_cashflow_top600.py [--limit N]
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
Q4 = PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT))

from Main.pit_universe import ETF_UNIVERSE  # noqa: E402
from run_weekly_rotation import DATA_CACHE, build_pit_universe  # noqa: E402

OUT = PROJECT_ROOT / "Data_Cache" / "fundamentals_cashflow.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36"}
BASE = "https://emweb.securities.eastmoney.com/PC_HSF10/NewFinanceAnalysis/"


def _get(url: str, params: dict, timeout: float = 15.0):
    qs = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
    req = urllib.request.Request(url + "?" + qs, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def is_stock(sym: str) -> bool:
    if sym in ETF_UNIVERSE:
        return False
    return not sym.split(".")[0].startswith(("5", "15", "16", "18"))


def fetch_symbol(sym: str) -> list:
    code = sym.split(".")[0]
    market = "SH" if sym.endswith("SH") else "SZ"
    symbol = f"{market}{code}"
    # 1) company type from the finance page
    try:
        html = _get(BASE + "Index", {"type": "web", "code": symbol.lower()}, timeout=20).decode("utf-8", errors="replace")
        m = re.search(r'id="hidctype"\s+value="(\d+)"', html)
        ctype = m.group(1) if m else "4"
    except Exception:
        return []
    try:
        # 2) report dates
        data = json.loads(_get(BASE + "xjllbDateAjaxNew",
                               {"companyType": ctype, "reportDateType": "0", "code": symbol}))
        dates = [r["REPORT_DATE"][:10] for r in data.get("data", [])]
        dates = [d for d in dates if d >= "2015-01-01"]
        if not dates:
            return []
        # 3) details in 5-date batches (API limit)
        recs = []
        for i in range(0, len(dates), 5):
            chunk = ",".join(dates[i:i + 5])
            body = json.loads(_get(BASE + "xjllbAjaxNew",
                                   {"companyType": ctype, "reportDateType": "0",
                                    "reportType": "1", "dates": chunk, "code": symbol}))
            for r in body.get("data", []):
                notice = r.get("NOTICE_DATE")
                ocf = r.get("NETCASH_OPERATE")
                np_ = r.get("NETPROFIT")
                if not notice:
                    continue
                try:
                    npf = float(np_) if pd.notna(np_) else float("nan")
                    ocff = float(ocf) if pd.notna(ocf) else float("nan")
                except (TypeError, ValueError):
                    continue
                ratio = ocff / npf if npf > 0 else float("nan")
                recs.append({"pub_date": str(pd.Timestamp(notice).date()),
                             "stat_date": str(r.get("REPORT_DATE", ""))[:10],
                             "ocf_np": ratio})
        return recs
    except Exception:
        return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=600)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    pit = build_pit_universe(DATA_CACHE)
    frames = pit["frames"]
    amounts = []
    for sym, df in frames.items():
        a = df["amount"].loc["2016-01-01":].sum(skipna=True) if "amount" in df.columns else 0.0
        amounts.append((sym, float(a)))
    amounts.sort(key=lambda kv: -kv[1])
    top = [s for s, _ in amounts[: args.limit] if is_stock(s)]
    print(f"{len(top)} stock names selected (ETFs excluded) of top-{args.limit}", flush=True)

    out = {}
    if OUT.exists():
        out = json.loads(OUT.read_text(encoding="utf-8"))
        print(f"resume: {len(out)} symbols already cached", flush=True)
    todo = [s for s in top if s not in out]
    print(f"{len(todo)} to fetch", flush=True)

    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(fetch_symbol, s): s for s in todo}
        for fut in cf.as_completed(futs):
            sym = futs[fut]
            try:
                out[sym] = fut.result()
            except Exception:
                out[sym] = []
            done += 1
            if done % 25 == 0:
                with_data = sum(1 for v in out.values() if v)
                el = time.time() - t0
                print(f"  {done}/{len(todo)} | {with_data} with data | {el:.0f}s", flush=True)
                OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    with_data = sum(1 for v in out.values() if v)
    print(f"finished: {with_data}/{len(top)} symbols with data -> {OUT} in {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
