"""Point-in-time fundamental factors from baostock annual reports.

The downloader ``tools/fetch_fundamentals.py`` caches annual (Q4) reports for
the most liquid names with ``pubDate`` (announcement date). This module aligns
each value PIT: a fundamental enters the panel only on/after its publication
date and stays until the next report, so there is no look-ahead.

Coverage is intentionally partial (top ~600 liquid names); uncovered symbols
fall back to the cross-sectional median inside the engine, so they are neither
boosted nor punished by the factor.

Factors (``FUNDAMENTAL_FIELDS``):
- gp_margin: gross margin (quality)
- roe: return on equity (quality)
- np_margin: net profit margin (quality)
- yoy_ni: net income YoY growth (growth)
- yoy_pni: parent net income YoY growth (growth)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


FUNDAMENTAL_FIELDS: Dict[str, str] = {
    "gp_margin": "gpMargin",
    "roe": "roeAvg",
    "np_margin": "npMargin",
    "yoy_ni": "yoyNI",
    "yoy_pni": "yoyPNI",
    "eps_ttm": "epsTTM",
    # balance-sheet / operation fields (tools/fetch_balance.py)
    "debt_ratio": "debt_ratio",
    "current_ratio": "current_ratio",
    "asset_turn": "asset_turn",
}


def load_fundamentals(path: Optional[Path] = None, top_n: Optional[int] = None) -> Dict[str, list]:
    """Load the cached {symbol: [annual report records]} map (empty when missing)."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "Data_Cache" / "fundamentals_annual.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if top_n is not None:
        # the cache is written in amount-ranked order, so the first ``top_n``
        # keys are the most liquid names (the PIT-validated coverage)
        data = dict(list(data.items())[: max(0, int(top_n))])
    return data


def load_all_fundamentals(
    profit_path: Optional[Path] = None,
    balance_path: Optional[Path] = None,
    top_n: Optional[int] = None,
) -> Dict[str, list]:
    """Merge the profit/growth and balance/operation caches by symbol+period."""
    root = Path(__file__).resolve().parents[1] / "Data_Cache"
    profit = load_fundamentals(profit_path or (root / "fundamentals_annual.json"), top_n=top_n)
    balance = load_fundamentals(balance_path or (root / "fundamentals_balance.json"), top_n=top_n)
    out: Dict[str, list] = {}
    for sym in set(profit) | set(balance):
        merged: dict = {}
        for rec in profit.get(sym, []) + balance.get(sym, []):
            key = (rec.get("pub_date"), rec.get("stat_date"))
            bucket = merged.setdefault(key, {})
            bucket.update(rec)
        out[sym] = list(merged.values())
    return out


def build_fundamental_panels(
    fundamentals: Dict[str, list],
    common: pd.DatetimeIndex,
    symbols: list,
    fields: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """PIT-aligned (date x symbol) panels for each requested fundamental field.

    Every value is forward-filled from its publication date, so row ``t`` only
    uses reports announced at or before ``t``.
    """
    fields = fields or FUNDAMENTAL_FIELDS
    covered = [s for s in symbols if s in fundamentals and fundamentals[s]]
    out = {fname: pd.DataFrame(index=common, columns=covered, dtype=float) for fname in fields}
    for sym in covered:
        records = pd.DataFrame(fundamentals[sym])
        if records.empty or "pub_date" not in records:
            continue
        records = records.dropna(subset=["pub_date"]).sort_values("pub_date")
        records["pub_date"] = pd.to_datetime(records["pub_date"], errors="coerce")
        records = records.dropna(subset=["pub_date"])
        for fname, col in fields.items():
            if col not in records:
                continue
            series = pd.Series(records[col].to_numpy(dtype=float), index=pd.to_datetime(records["pub_date"]))
            aligned = series.reindex(common)
            out[fname][sym] = aligned.ffill().to_numpy(dtype=float)
    return out
