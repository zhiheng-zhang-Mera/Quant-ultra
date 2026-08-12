"""Sector-momentum factor for the "target-sector dynamic switching" layer.

Uses the CSRC industry map fetched by ``tools/fetch_sector_map.py``
(``Data_Cache/sector_map_full.json``). For every symbol the factor is the
cross-sectionally z-scored 20-day momentum of its industry (the mean of its
members' PIT 20d momentum), so the composite can tilt toward the strongest
industries without stock-level sector data on the fly.

Honesty notes:
* The industry map is a snapshot (as of the fetch date). Industry membership
  changes are rare for A-share large caps but not modelled over the backtest
  window - disclosed as an approximation.
* Symbols without a map entry (e.g. ETFs) fall back to the market average so
  they are neither boosted nor penalized.

The factor is opt-in via ``RotationParams.sector_momentum_weight`` (default 0),
keeping the evidence-gated production score unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def load_sector_map(path: Optional[Path] = None) -> Dict[str, str]:
    """Load {symbol: industry} from the cached CSRC map (empty when missing)."""
    if path is None:
        path = Path(__file__).resolve().parents[1] / "Data_Cache" / "sector_map_full.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def industry_momentum_series(
    panel,
    date: pd.Timestamp,
    sector_map: Dict[str, str],
) -> Optional[pd.Series]:
    """Cross-sectional z-score of each symbol's industry 20d momentum (PIT)."""
    if not sector_map or 20 not in panel.momentum or date not in panel.momentum[20].index:
        return None
    mom20 = panel.momentum[20].loc[date]
    valid = mom20.notna()
    if valid.sum() < 10:
        return None
    symbols = [s for s in panel.symbols if valid.get(s, False)]
    industries = [sector_map.get(s, "__market__") for s in symbols]
    industry_mom = pd.Series(mom20[symbols].to_numpy(), index=pd.Index(industries, name="industry"))
    industry_avg = industry_mom.groupby(level=0).mean()
    # market-average fallback for unmapped symbols: the mean of the mapped
    # industries' momentum (so an unmapped ETF is neither boosted nor punished)
    mapped_avg = industry_avg[industry_avg.index != "__market__"]
    market_avg = float(mapped_avg.mean()) if len(mapped_avg) else float(industry_mom.mean())
    out = pd.Series(industry_avg.reindex(industries).to_numpy(), index=pd.Index(symbols))
    out = out.where(pd.Series([s in sector_map for s in symbols], index=symbols), market_avg)
    out = out.fillna(market_avg)
    # cross-sectional z-score of the industry momentum
    mu = float(out.mean())
    sd = float(out.std(ddof=0))
    if sd <= 1e-12:
        return None
    return (out - mu) / sd
