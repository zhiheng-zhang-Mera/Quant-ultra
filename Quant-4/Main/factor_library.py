"""Open-source factor library (PIT, pluggable into the composite score).

The engine's built-in factors are momentum/trend/reversal/low-vol/liquidity/
dividend/breakout. This module exposes a small, honest subset of factors from
public open-source factor sets - primarily Microsoft's qlib ``Alpha158``
(KBAR feature family) - that are computable point-in-time from the panel's
price/volume/amount data. Every entry carries its source and formula so the
repo can be audited against the upstream definitions.

All functions are pure and PIT by construction: ``compute_factor`` only uses
rows at or before ``date``. Factors are opt-in: the engine consumes them only
through ``RotationParams.extra_factor_weights`` (default empty), so the
evidence-gated production score is unchanged unless a user mounts them.

Reference (upstream):
- Microsoft qlib Alpha158: https://github.com/microsoft/qlib (KBAR features:
  ROC/MA/STD/MAX/IDLL/RSI/... computed on rolling windows of OHLCV)
- WorldQuant 101 Alphas (WQ101) subset implementable from price/volume:
  https://arxiv.org/abs/1601.00991
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd


# name -> {"source", "formula", "family"}
FACTOR_SPECS: Dict[str, dict] = {
    "roc5": {"source": "qlib Alpha158 (KBAR.ROC)", "formula": "close/close_5 - 1", "family": "momentum"},
    "roc10": {"source": "qlib Alpha158 (KBAR.ROC)", "formula": "close/close_10 - 1", "family": "momentum"},
    "roc20": {"source": "qlib Alpha158 (KBAR.ROC)", "formula": "close/close_20 - 1", "family": "momentum"},
    "roc60": {"source": "qlib Alpha158 (KBAR.ROC)", "formula": "close/close_60 - 1", "family": "momentum"},
    "ma5": {"source": "qlib Alpha158 (KBAR.MA)", "formula": "close/MA5 - 1", "family": "trend"},
    "ma10": {"source": "qlib Alpha158 (KBAR.MA)", "formula": "close/MA10 - 1", "family": "trend"},
    "ma20": {"source": "qlib Alpha158 (KBAR.MA)", "formula": "close/MA20 - 1", "family": "trend"},
    "ma60": {"source": "qlib Alpha158 (KBAR.MA)", "formula": "close/MA60 - 1", "family": "trend"},
    "std5": {"source": "qlib Alpha158 (KBAR.STD)", "formula": "std(ret,5)", "family": "volatility"},
    "std20": {"source": "qlib Alpha158 (KBAR.STD)", "formula": "std(ret,20)", "family": "volatility"},
    "std60": {"source": "qlib Alpha158 (KBAR.STD)", "formula": "std(ret,60)", "family": "volatility"},
    "max20": {"source": "qlib Alpha158 (KBAR.MAX)", "formula": "max(ret,20)", "family": "extremes"},
    "idll20": {"source": "qlib Alpha158 (KBAR.IDLL)", "formula": "min(ret,20)", "family": "extremes"},
    "rsi14": {"source": "qlib Alpha158 (KBAR.RSI)", "formula": "Wilder RSI(14)", "family": "momentum"},
    "vol_ratio20": {"source": "qlib Alpha158 (KBAR.VMA)", "formula": "MA(volume,5)/MA(volume,20)", "family": "liquidity"},
}


def _wilder_rsi(close: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    # zero average losses (monotonic rise) -> RSI 100 by convention
    return rsi.where(avg_loss > 0, 100.0)


def _cache_key(panel, name: str, window: int) -> str:
    return f"{name}:{window}"


def _cached(panel, key: str):
    return getattr(panel, "_factor_cache", {}).get(key)


def _store(panel, key: str, frame: pd.DataFrame) -> pd.DataFrame:
    cache = getattr(panel, "_factor_cache", None)
    if cache is None:
        cache = {}
        panel._factor_cache = cache
    cache[key] = frame
    return frame


def compute_factor(panel, name: str, date: pd.Timestamp) -> Optional[pd.Series]:
    """Return the cross-sectional PIT factor row at ``date`` (or None)."""
    if name not in FACTOR_SPECS:
        raise ValueError(f"unknown open-source factor {name!r}; choose from {sorted(FACTOR_SPECS)}")
    if date not in panel.common:
        return None
    close = panel.close
    if name in ("roc5", "roc10", "roc20", "roc60"):
        window = int(name[3:])
        key = _cache_key(panel, name, window)
        frame = _cached(panel, key)
        if frame is None:
            frame = _store(panel, key, close / close.shift(window) - 1.0)
        return frame.loc[date]
    if name in ("ma5", "ma10", "ma20", "ma60"):
        window = int(name[2:])
        key = _cache_key(panel, name, window)
        frame = _cached(panel, key)
        if frame is None:
            ma = close.rolling(window, min_periods=min(window, 10)).mean()
            frame = _store(panel, key, close / ma - 1.0)
        return frame.loc[date]
    rets = close.pct_change(fill_method=None)
    if name in ("std5", "std20", "std60"):
        window = int(name[3:])
        key = _cache_key(panel, name, window)
        frame = _cached(panel, key)
        if frame is None:
            frame = _store(panel, key, rets.rolling(window, min_periods=min(window, 10)).std(ddof=0))
        return frame.loc[date]
    if name == "max20":
        key = _cache_key(panel, name, 20)
        frame = _cached(panel, key)
        if frame is None:
            frame = _store(panel, key, rets.rolling(20, min_periods=5).max())
        return frame.loc[date]
    if name == "idll20":
        key = _cache_key(panel, name, 20)
        frame = _cached(panel, key)
        if frame is None:
            frame = _store(panel, key, rets.rolling(20, min_periods=5).min())
        return frame.loc[date]
    if name == "rsi14":
        key = _cache_key(panel, name, 14)
        frame = _cached(panel, key)
        if frame is None:
            frame = _store(panel, key, _wilder_rsi(close, 14))
        return frame.loc[date]
    if name == "vol_ratio20":
        key = _cache_key(panel, name, 20)
        frame = _cached(panel, key)
        if frame is None:
            vol = panel.volume
            frame = _store(panel, key, vol.rolling(5).mean() / vol.rolling(20).mean().replace(0, np.nan))
        return frame.loc[date]
    return None


def compute_factors(panel, date: pd.Timestamp, names) -> Dict[str, pd.Series]:
    """Compute a set of factors at ``date``, skipping all-NaN rows."""
    out: Dict[str, pd.Series] = {}
    for name in names:
        row = compute_factor(panel, name, date)
        if row is not None and row.notna().sum() >= 5:
            out[name] = row
    return out
