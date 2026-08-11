"""Point-in-time per-symbol take-profit / stop-loss bands (bottom-up).

The engine's default stop band is a pair of global constants
(``take_profit_pct`` / ``stop_loss_pct``). This module decouples a bottom-up
alternative: every held symbol gets its own band, sized from its own realized
ATR, so calm names are not churned by a too-wide static band and volatile
names are not stopped out by noise.

Design rules
------------
* **PIT by construction**: all inputs (high/low/close) are backward-looking,
  and every band row ``t`` depends only on rows ``<= t``.
* **Vectorized precompute**: ``precompute_stop_bands`` builds full panels once
  per backtest; the engine loop only reads the band of a held symbol at the
  current date (no per-day recomputation).
* **Bounded**: bands are clipped to [floor, cap] and take-profit is forced to
  stay above stop-loss (``take >= stop * 1.15``), so the mechanism cannot
  produce degenerate bands.
* **Graceful fallback**: NaN bands (missing data / not enough history) are
  ignored by the engine, which falls back to the static global band.

The module imports nothing from ``Main.weekly_rotation`` so it can be unit
tested in isolation and reused by any engine or CLI.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


def wilder_atr(
    high: pd.DataFrame,
    low: pd.DataFrame,
    close: pd.DataFrame,
    window: int = 14,
) -> pd.DataFrame:
    """Wilder's ATR(14) over a wide (date x symbol) panel, vectorized.

    True range is computed column-wise from PIT high/low and the previous
    close; the Wilder recursion is seeded with the simple mean of the first
    ``window`` *valid* true ranges (so symbols that list later start their
    own seed) and then iterated row-wise (a loop over the time axis only,
    with full cross-section numpy ops per row).
    """
    window = max(2, int(window))
    index, columns = high.index, high.columns
    high_a = high.to_numpy(dtype=float)
    low_a = low.to_numpy(dtype=float)
    close_a = close.to_numpy(dtype=float)
    prev_close = np.empty_like(close_a)
    prev_close[0] = np.nan
    prev_close[1:] = close_a[:-1]
    tr = np.maximum(high_a - low_a, np.maximum(np.abs(high_a - prev_close), np.abs(low_a - prev_close)))
    t_rows, n_syms = tr.shape
    atr = np.full_like(tr, np.nan)
    valid = ~np.isnan(tr)
    cum = np.cumsum(valid, axis=0)
    seed_rows, seed_cols = np.where(cum == window)
    for r, c in zip(seed_rows, seed_cols):
        col = tr[: r + 1, c]
        vals = col[~np.isnan(col)]
        atr[r, c] = float(np.mean(vals[-window:]))
    for i in range(1, t_rows):
        upd = np.isfinite(atr[i - 1]) & np.isfinite(tr[i])
        if upd.any():
            atr[i, upd] = (atr[i - 1, upd] * (window - 1) + tr[i, upd]) / window
    return pd.DataFrame(atr, index=index, columns=columns)


def precompute_stop_bands(
    frames: Dict[str, pd.DataFrame],
    common: pd.DatetimeIndex,
    symbols: list,
    params,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Build PIT stop/take fraction panels for every symbol in ``symbols``.

    Returns ``(stop_band, take_band)``; both are ``None`` when dynamic stops
    are disabled. Bands are fractions of the current price (the engine tests
    ``close / entry_price - 1`` against them, and entry vs current price are
    within the daily limit, so the difference is second-order and bounded).

    stop_i = clip(k_sl * ATR_i / close_i,  [stops_floor, stops_cap])
    take_i = clip(k_tp * ATR_i / close_i,  [stops_take_floor, stops_take_cap])
    take_i >= 1.15 * stop_i is enforced after clipping.
    """
    if not bool(getattr(params, "dynamic_stops", False)):
        return None, None
    high = pd.DataFrame({s: frames[s]["high"].reindex(common) for s in symbols})
    low = pd.DataFrame({s: frames[s]["low"].reindex(common) for s in symbols})
    close = pd.DataFrame({s: frames[s]["close"].reindex(common) for s in symbols})
    atr = wilder_atr(high, low, close, window=int(getattr(params, "stops_atr_window", 14)))
    atr_frac = atr / close.replace(0, np.nan)
    sl_mult = float(getattr(params, "stops_atr_sl_mult", 2.5))
    tp_mult = float(getattr(params, "stops_atr_tp_mult", 4.0))
    floor = float(getattr(params, "stops_floor", 0.04))
    cap = float(getattr(params, "stops_cap", 0.15))
    take_floor = float(getattr(params, "stops_take_floor", 0.06))
    take_cap = float(getattr(params, "stops_take_cap", 0.30))
    stop_band = (sl_mult * atr_frac).clip(lower=floor, upper=cap)
    take_band = (tp_mult * atr_frac).clip(lower=take_floor, upper=take_cap)
    # take-profit must clear stop-loss; restore the minimum gap after clipping.
    take_band = take_band.where(take_band >= stop_band * 1.15, stop_band * 1.15)
    take_band = take_band.clip(upper=take_cap)
    return stop_band, take_band


def band_at(
    panel,
    symbol: str,
    date: pd.Timestamp,
    static_stop: float,
    static_take: float,
) -> Tuple[float, float]:
    """Read the PIT band for one symbol/date with static fallback on NaN."""
    if getattr(panel, "stop_band", None) is None or symbol not in panel.stop_band.columns:
        return float(static_stop), float(static_take)
    if date not in panel.stop_band.index:
        return float(static_stop), float(static_take)
    stop = panel.stop_band.at[date, symbol]
    take = panel.take_band.at[date, symbol]
    stop = float(stop) if pd.notna(stop) else float(static_stop)
    take = float(take) if pd.notna(take) else float(static_take)
    return stop, take
