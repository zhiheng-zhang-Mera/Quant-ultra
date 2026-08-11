"""Tests for the decoupled bottom-up stop-band module (Main.dynamic_stops):
Wilder ATR correctness, PIT-ness, band bounds, disabled default, and the
engine hook that reads per-symbol bands."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from Main.dynamic_stops import band_at, precompute_stop_bands, wilder_atr  # noqa: E402
from Main.weekly_rotation import RotationParams, weekly_rotation_backtest  # noqa: E402


def _synthetic_frames(n_days: int = 180) -> dict:
    dates = pd.bdate_range("2023-01-02", periods=n_days)
    frames = {}
    for i, sym in enumerate(["600000.SH", "000001.SZ", "510300.SH"]):
        base = 10.0 * (1 + i)
        close = base * np.linspace(1.0, 1.6, len(dates))
        open_ = close * 0.999
        high = close * 1.015
        low = close * 0.985
        volume = np.full(len(dates), 2_000_000.0)
        amount = volume * close
        frames[sym] = pd.DataFrame(
            {
                "date": dates,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
            }
        ).set_index("date")
    return frames


def test_wilder_atr_matches_manual_recursion():
    """Vectorized ATR must match the textbook Wilder recursion exactly."""
    dates = pd.bdate_range("2024-01-01", periods=30)
    high = pd.DataFrame({"A": np.linspace(10, 14, 30) * 1.02}, index=dates)
    low = pd.DataFrame({"A": np.linspace(10, 14, 30) * 0.98}, index=dates)
    close = pd.DataFrame({"A": np.linspace(10, 14, 30)}, index=dates)
    atr = wilder_atr(high, low, close, window=14)
    tr = []
    prev = np.nan
    for i in range(30):
        h, l, c = high["A"].iloc[i], low["A"].iloc[i], close["A"].iloc[i]
        tr.append(max(h - l, abs(h - prev), abs(l - prev)) if not np.isnan(prev) else np.nan)
        prev = c
    tr = np.array(tr)
    manual = np.full(30, np.nan)
    # first 14 valid TRs are rows 1..14 (row 0 has no previous close)
    manual[14] = np.nanmean(tr[1:15])
    for i in range(15, 30):
        manual[i] = (manual[i - 1] * 13 + tr[i]) / 14
    got = atr["A"].to_numpy()
    assert np.allclose(got[14:], manual[14:], rtol=1e-9, equal_nan=True)


def test_stop_bands_are_point_in_time():
    """Band at date t must not change when later data is altered."""
    frames = _synthetic_frames()
    symbols = sorted(frames)
    common = frames[symbols[0]].index
    p1 = RotationParams(dynamic_stops=True)
    s1, t1 = precompute_stop_bands(frames, common, symbols, p1)
    # Corrupt everything after day 100; earlier bands must be identical.
    frames2 = {s: f.copy() for s, f in frames.items()}
    for f in frames2.values():
        f.loc[f.index[100]:, ["high", "low", "close"]] *= 5.0
    common2 = frames2[symbols[0]].index
    s2, t2 = precompute_stop_bands(frames2, common2, symbols, p1)
    assert np.allclose(s1.iloc[:80].to_numpy(), s2.iloc[:80].to_numpy(), equal_nan=True)
    assert np.allclose(t1.iloc[:80].to_numpy(), t2.iloc[:80].to_numpy(), equal_nan=True)


def test_stop_bands_bounds_and_take_gap():
    """Bands stay inside [floor, cap] and take-profit clears stop-loss."""
    frames = _synthetic_frames()
    symbols = sorted(frames)
    common = frames[symbols[0]].index
    p = RotationParams(dynamic_stops=True)
    stop, take = precompute_stop_bands(frames, common, symbols, p)
    assert stop is not None and take is not None
    valid = stop.notna() & take.notna()
    assert float(stop[valid].min().min()) >= p.stops_floor - 1e-12
    assert float(stop[valid].max().max()) <= p.stops_cap + 1e-12
    assert float(take[valid].min().min()) >= p.stops_take_floor - 1e-12
    assert float(take[valid].max().max()) <= p.stops_take_cap + 1e-12
    ratio = (take[valid] / stop[valid]).to_numpy()
    assert np.nanmin(ratio) >= 1.15 - 1e-9


def test_disabled_default_returns_none():
    p = RotationParams()
    assert p.dynamic_stops is False
    frames = _synthetic_frames()
    symbols = sorted(frames)
    common = frames[symbols[0]].index
    stop, take = precompute_stop_bands(frames, common, symbols, p)
    assert stop is None and take is None


def test_band_at_falls_back_to_static_on_nan_or_missing():
    p = RotationParams(dynamic_stops=True, stop_loss_pct=0.07, take_profit_pct=0.12)
    frames = _synthetic_frames()
    symbols = sorted(frames)
    common = frames[symbols[0]].index
    stop, take = precompute_stop_bands(frames, common, symbols, p)
    # missing symbol -> static fallback
    assert band_at(type("P", (), {"stop_band": stop, "take_band": take})(), "999999.SH", common[0], 0.07, 0.12) == (0.07, 0.12)
    # valid symbol -> dynamic band in range
    s, t = band_at(type("P", (), {"stop_band": stop, "take_band": take})(), symbols[0], common[50], 0.07, 0.12)
    assert p.stops_floor <= s <= p.stops_cap
    assert p.stops_take_floor <= t <= p.stops_take_cap


def test_engine_runs_with_dynamic_stops_enabled():
    """The engine hook must run end-to-end and keep the static default intact."""
    frames = _synthetic_frames()
    p = RotationParams(
        dynamic_stops=True,
        enable_intraweek_stops=True,
        rebalance_weekday=None,
        rebalance_days=10,
        min_volume_days=20,
        defensive_hold_assets=(),
    )
    result = weekly_rotation_backtest(frames, p)
    assert result["summary"]["observations"] > 0
    # static default path still works and has no band panels
    p2 = RotationParams(enable_intraweek_stops=True)
    result2 = weekly_rotation_backtest(frames, p2)
    assert result2["summary"]["observations"] == result["summary"]["observations"]
