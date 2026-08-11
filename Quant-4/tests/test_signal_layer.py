"""Tests for the signal-layer expansion (MAX effect / Amihud illiquidity):
default-off, PIT construction, and engine integration with renormalization."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from Main.weekly_rotation import RotationParams, precompute_panels, weekly_rotation_backtest  # noqa: E402


def _synthetic_frames() -> dict:
    dates = pd.bdate_range("2023-01-02", periods=180)
    frames = {}
    for i, sym in enumerate(["600000.SH", "000001.SZ", "510300.SH"]):
        base = 10.0 * (1 + i)
        close = base * np.linspace(1.0, 1.6, len(dates))
        open_ = close * 0.999
        high = close * 1.01
        low = close * 0.99
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


def test_signal_layer_default_off():
    p = RotationParams()
    assert p.max_ret_weight == 0.0
    assert p.illiquidity_weight == 0.0
    panel = precompute_panels(_synthetic_frames(), p)
    assert panel.max_ret is None
    assert panel.illiquidity is None


def test_signal_panels_are_pit_and_finite():
    frames = _synthetic_frames()
    p = RotationParams(max_ret_weight=0.15, illiquidity_weight=0.10)
    panel = precompute_panels(frames, p)
    assert panel.max_ret is not None and panel.illiquidity is not None
    assert panel.max_ret.notna().sum().sum() > 0
    assert panel.illiquidity.notna().sum().sum() > 0
    # PIT: altering future closes must not change earlier panel rows
    frames2 = {s: f.copy() for s, f in frames.items()}
    for f in frames2.values():
        f.loc[f.index[100]:, "close"] *= 3.0
    panel2 = precompute_panels(frames2, p)
    assert np.allclose(panel.max_ret.iloc[:80].to_numpy(), panel2.max_ret.iloc[:80].to_numpy(), equal_nan=True)
    assert np.allclose(panel.illiquidity.iloc[:80].to_numpy(), panel2.illiquidity.iloc[:80].to_numpy(), equal_nan=True)


def test_signal_layer_runs_in_engine():
    frames = _synthetic_frames()
    base = RotationParams(
        rebalance_weekday=None,
        rebalance_days=10,
        min_volume_days=20,
        defensive_hold_assets=(),
        enable_intraweek_stops=True,
    )
    with_signals = RotationParams(
        rebalance_weekday=None,
        rebalance_days=10,
        min_volume_days=20,
        defensive_hold_assets=(),
        enable_intraweek_stops=True,
        max_ret_weight=0.15,
        illiquidity_weight=0.10,
        breakout_weight=0.10,
    )
    r0 = weekly_rotation_backtest(frames, base)
    r1 = weekly_rotation_backtest(frames, with_signals)
    assert r0["summary"]["observations"] == r1["summary"]["observations"]
    assert r1["summary"]["final_equity"] > 0
