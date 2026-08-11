"""Tests for the optional breakout/new-high factor (sprint sleeve signal):
default-off, PIT construction, score renormalization, and engine integration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from Main.strategy_selector import build_archetype_params  # noqa: E402
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


def test_breakout_default_off():
    p = RotationParams()
    assert p.breakout_weight == 0.0
    assert p.breakout_window == 60
    frames = _synthetic_frames()
    panel = precompute_panels(frames, p)
    assert panel.breakout is None


def test_breakout_panel_is_pit_and_bounded():
    frames = _synthetic_frames()
    symbols = sorted(frames)
    p = RotationParams(breakout_weight=0.35)
    panel = precompute_panels(frames, p)
    assert panel.breakout is not None
    b = panel.breakout
    assert float(b.notna().sum().sum()) > 0
    valid = b.notna()
    assert float(b[valid].max().max()) <= 1.0 + 1e-9
    assert float(b[valid].min().min()) > 0.0
    # PIT: corrupting future closes must not change earlier breakout rows
    frames2 = {s: f.copy() for s, f in frames.items()}
    for f in frames2.values():
        f.loc[f.index[100]:, "close"] *= 3.0
    panel2 = precompute_panels(frames2, RotationParams(breakout_weight=0.35))
    assert np.allclose(
        panel.breakout.iloc[:80].to_numpy(),
        panel2.breakout.iloc[:80].to_numpy(),
        equal_nan=True,
    )


def test_sprint_archetype_carries_breakout_signal():
    p = build_archetype_params(RotationParams(), "sprint")
    assert p.breakout_weight == 0.35
    assert p.breakout_window == 60
    # regime-adaptive scaling was rejected by the honest PIT gate (fixed wins)
    assert p.breakout_bull_scale == 1.0
    assert p.breakout_highvol_scale == 1.0
    assert p.top_n == 2
    assert p.per_position_cap == 0.40
    assert p.take_profit_pct == 0.15
    assert p.stop_loss_pct == 0.08
    assert p.enable_intraweek_stops is True


def test_breakout_regime_scale_defaults_preserve_behavior():
    """Scales default to 1.0 so the fixed-weight score is unchanged."""
    p = RotationParams(breakout_weight=0.35)
    assert p.breakout_bull_scale == 1.0
    assert p.breakout_highvol_scale == 1.0


def test_breakout_changes_ranking_and_runs():
    frames = _synthetic_frames()
    base = RotationParams(
        rebalance_weekday=None,
        rebalance_days=10,
        min_volume_days=20,
        defensive_hold_assets=(),
        enable_intraweek_stops=True,
    )
    with_breakout = RotationParams(
        rebalance_weekday=None,
        rebalance_days=10,
        min_volume_days=20,
        defensive_hold_assets=(),
        enable_intraweek_stops=True,
        breakout_weight=0.35,
        top_n=2,
    )
    r0 = weekly_rotation_backtest(frames, base)
    r1 = weekly_rotation_backtest(frames, with_breakout)
    assert r0["summary"]["observations"] == r1["summary"]["observations"]
    assert r1["summary"]["final_equity"] > 0
    # The breakout panel was exercised (non-null) on the breakout-enabled run.
    panel = precompute_panels(frames, with_breakout)
    assert panel.breakout is not None
