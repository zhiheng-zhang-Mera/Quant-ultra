"""Tests for the minimum-turnover rebalance skip (drift gate)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Main.weekly_rotation import RotationParams, weekly_rotation_backtest  # noqa: E402


def _frames(n=220, symbols=6):
    dates = pd.bdate_range("2023-01-02", periods=n)
    result = {}
    t = np.arange(n)
    for i in range(symbols):
        # phase-shifted cycles so relative momentum crosses over repeatedly;
        # monotonic drift keeps every name tradeable throughout the window
        cycle = 0.35 * np.sin(2 * np.pi * t / 90 + i * 2 * np.pi / symbols)
        close = 10.0 * np.exp(0.0012 * t + cycle)
        result[f"00000{i + 1}.SZ"] = pd.DataFrame({
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.full(n, 10_000_000.0),
            "amount": close * 10_000_000.0,
        }, index=dates)
    return result


def _params(**kwargs):
    base = RotationParams(
        start_date="2023-01-02",
        regime_ma=20, regime_ma_fast=5, min_volume_days=20,
        trend_filter_long=0, min_adv=0, require_relative_strength=False,
        rebalance_days=10, rebalance_weekday=None, top_n=2,
        max_etf_positions=0, enable_board_lots=False,
        enable_intraweek_stops=True,
    )
    return replace(base, **kwargs)


def test_drift_skip_default_zero_is_byte_compatible():
    frames = _frames()
    base = _params()
    with_off = _params(rebalance_min_turnover=0.0)
    r0 = weekly_rotation_backtest(frames, base)["returns"]
    r1 = weekly_rotation_backtest(frames, with_off)["returns"]
    pd.testing.assert_frame_equal(r0, r1)


def test_drift_skip_reduces_turnover_and_cost():
    frames = _frames()
    base = _params()
    skip = _params(rebalance_min_turnover=0.30)
    r0 = weekly_rotation_backtest(frames, base)
    r1 = weekly_rotation_backtest(frames, skip)
    s0, s1 = r0["summary"], r1["summary"]
    # aggregate turnover and cost must not increase when the drift gate is active
    assert s1["total_cost_fraction"] <= s0["total_cost_fraction"] + 1e-12
    assert float(r1["returns"]["turnover"].sum()) <= float(r0["returns"]["turnover"].sum()) + 1e-12
    assert s1["closed_trades_count"] <= s0["closed_trades_count"]
    # and it must still be a valid, profitable-in-sample run
    assert s1["final_equity"] > 0
    assert s1["observations"] == s0["observations"]


def test_drift_skip_does_not_freeze_forever():
    """A drift gate must not stop the book from rebalancing altogether: with
    cyclical relative strength the active-rebalance count stays well above 1
    even under a 30% minimum-turnover gate."""
    frames = _frames(n=260)
    skip = _params(rebalance_min_turnover=0.30, hold_persistent=False)
    r = weekly_rotation_backtest(frames, skip)
    active = int((r["returns"]["turnover"] > 1e-9).sum())
    assert active > 1, "with a drift gate the book must still rebalance over 260 days"
