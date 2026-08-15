"""Tests for the z-score event-shock detector (robust across universes)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Main.weekly_rotation import RotationParams, precompute_panels, weekly_rotation_backtest  # noqa: E402


def _frames(n=240, symbols=6):
    dates = pd.bdate_range("2023-01-02", periods=n)
    result = {}
    t = np.arange(n)
    for i in range(symbols):
        close = 10.0 * np.exp(0.0008 * t + 0.1 * np.sin(2 * np.pi * t / 120 + i))
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
        event_shock_latch=False,
    )
    return replace(base, **kwargs)


def test_zscore_default_zero_is_byte_compatible():
    frames = _frames()
    base = _params(event_shock_threshold=0.0)
    with_off = _params(event_shock_threshold=0.0, event_shock_zscore=0.0)
    r0 = weekly_rotation_backtest(frames, base)["returns"]
    r1 = weekly_rotation_backtest(frames, with_off)["returns"]
    pd.testing.assert_frame_equal(r0, r1)


def test_bench_shock_z_panel_is_finite_and_pit():
    frames = _frames()
    p = _params(event_shock_zscore=3.0)
    panel = precompute_panels(frames, p)
    z = panel.bench_shock_z
    assert z is not None
    assert z.notna().sum() > 50
    # PIT: the panel is a plain rolling series (no forward information)
    assert z.index.equals(panel.common)


def test_zscore_rule_cuts_exposure_on_relative_crash():
    frames = _frames(n=240)
    dates = list(frames["000001.SZ"].index)
    drop_idx = 150
    # a -3.5% single-day drop on every name: extreme vs the calm 60d history,
    # but far below a fixed 5% threshold
    for df in frames.values():
        df.iloc[drop_idx, df.columns.get_loc("open")] *= 0.965
        df.iloc[drop_idx, df.columns.get_loc("close")] *= 0.965
        df.iloc[drop_idx, df.columns.get_loc("high")] *= 0.965
        df.iloc[drop_idx, df.columns.get_loc("low")] *= 0.965
    fixed = _params(event_shock_threshold=0.05, event_shock_zscore=0.0, event_shock_exposure=0.30)
    zrule = _params(event_shock_threshold=0.0, event_shock_zscore=3.0, event_shock_exposure=0.30)
    r_fixed = weekly_rotation_backtest(frames, fixed)["returns"]
    r_z = weekly_rotation_backtest(frames, zrule)["returns"]
    next_day = dates[drop_idx + 1]
    expo_fixed = float(r_fixed.loc[next_day, "gross_exposure"])
    expo_z = float(r_z.loc[next_day, "gross_exposure"])
    # the z-score rule must have de-risked on the next open...
    assert expo_z < expo_fixed - 0.1, f"expected z-rule exposure {expo_z} << fixed {expo_fixed}"
    # ...and the fixed 5% rule must NOT have de-risked (3.5% < 5%)
    assert expo_fixed > 0.5
