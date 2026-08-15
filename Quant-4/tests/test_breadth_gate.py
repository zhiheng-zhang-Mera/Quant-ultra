"""Tests for the market-breadth confirmation gate (default-off)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Main.weekly_rotation import RotationParams, weekly_rotation_backtest  # noqa: E402


def _frames(n=260, symbols=10):
    dates = pd.bdate_range("2023-01-02", periods=n)
    result = {}
    t = np.arange(n)
    for i in range(symbols):
        close = 10.0 * np.exp(0.0008 * t + 0.15 * np.sin(2 * np.pi * t / 100 + i * 2 * np.pi / symbols))
        result[f"00000{i + 1}.SZ"] = pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99, "close": close,
            "volume": np.full(n, 10_000_000.0), "amount": close * 10_000_000.0,
        }, index=dates)
    return result


def _params(**kwargs):
    base = RotationParams(
        start_date="2023-01-02",
        regime_ma=20, regime_ma_fast=5, min_volume_days=20,
        trend_filter_long=0, min_adv=0, require_relative_strength=False,
        rebalance_days=10, rebalance_weekday=None, top_n=3,
        max_etf_positions=0, enable_board_lots=False,
    )
    return replace(base, **kwargs)


def test_breadth_gate_default_zero_is_byte_compatible():
    frames = _frames()
    base = _params()
    with_off = _params(min_breadth_for_buys=0.0)
    r0 = weekly_rotation_backtest(frames, base)["returns"]
    r1 = weekly_rotation_backtest(frames, with_off)["returns"]
    pd.testing.assert_frame_equal(r0, r1)


def test_breadth_gate_de_risks_on_narrow_market():
    """With a very high breadth requirement the strategy must hold mostly cash,
    because the synthetic market is dominated by a sine-cycle cross-section
    where the above-60d-MA share oscillates well below 0.99."""
    frames = _frames()
    base = _params()
    gated = _params(min_breadth_for_buys=0.99)
    r0 = weekly_rotation_backtest(frames, base)["returns"]
    r1 = weekly_rotation_backtest(frames, gated)["returns"]
    assert float(r1["gross_exposure"].mean()) < float(r0["gross_exposure"].mean()) - 0.2
    # and the gate cannot produce leverage or negative exposure
    assert float(r1["gross_exposure"].max()) <= 1.0 + 1e-9
    assert float(r1["gross_exposure"].min()) >= -1e-9


def test_breadth_gate_still_participates_when_breadth_is_high():
    """A moderate threshold (0.0 < t < 0.2) must still allow most weeks, so the
    book is not permanently parked in cash."""
    frames = _frames()
    gated = _params(min_breadth_for_buys=0.10)
    r = weekly_rotation_backtest(frames, gated)["returns"]
    assert float(r["gross_exposure"].mean()) > 0.3
