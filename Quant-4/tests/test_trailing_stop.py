"""Tests for the peak-based trailing stop (default-off, byte-compatible)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from Main.weekly_rotation import RotationParams, weekly_rotation_backtest  # noqa: E402


def _frames(n=240, symbols=6):
    dates = pd.bdate_range("2023-01-02", periods=n)
    result = {}
    t = np.arange(n)
    for i in range(symbols):
        close = 10.0 * np.exp(0.0015 * t + 0.25 * np.sin(2 * np.pi * t / 100 + i * 2 * np.pi / symbols))
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
        rebalance_days=15, rebalance_weekday=None, top_n=2,
        max_etf_positions=0, enable_board_lots=False,
        enable_intraweek_stops=True,
        stop_loss_pct=0.07,
        take_profit_pct=0.12,
    )
    return replace(base, **kwargs)


def test_trailing_stop_default_zero_is_byte_compatible():
    frames = _frames()
    base = _params()
    with_off = _params(trailing_stop_pct=0.0)
    r0 = weekly_rotation_backtest(frames, base)["returns"]
    r1 = weekly_rotation_backtest(frames, with_off)["returns"]
    pd.testing.assert_frame_equal(r0, r1)


def test_trailing_stop_lets_winners_run():
    """A name that rallies then pulls back by less than the trailing band must
    stay held; the same name would have been stopped by a fixed entry-based
    stop if it had dipped below the entry band after running up."""
    frames = _frames(n=300)
    # boost one name so it dominates the top-2 for the whole window
    base = 10.0 * np.exp(0.004 * np.arange(300) + 0.3 * np.sin(2 * np.pi * np.arange(300) / 120))
    frames["000001.SZ"] = pd.DataFrame({
        "open": base, "high": base * 1.01, "low": base * 0.99, "close": base,
        "volume": np.full(300, 10_000_000.0), "amount": base * 10_000_000.0,
    }, index=frames["000001.SZ"].index)
    fixed = _params()
    trailing = _params(trailing_stop_pct=0.12, take_profit_pct=0.0)
    r_fixed = weekly_rotation_backtest(frames, fixed)
    r_trail = weekly_rotation_backtest(frames, trailing)
    trades_fixed = r_fixed["closed_trades"]
    trades_trail = r_trail["closed_trades"]
    # the trailing book must have traded fewer distinct exits on the boosted name
    exits_fixed = int((trades_fixed["symbol"] == "000001.SZ").sum()) if len(trades_fixed) else 0
    exits_trail = int((trades_trail["symbol"] == "000001.SZ").sum()) if len(trades_trail) else 0
    assert exits_trail <= exits_fixed, f"trailing should not exit more often: {exits_trail} vs {exits_fixed}"
    # and it must still be a valid run
    assert r_trail["summary"]["final_equity"] > 0
    assert r_fixed["summary"]["observations"] == r_trail["summary"]["observations"]


def test_trailing_stop_still_caps_losses():
    """A name that collapses from its peak by more than the trailing band is
    stopped, so deep drawdowns cannot ride through the trailing mechanism."""
    frames = _frames(n=300)
    dates = list(frames["000001.SZ"].index)
    # the boosted name crashes -30% from its peak at day 200
    base = 10.0 * np.exp(0.003 * np.arange(300))
    base[200:] = base[200] * 0.70 * np.exp(-0.001 * np.arange(100))
    frames["000001.SZ"] = pd.DataFrame({
        "open": base, "high": base * 1.01, "low": base * 0.99, "close": base,
        "volume": np.full(300, 10_000_000.0), "amount": base * 10_000_000.0,
    }, index=frames["000001.SZ"].index)
    trailing = _params(trailing_stop_pct=0.10, take_profit_pct=0.0)
    r = weekly_rotation_backtest(frames, trailing)
    assert r["summary"]["final_equity"] > 0
    # the crash must not translate into a portfolio drawdown deeper than ~ the
    # trailing band plus noise (single position weight <= 0.5 with top_n=2)
    mdd = r["summary"]["max_drawdown"]
    assert mdd > -0.25, f"trailing stop failed to cap the crash: mdd={mdd:.2%}"
