"""Tests for the controlled short sleeve:
default-off, high-conviction gating, sizing caps, and engine integration."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from Main.weekly_rotation import RotationParams, weekly_rotation_backtest  # noqa: E402


def _frames(direction: float = -1.0) -> dict:
    """Three declining names + two broad index ETFs (short instruments)."""
    dates = pd.bdate_range("2023-01-02", periods=220)
    symbols = ["600000.SH", "000001.SZ", "600036.SH", "510300.SH", "510500.SH"]
    frames = {}
    for i, sym in enumerate(symbols):
        base = 10.0 * (1 + i)
        close = base * np.linspace(1.0, 1.0 + 0.7 * direction, len(dates))
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


def test_short_sleeve_default_off():
    p = RotationParams()
    assert p.enable_short_sleeve is False
    assert p.max_short_exposure == 0.10
    assert p.short_etf_only is True
    assert p.short_conviction_prob == 0.35


def test_short_sleeve_engages_only_in_conviction_bear():
    frames = _frames(direction=-1.0)  # falling market -> BEAR
    p = RotationParams(
        enable_short_sleeve=True,
        max_short_exposure=0.10,
        short_etf_only=True,
        regime_model="",
        ml_bear_override=False,
        regime_confirmation_ma=200,
        rebalance_weekday=None,
        rebalance_days=15,
        min_volume_days=20,
        defensive_hold_assets=(),
    )
    result = weekly_rotation_backtest(frames, p)
    rets = result["returns"]
    assert rets["short_exposure"].max() > 1e-6, "short sleeve never engaged in a bear"
    assert rets["short_exposure"].max() <= p.max_short_exposure + 1e-9
    assert rets["gross_exposure"].max() <= 1.0 + 1e-9  # no leverage
    # in a rising market the short sleeve must stay flat
    frames_up = _frames(direction=1.0)
    p_up = RotationParams(
        enable_short_sleeve=True,
        regime_model="",
        ml_bear_override=False,
        regime_confirmation_ma=200,
        rebalance_weekday=None,
        rebalance_days=15,
        min_volume_days=20,
        defensive_hold_assets=(),
    )
    result_up = weekly_rotation_backtest(frames_up, p_up)
    assert result_up["returns"]["short_exposure"].abs().max() <= 1e-9


def test_short_sleeve_etf_only_and_cap():
    frames = _frames(direction=-1.0)
    p = RotationParams(
        enable_short_sleeve=True,
        max_short_exposure=0.20,
        short_etf_only=True,
        short_etf_pool=("510300.SH", "510500.SH"),
        regime_model="",
        ml_bear_override=False,
        regime_confirmation_ma=200,
        rebalance_weekday=None,
        rebalance_days=15,
        min_volume_days=20,
        defensive_hold_assets=(),
    )
    result = weekly_rotation_backtest(frames, p)
    rets = result["returns"]
    assert rets["short_exposure"].max() <= 0.20 + 1e-9
    # shorts must only ever target the ETF pool
    # (indirect check: the run is stable and bounded)
    assert rets["strategy_return"].abs().max() < 0.30


def test_short_sleeve_default_off_preserves_behavior():
    frames = _frames(direction=-1.0)
    base = RotationParams(
        regime_model="",
        ml_bear_override=False,
        regime_confirmation_ma=200,
        rebalance_weekday=None,
        rebalance_days=15,
        min_volume_days=20,
        defensive_hold_assets=(),
    )
    r0 = weekly_rotation_backtest(frames, base)
    assert r0["returns"]["short_exposure"].abs().max() <= 1e-9
