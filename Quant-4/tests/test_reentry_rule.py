"""Tests for the earlier re-entry rule (direction 2, Round 19.5).

reentry_skip_confirmation_periods is default-off (byte-identical); when
enabled after N consecutive non-BULL rebalance periods, the BULL gate may
skip the long confirmation MA requirement while every other bull condition
(close>MA40, MA10>MA40, positive 20d momentum) holds.
"""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Main.weekly_rotation import RotationParams, _reentry_fast_ok, weekly_rotation_backtest  # noqa: E402


def _params(**kwargs):
    return replace(RotationParams(reentry_skip_confirmation_periods=2, ml_bear_override=True), **kwargs)


def test_reentry_off_by_default():
    assert RotationParams().reentry_skip_confirmation_periods == 0
    assert _reentry_fast_ok(RotationParams(), 5, False, 10.0, 9.0, 8.0, 0.05) is False


def test_reentry_requires_enough_nonbull_periods():
    p = _params()
    # only 1 non-bull period < required 2
    assert _reentry_fast_ok(p, 1, False, 10.0, 9.0, 8.0, 0.05) is False
    assert _reentry_fast_ok(p, 2, False, 10.0, 9.0, 8.0, 0.05) is True


def test_reentry_requires_full_fast_bull_conditions():
    p = _params()
    assert _reentry_fast_ok(p, 2, False, 10.0, 8.0, 9.0, 0.05) is False  # ma_fast <= ma_slow
    assert _reentry_fast_ok(p, 2, False, 10.0, 9.0, 8.0, -0.01) is False  # momentum <= 0
    assert _reentry_fast_ok(p, 2, False, 7.0, 9.0, 8.0, 0.05) is False    # close <= ma_slow
    assert _reentry_fast_ok(p, 2, True, 10.0, 9.0, 8.0, 0.05) is False    # risk-off latch held
    assert _reentry_fast_ok(_params(ml_bear_override=False), 2, False, 10.0, 9.0, 8.0, 0.05) is False


def test_reentry_handles_nan_inputs():
    p = _params()
    assert _reentry_fast_ok(p, 2, False, np.nan, 9.0, 8.0, 0.05) is False
    assert _reentry_fast_ok(p, 2, False, 10.0, None, 8.0, 0.05) is False


def _frames(n=240, symbols=6):
    dates = pd.bdate_range("2024-01-02", periods=n)
    result = {}
    # all symbols share a close path; the equal-weight benchmark therefore
    # follows it exactly (benchmark_exclude is empty)
    close_path = np.full(n, 10.0)
    close_path[:150] = 10.0 * np.exp(np.linspace(0, 0.12, 150))   # slow bull
    close_path[150:175] = close_path[149] * np.exp(np.linspace(0, -0.35, 25))  # crash
    close_path[175:] = close_path[174] * np.exp(np.linspace(0, 0.40, n - 175))  # strong recovery
    for i in range(symbols):
        close = close_path * (1.0 + 0.01 * i)
        result[f"00000{i + 1}.SZ"] = pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": np.full(n, 10_000_000.0),
            "amount": close * 10_000_000.0,
        }, index=dates)
    return result


def test_reentry_restores_bull_book_before_confirmation_ma():
    """After the benchmark crashes and then recovers above MA20 (regime_ma)
    with MA5>MA20 and positive momentum but still below the long confirmation
    MA, the re-entry rule must restore the BULL book while the strict gate
    stays NEUTRAL."""
    frames = _frames()
    base = RotationParams(
        regime_ma=20, regime_ma_fast=5, regime_confirmation_ma=60,
        min_volume_days=20, trend_filter_long=0, min_adv=0,
        require_relative_strength=False, rebalance_days=5, rebalance_weekday=None,
        top_n=2, max_etf_positions=0, enable_board_lots=False,
        event_shock_threshold=0.0, ml_bear_override=True,
        reentry_skip_confirmation_periods=1,
    )
    res = weekly_rotation_backtest(frames, base, regime_detector_kwargs={"bull_threshold": 0.55})
    regimes = res["regimes"]
    # find a rebalance after the recovery (close > MA20, close < MA60) where the
    # strict gate says NEUTRAL but the re-entry rule says BULL
    strict_off = weekly_rotation_backtest(
        frames, replace(base, reentry_skip_confirmation_periods=0),
        regime_detector_kwargs={"bull_threshold": 0.55},
    )["regimes"]
    reentries = regimes[(regimes["advice_zh"].astype(str).str.contains("提前再入场"))]
    assert len(reentries) >= 1, "re-entry rule must fire after the recovery"
    # the same dates must be NEUTRAL (not BULL) under the strict gate
    for d in reentries.index:
        strict = strict_off.loc[d]
        assert strict["regime"] != "BULL", f"strict gate must not be BULL at {d}"
        assert regimes.loc[d, "regime"] == "BULL", f"re-entry must restore BULL at {d}"
